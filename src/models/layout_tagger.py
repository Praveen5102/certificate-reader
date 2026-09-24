"""Fine-tuned layout model as a tagger for the inference pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..annotation.anchors import TemplateIndex
from ..annotation.layout import PageWords
from ..inference.table import Cell
from .features import ID2LABEL, Encoder, PageInput, collate


def _join_parts(model_dir: Path) -> None:
    """Deployment export (scripts/export_model.py): fp16 weights split into
    <=90 MB parts so they fit in git. Join them once into model.safetensors
    and verify the SHA-256 before use."""
    import hashlib
    manifest = json.loads((model_dir / "parts.json").read_text(encoding="utf-8"))
    data = b"".join((model_dir / p).read_bytes() for p in manifest["parts"])
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise RuntimeError("model parts are corrupt or incomplete (SHA-256 mismatch)")
    tmp = model_dir / "model.safetensors.tmp"
    tmp.write_bytes(data)
    tmp.replace(model_dir / "model.safetensors")


def load_model(model_dir: str | Path):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists() and (model_dir / "parts.json").exists():
        _join_parts(model_dir)
    meta = json.loads((model_dir / "certml_meta.json").read_text(encoding="utf-8"))
    # RoBERTa BPE needs add_prefix_space for pre-split words (LiLT)
    kwargs = {"add_prefix_space": True} if meta["family"] == "lilt" else {}
    tok = AutoTokenizer.from_pretrained(model_dir, **kwargs)
    # float32 compute on CPU even when the weights are stored as float16
    model = AutoModelForTokenClassification.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    return model, tok, meta


class LayoutTagger:
    def __init__(self, model_dir: str | Path, batch_size: int = 4):
        self.model, self.tok, self.meta = load_model(model_dir)
        self.name = f"{self.meta['family']}:{Path(model_dir).parent.name}"
        self.enc = Encoder(self.tok, self.meta["family"], self.meta["max_seq_length"], self.meta["window_stride"],
                           self.meta.get("bbox_scale", 1000))
        self.batch_size = batch_size

    @torch.no_grad()
    def word_probs(self, page: PageInput) -> list[torch.Tensor | None]:
        windows = self.enc.encode(page)
        sums: list[torch.Tensor | None] = [None] * len(page.words)
        counts = [0] * len(page.words)
        for i in range(0, len(windows), self.batch_size):
            chunk = windows[i:i + self.batch_size]
            batch = collate(chunk, self.tok.pad_token_id)
            batch.pop("labels")
            logits = self.model(**batch).logits
            for b, w in enumerate(chunk):
                for t, (wid, first) in enumerate(zip(w["word_ids"], w["first"])):
                    if wid is None or not first:
                        continue
                    sums[wid] = logits[b, t] if sums[wid] is None else sums[wid] + logits[b, t]
                    counts[wid] += 1
        return [None if s is None else torch.softmax(s / counts[k], dim=-1) for k, s in enumerate(sums)]

    def tag(self, pages: list[PageWords], tmpls: dict[int, TemplateIndex]) -> dict[str, list[Cell]]:
        out: dict[str, list[Cell]] = {}
        for p in pages:
            if not p.words:
                continue
            pin = PageInput(doc="", page=p.page, words=[w.text for w in p.words], boxes=[list(w.bbox) for w in p.words],
                            width=p.width, height=p.height)
            probs = self.word_probs(pin)
            spans = decode_spans(p, probs)
            for label, words, conf in spans:
                out.setdefault(label, []).append(Cell(words, label, conf))
        return out


def decode_spans(page: PageWords, probs) -> list[tuple[str, list, float]]:
    """BIO decoding in reading order of OCR lines. A span continues with I-X
    words (or B-X words of single-word labels never merge). Confidence = min
    over span words of (predicted-label probability x OCR word confidence)."""
    order = sorted(range(len(page.words)), key=lambda i: (page.words[i].line_id, page.words[i].bbox[0]))
    spans, cur, cur_label = [], [], None
    single = {"MAX_MARKS", "MARKS", "TOTAL_MARKS", "HALL_TICKET", "CGPA"}

    def close():
        nonlocal cur, cur_label
        if cur:
            words = [page.words[i] for i, _ in cur]
            conf = min(pr * page.words[i].conf for i, pr in cur)
            spans.append((cur_label, words, round(float(conf), 4)))
        cur, cur_label = [], None

    prev_line = None
    for i in order:
        pr = probs[i]
        if pr is None:
            close()
            continue
        k = int(pr.argmax())
        lab = ID2LABEL[k]
        p = float(pr[k])
        w = page.words[i]
        if lab == "O":
            close()
        elif lab.startswith("B-") or cur_label != lab[2:] or lab[2:] in single or \
                (prev_line is not None and w.line_id != prev_line and lab[2:] in {"SUBJECT"}):
            close()
            cur_label, cur = lab[2:], [(i, p)]
        else:
            cur.append((i, p))
        prev_line = w.line_id
    close()
    return spans
