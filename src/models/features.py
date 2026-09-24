"""Words + boxes (+ labels) -> model input windows. Shared by training and inference."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..annotation.align import IGNORE, bio_label_list

LABELS = bio_label_list()
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class PageInput:
    doc: str
    page: int
    words: list[str]
    boxes: list[list[int]]          # pixel boxes
    width: int
    height: int
    labels: list[str] | None = None  # word labels incl. IGNORE


def normalize_box(b, w, h, scale=1000) -> list[int]:
    x0, y0, x1, y1 = b
    f = lambda v, d: max(0, min(scale, int(round(scale * v / max(1, d)))))  # noqa: E731
    return [f(x0, w), f(y0, h), max(f(x0, w), f(x1, w)), max(f(y0, h), f(y1, h))]


def jitter_boxes(boxes: list[list[int]], jitter: float, scale_amt: float, rng: random.Random, s=1000):
    """Layout augmentation: small independent box jitter + global page scale.
    Text and word order are untouched."""
    k = 1.0 + rng.uniform(-scale_amt, scale_amt)
    out = []
    for x0, y0, x1, y1 in boxes:
        j = lambda: rng.uniform(-jitter, jitter) * s  # noqa: E731
        nb = [x0 * k + j(), y0 * k + j(), x1 * k + j(), y1 * k + j()]
        nb = [max(0, min(s, int(round(v)))) for v in nb]
        out.append([nb[0], nb[1], max(nb[0], nb[2]), max(nb[1], nb[3])])
    return out


class Encoder:
    """Tokenizes one page into overlapping windows of <= max_len tokens."""

    def __init__(self, tokenizer, family: str, max_len: int = 512, stride: int = 128, scale: int = 1000):
        self.tok = tokenizer
        self.family = family
        self.max_len = max_len
        self.stride = stride
        self.scale = scale

    def encode(self, page: PageInput, norm_boxes: list[list[int]] | None = None) -> list[dict]:
        boxes = norm_boxes or [normalize_box(b, page.width, page.height, self.scale) for b in page.boxes]
        words = [w if w.strip() else "_" for w in page.words]
        # The LiLT checkpoint ships a LayoutLMv3 tokenizer, which takes boxes directly.
        if "LayoutLMv3" in type(self.tok).__name__:
            enc = self.tok(words, boxes=boxes, truncation=False, add_special_tokens=False)
        else:
            enc = self.tok(words, is_split_into_words=True, truncation=False, add_special_tokens=False)
        ids = enc["input_ids"]
        word_ids = enc.word_ids()
        tok_boxes = [boxes[w] if w is not None else [0, 0, 0, 0] for w in word_ids]
        # label only the first sub-token of each word
        tok_labels, first_of_word, prev = [], [], None
        for wid in word_ids:
            is_first = wid is not None and wid != prev
            first_of_word.append(is_first)
            if page.labels is None or not is_first:
                tok_labels.append(-100)
            else:
                lab = page.labels[wid]
                tok_labels.append(-100 if lab == IGNORE else LABEL2ID[lab])
            prev = wid
        body = self.max_len - 2
        windows, start = [], 0
        while True:
            end = min(start + body, len(ids))
            windows.append(self._window(ids[start:end], tok_boxes[start:end], tok_labels[start:end],
                                        word_ids[start:end], first_of_word[start:end]))
            if end >= len(ids):
                break
            start = end - self.stride
        return windows

    def _window(self, ids, boxes, labels, word_ids, first):
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        return {
            "input_ids": [cls] + ids + [sep],
            "bbox": [[0, 0, 0, 0]] + boxes + [[self.scale] * 4],
            "labels": [-100] + labels + [-100],
            "word_ids": [None] + list(word_ids) + [None],
            "first": [False] + list(first) + [False],
        }


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "bbox": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["bbox"].append(b["bbox"] + [[0, 0, 0, 0]] * pad)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def load_bio_pages(bio_doc: dict, include_notes: bool = True) -> list[PageInput]:
    pages = []
    for p in bio_doc["pages"]:
        if p["role"] == "notes" and not include_notes:
            continue
        pages.append(PageInput(doc=bio_doc["file"], page=p["page"], words=[w["text"] for w in p["words"]],
                               boxes=[w["bbox"] for w in p["words"]], width=p["width"], height=p["height"],
                               labels=[w["label"] for w in p["words"]]))
    return pages
