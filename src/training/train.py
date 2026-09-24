"""Fine-tuning loop for word-level BIO token classification (CPU-friendly).

A plain PyTorch loop rather than HF Trainer: the dataset is ~100 pages, the
loop is short and fully visible, and it avoids Trainer's dependency on
`datasets`/pandas (blocked on this machine). Model selection uses the
VALIDATION split only; the test split is never loaded here.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..annotation.align import IGNORE
from ..common.io import read_json, write_json
from ..models.features import ID2LABEL, LABEL2ID, LABELS, Encoder, collate, jitter_boxes, load_bio_pages, normalize_box


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_partition_pages(partition: str, include_notes: bool, augmented_copies: int = 0) -> list:
    split = read_json("data/split.json")
    if partition == "test":
        raise RuntimeError("the training code must never load the test partition")
    pages = []
    for f in split["partitions"][partition]:
        p = Path("annotations/bio") / f"{Path(f).stem}.json"
        if p.exists():
            pages += load_bio_pages(read_json(p), include_notes)
        if partition == "train":   # offline-augmented copies (scripts/augment_pages.py), train only
            for k in range(augmented_copies):
                a = Path("annotations/bio_augmented") / f"{Path(f).stem}_aug{k}.json"
                if a.exists():
                    pages += load_bio_pages(read_json(a), include_notes)
    return pages


def build_model(model_cfg: dict):
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    kwargs = {"add_prefix_space": True} if model_cfg["family"] == "lilt" else {}
    tok = AutoTokenizer.from_pretrained(model_cfg["base_checkpoint"], **kwargs)
    model = AutoModelForTokenClassification.from_pretrained(
        model_cfg["base_checkpoint"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    base = getattr(model, model.base_model_prefix)
    if model_cfg.get("freeze_embeddings"):
        for name, prm in base.named_parameters():
            if "embeddings" in name:
                prm.requires_grad = False
    n_freeze = int(model_cfg.get("freeze_lower_layers", 0))
    if n_freeze:
        for layer in base.encoder.layer[:n_freeze]:
            for prm in layer.parameters():
                prm.requires_grad = False
    return model, tok


def encode_pages(enc: Encoder, pages, aug: dict | None = None, rng: random.Random | None = None) -> list[dict]:
    out = []
    for p in pages:
        boxes = None
        if aug and rng and rng.random() < aug.get("probability", 0):
            boxes = jitter_boxes([normalize_box(b, p.width, p.height, enc.scale) for b in p.boxes],
                                 aug["bbox_jitter"], aug["page_scale"], rng, enc.scale)
        out += [w for w in enc.encode(p, boxes) if any(l != -100 for l in w["labels"])]
    return out


@torch.no_grad()
def evaluate_words(model, enc: Encoder, pages, batch_size: int) -> dict:
    """Word-level micro P/R/F1 over entity labels (IGNORE words excluded)."""
    model.eval()
    tp = fp = fn = 0
    per = {}
    for p in pages:
        wins = enc.encode(p)
        sums = [None] * len(p.words)
        for i in range(0, len(wins), batch_size):
            chunk = wins[i:i + batch_size]
            b = collate(chunk, enc.tok.pad_token_id)
            b.pop("labels")
            logits = model(**b).logits
            for bi, w in enumerate(chunk):
                for t, (wid, first) in enumerate(zip(w["word_ids"], w["first"])):
                    if wid is not None and first:
                        sums[wid] = logits[bi, t] if sums[wid] is None else sums[wid] + logits[bi, t]
        for wid, s in enumerate(sums):
            gold = p.labels[wid]
            if s is None or gold == IGNORE:
                continue
            pred = ID2LABEL[int(s.argmax())]
            g, q = gold[2:] if gold != "O" else "O", pred[2:] if pred != "O" else "O"
            if q != "O" and q == g:
                tp += 1
                per.setdefault(g, [0, 0, 0])[0] += 1
            else:
                if q != "O":
                    fp += 1
                    per.setdefault(q, [0, 0, 0])[1] += 1
                if g != "O":
                    fn += 1
                    per.setdefault(g, [0, 0, 0])[2] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_label = {k: {"tp": v[0], "fp": v[1], "fn": v[2],
                     "f1": round(2 * v[0] / (2 * v[0] + v[1] + v[2]), 4) if v[0] else 0.0} for k, v in per.items()}
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "per_label": per_label}


def train(model_cfg: dict, train_cfg: dict, out_dir: Path, max_steps: int | None = None,
          limit_docs: int | None = None) -> dict:
    set_seed(train_cfg["seed"])
    torch.set_num_threads(int(train_cfg.get("num_threads", 4)))
    model, tok = build_model(model_cfg)
    enc = Encoder(tok, model_cfg["family"], model_cfg["max_seq_length"], model_cfg["window_stride"],
                  model_cfg.get("bbox_scale", 1000))
    train_pages = load_partition_pages("train", train_cfg.get("include_notes_pages", True),
                                       int((train_cfg.get("augmentation") or {}).get("offline_augmented_copies", 0)))
    val_pages = load_partition_pages("validation", train_cfg.get("include_notes_pages", True))
    if limit_docs:
        train_pages, val_pages = train_pages[:limit_docs], val_pages[:max(1, limit_docs // 2)]
    rng = random.Random(train_cfg["seed"])

    weights = torch.ones(len(LABELS))
    weights[LABEL2ID["O"]] = float(train_cfg.get("o_class_weight", 1.0))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    bs, acc = int(train_cfg["batch_size"]), int(train_cfg["gradient_accumulation_steps"])
    n_windows = len(encode_pages(enc, train_pages))
    steps_per_epoch = math.ceil(n_windows / (bs * acc))
    total = max_steps or steps_per_epoch * int(train_cfg["epochs"])
    warm = int(total * float(train_cfg["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    history, best, bad_epochs, step = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        windows = encode_pages(enc, train_pages, train_cfg.get("augmentation"), rng)
        rng.shuffle(windows)
        run_loss, nb = 0.0, 0
        for i in range(0, len(windows), bs):
            batch = collate(windows[i:i + bs], tok.pad_token_id)
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) / acc
            loss.backward()
            run_loss += loss.item() * acc
            nb += 1
            if nb % acc == 0:
                torch.nn.utils.clip_grad_norm_(params, float(train_cfg["max_grad_norm"]))
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if max_steps and step >= max_steps:
                    break
        val = evaluate_words(model, enc, val_pages, bs)
        rec = {"epoch": epoch + 1, "step": step, "train_loss": round(run_loss / max(1, nb), 4),
               "val_precision": val["precision"], "val_recall": val["recall"], "val_f1": val["f1"],
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["f1"] > best:
            best, bad_epochs = val["f1"], 0
            save(model, tok, model_cfg, out_dir / "model")
            write_json(out_dir / "best_validation_word_metrics.json", {"epoch": epoch + 1, **val})
        else:
            bad_epochs += 1
            if bad_epochs >= int(train_cfg["early_stopping_patience"]):
                break
        if max_steps and step >= max_steps:
            break
    return {"history": history, "best_validation_entity_f1": best, "train_windows": n_windows,
            "train_pages": len(train_pages), "validation_pages": len(val_pages),
            "steps": step, "training_time_s": round(time.time() - t0, 1)}


def save(model, tok, model_cfg: dict, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    (d / "certml_meta.json").write_text(json.dumps({
        "family": model_cfg["family"], "base_checkpoint": model_cfg["base_checkpoint"],
        "max_seq_length": model_cfg["max_seq_length"], "window_stride": model_cfg["window_stride"],
        "bbox_scale": model_cfg.get("bbox_scale", 1000), "labels": LABELS}, indent=2), encoding="utf-8")
