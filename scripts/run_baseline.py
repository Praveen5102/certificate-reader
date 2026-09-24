"""exp_001: rule baseline on cached OCR, evaluated per partition.

Usage:
    python scripts/run_baseline.py                      # train + validation
    python scripts/run_baseline.py --include-test       # also the held-out test split (logged)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import doc_id, load_yaml, read_json, read_jsonl, write_json  # noqa: E402
from src.evaluation.evaluate import evaluate_partition  # noqa: E402
from src.inference.pipeline import CertificateExtractor  # noqa: E402
from src.training.tracking import (dataset_fingerprint, environment, log_test_evaluation,  # noqa: E402
                                   new_experiment_dir, write_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--name", default="rule_baseline")
    args = ap.parse_args()
    dcfg = load_yaml("configs/dataset.yaml")
    exp = new_experiment_dir(args.name)
    pred_dir = exp / "predictions"
    ex = CertificateExtractor(tagger="rule_baseline")
    manifest = read_json("data/dataset_manifest.json")["documents"]
    t0 = time.time()
    for d in manifest:  # predict every PDF (incl. non-target / untruthed) for the classification report
        res = ex.extract_from_ocr(d["file"], [read_json(p) for p in d["ocr_files"]])
        write_json(pred_dir / f"{doc_id(d['file'])}.json", {"flat": res.to_flat(), "full": res.model_dump()})
    elapsed = time.time() - t0
    # document classification over every PDF (target, non-target and untruthed)
    truth = {r["file"]: r for r in read_jsonl(dcfg["truth_file"])}
    cls_rows = []
    for d in manifest:
        full = read_json(pred_dir / f"{doc_id(d['file'])}.json")["full"]
        t = truth.get(d["file"])
        tt = t["document_type"] if t else None
        cls_rows.append({"file": d["file"], "truth_document_type": tt, "predicted": full["document_type"],
                         "score": full.get("document_type_score"),
                         "agrees": None if tt is None else ((tt == "Intermediate certificate")
                                                            == (full["document_type"] == "intermediate_certificate"))})
    judged = [r for r in cls_rows if r["agrees"] is not None]
    write_json(exp / "classification_report.json", {
        "documents": len(cls_rows), "with_truth": len(judged), "agree": sum(r["agrees"] for r in judged),
        "non_target_truth": [r for r in cls_rows if r["truth_document_type"] not in (None, "Intermediate certificate")],
        "no_truth": [r for r in cls_rows if r["truth_document_type"] is None],
        "disagreements": [r for r in judged if not r["agrees"]], "rows": cls_rows})
    parts =["train", "validation"] + (["test"] if args.include_test else [])
    metrics = {}
    for part in parts:
        metrics[part] = evaluate_partition(pred_dir, part, dcfg["truth_file"], exp)
        if part == "test":
            log_test_evaluation(exp, "rule baseline (no training, no tuning on test)")
    ocfg = load_yaml("configs/ocr.yaml")
    write_run(exp, {
        "kind": "baseline", "model": "rule_baseline_v1", "base_checkpoint": None,
        "dataset": dataset_fingerprint(dcfg), "random_seed": None,
        "hyperparameters": None, "ocr": {"engine": ocfg["engine"], "config": ocfg},
        "fields_config": load_yaml("configs/fields.yaml"),
        "partitions_evaluated": parts, "prediction_time_s": round(elapsed, 1),
        "environment": environment(),
        "notes": "Re-implementation of an anchor-phrase OCR/rule baseline; the original baseline code was not "
                 "supplied. Rules were designed on train/validation documents only.",
        "metrics_files": [f"metrics_{p}.json" for p in parts],
    })
    print(f"{exp.name}: wrote predictions for {len(manifest)} PDFs in {elapsed:.0f}s")
    for part, m in metrics.items():
        row = {f: (m["fields"][f]["all"]["accuracy"], m["fields"][f]["all"]["n_annotated"]) for f in m["fields"]}
        print(part, row)
        print(part, "subjects", {k: m["subjects"]["all"][k] for k in ("truth_rows", "name_accuracy", "paper_accuracy",
                                                                       "marks_accuracy", "maximum_accuracy",
                                                                       "complete_row_accuracy", "row_precision")})


if __name__ == "__main__":
    main()
