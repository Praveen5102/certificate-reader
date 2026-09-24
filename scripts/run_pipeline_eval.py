"""Evaluate the full current pipeline (OCR cache + tagger + table readers) as a new experiment.

    python scripts/run_pipeline_eval.py --name pipeline_v2 --model-dir experiments/exp_004_lilt/model
    python scripts/run_pipeline_eval.py ... --final-test        # also the held-out test split (logged)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import doc_id, load_yaml, read_json, write_json  # noqa: E402
from src.evaluation.evaluate import evaluate_partition  # noqa: E402
from src.inference.pipeline import CertificateExtractor  # noqa: E402
from src.training.tracking import (dataset_fingerprint, environment, log_test_evaluation,  # noqa: E402
                                   new_experiment_dir, write_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tagger", default="model", choices=["model", "rule_baseline"])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    dcfg, ocfg = load_yaml("configs/dataset.yaml"), load_yaml("configs/ocr.yaml")
    exp = new_experiment_dir(args.name)
    ex = CertificateExtractor(tagger=args.tagger, model_dir=args.model_dir)
    split = read_json("data/split.json")["partitions"]
    manifest = {d["file"]: d for d in read_json("data/dataset_manifest.json")["documents"]}
    parts = ["train", "validation"] + (["test"] if args.final_test else [])
    t0, n = time.time(), 0
    for part in parts:
        for f in split[part]:
            res = ex.extract_from_ocr(f, [read_json(p) for p in manifest[f]["ocr_files"]])
            write_json(exp / "predictions" / f"{doc_id(f)}.json", {"flat": res.to_flat(), "full": res.model_dump()})
            n += 1
    for part in parts:
        evaluate_partition(exp / "predictions", part, dcfg["truth_file"], exp)
        if part == "test":
            log_test_evaluation(exp, args.note or "full pipeline evaluation")
    write_run(exp, {"kind": "pipeline_evaluation", "model": args.tagger, "model_dir": args.model_dir,
                    "dataset": dataset_fingerprint(dcfg), "ocr": {"engine": ocfg["engine"], "config": ocfg},
                    "inference_config": load_yaml("configs/inference.yaml"), "environment": environment(),
                    "partitions_evaluated": parts, "inference_time_s_per_doc": round((time.time() - t0) / max(1, n), 2),
                    "notes": args.note})
    print("done:", exp)


if __name__ == "__main__":
    main()
