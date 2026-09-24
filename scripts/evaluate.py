"""Evaluate an experiment's saved predictions.

    python scripts/evaluate.py --exp experiments/exp_004_lilt --partitions validation
    python scripts/evaluate.py --exp experiments/exp_004_lilt --partitions test   # logged

Results are written to <exp>/<out-subdir>/ (default: eval_<timestamp>) so an
existing experiment's metrics are never overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import load_yaml, resolve  # noqa: E402
from src.evaluation.evaluate import evaluate_partition  # noqa: E402
from src.training.tracking import log_test_evaluation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--partitions", nargs="+", default=["validation"], choices=["train", "validation", "test"])
    ap.add_argument("--out-subdir", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    exp = resolve(args.exp)
    pred_dir = exp / "predictions"
    if not pred_dir.exists():
        sys.exit(f"no predictions in {pred_dir}")
    out = exp / (args.out_subdir or f"eval_{dt.datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=False)
    dcfg = load_yaml("configs/dataset.yaml")
    for part in args.partitions:
        m = evaluate_partition(pred_dir, part, dcfg["truth_file"], out)
        if part == "test":
            log_test_evaluation(exp, f"scripts/evaluate.py -> {out.name}; {args.note}")
        print(part, json.dumps({f: v["all"]["accuracy"] for f, v in m["fields"].items()}))
        print(part, "subjects complete-row", m["subjects"]["all"]["complete_row_accuracy"],
              "| error categories", m["error_categories"])


if __name__ == "__main__":
    main()
