"""Fine-tune the layout model, then evaluate the full pipeline on train/validation.

    python scripts/train.py --name lilt                 # full run -> experiments/exp_NNN_lilt/
    python scripts/train.py --smoke --out <scratch>     # 2-step code-path check; not an experiment

The test partition is evaluated only with --final-test (logged in
experiments/test_evaluations.log); never use it to choose settings.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import doc_id, load_yaml, read_json, write_json  # noqa: E402
from src.training.tracking import (dataset_fingerprint, environment, log_test_evaluation,  # noqa: E402
                                   new_experiment_dir, write_run)
from src.training.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lilt")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--smoke", action="store_true", help="2 optimizer steps on 4 pages; verifies the code path")
    ap.add_argument("--out", help="output dir for --smoke (must not be under experiments/)")
    ap.add_argument("--final-test", action="store_true", help="also evaluate on the held-out test split (logged)")
    args = ap.parse_args()

    mcfg, tcfg = load_yaml(args.model_config), load_yaml(args.training_config)
    dcfg, ocfg = load_yaml("configs/dataset.yaml"), load_yaml("configs/ocr.yaml")
    if args.smoke:
        if not args.out:
            sys.exit("--smoke requires --out")
        exp = Path(args.out)
        exp.mkdir(parents=True, exist_ok=True)
        tcfg = {**tcfg, "epochs": 1, "gradient_accumulation_steps": 1, "batch_size": 2}
        result = train(mcfg, tcfg, exp, max_steps=2, limit_docs=4)
    else:
        exp = new_experiment_dir(args.name)
        result = train(mcfg, tcfg, exp)

    # full-pipeline predictions with the selected checkpoint
    from src.evaluation.evaluate import evaluate_partition
    from src.inference.pipeline import CertificateExtractor
    ex = CertificateExtractor(tagger="model", model_dir=str(exp / "model"))
    split = read_json("data/split.json")
    manifest = {d["file"]: d for d in read_json("data/dataset_manifest.json")["documents"]}
    parts = ["validation"] if args.smoke else ["train", "validation"] + (["test"] if args.final_test else [])
    t0 = time.time()
    n_pred = 0
    for part in parts:
        files = split["partitions"][part][: (2 if args.smoke else None)]
        for f in files:
            res = ex.extract_from_ocr(f, [read_json(p) for p in manifest[f]["ocr_files"]])
            write_json(exp / "predictions" / f"{doc_id(f)}.json", {"flat": res.to_flat(), "full": res.model_dump()})
            n_pred += 1
    infer_s = time.time() - t0
    metrics = {}
    if not args.smoke:
        for part in parts:
            metrics[part] = evaluate_partition(exp / "predictions", part, dcfg["truth_file"], exp)
            if part == "test":
                log_test_evaluation(exp, "final test evaluation of trained model")
    if args.smoke:   # not an experiment: no index entry
        write_json(exp / "run.json", {"kind": "smoke_test", "training": result, "environment": environment()})
    else:
        write_run(exp, {
            "kind": "training", "model": mcfg["family"], "base_checkpoint": mcfg["base_checkpoint"],
            "model_config": mcfg, "dataset": dataset_fingerprint(dcfg), "random_seed": tcfg["seed"],
            "hyperparameters": {k: tcfg[k] for k in ("learning_rate", "batch_size", "gradient_accumulation_steps",
                                                       "epochs", "weight_decay", "warmup_ratio", "o_class_weight")},
            "training_config": tcfg, "image_resolution": dcfg["rendering"],
            "ocr": {"engine": ocfg["engine"], "config": ocfg}, "environment": environment(),
            "training": result, "inference_time_s_per_doc": round(infer_s / max(1, n_pred), 2),
            "partitions_evaluated": parts,
        })
    print("done:", exp)


if __name__ == "__main__":
    main()
