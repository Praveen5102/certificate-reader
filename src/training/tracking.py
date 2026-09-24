"""File-based experiment tracking.

MLflow / W&B were considered; MLflow imports pandas, whose native modules are
blocked by this machine's Windows Application Control policy. A plain,
append-only directory layout is used instead: every run gets its own
experiments/exp_NNN_<name>/ directory (never overwritten) with a run.json that
records everything spec §20 asks for, plus experiments/index.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

from ..common.io import PROJECT_ROOT, read_json, sha256_file, write_json

EXP_ROOT = PROJECT_ROOT / "experiments"


def new_experiment_dir(name: str) -> Path:
    EXP_ROOT.mkdir(exist_ok=True)
    nums = [int(m.group(1)) for p in EXP_ROOT.iterdir() if (m := re.match(r"exp_(\d{3})_", p.name))]
    nxt = max(nums, default=0) + 1
    d = EXP_ROOT / f"exp_{nxt:03d}_{name}"
    d.mkdir(parents=False, exist_ok=False)   # never overwrite
    return d


def environment() -> dict[str, Any]:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    except Exception:
        env["torch"] = None
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    return env


def dataset_fingerprint(dataset_cfg: dict) -> dict[str, Any]:
    split = read_json(PROJECT_ROOT / "data" / "split.json")
    return {"dataset_version": dataset_cfg["dataset_version"],
            "truth_file": dataset_cfg["truth_file"],
            "truth_sha256": sha256_file(PROJECT_ROOT / dataset_cfg["truth_file"]),
            "split_file": "data/split.json", "split_sha256": sha256_file(PROJECT_ROOT / "data" / "split.json"),
            "split_seed": split["seed"], "split_sizes": split["sizes"]}


def write_run(exp_dir: Path, run: dict[str, Any]) -> None:
    run = {"experiment": exp_dir.name, "created": dt.datetime.now().isoformat(timespec="seconds"), **run}
    write_json(exp_dir / "run.json", run)
    with open(EXP_ROOT / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"experiment": exp_dir.name, "created": run["created"], "kind": run.get("kind"),
                            "model": run.get("model")}) + "\n")


def log_test_evaluation(exp_dir: Path, note: str = "") -> None:
    """Every evaluation on the TEST partition is logged, so repeated test use
    (a form of tuning on test) is visible in the audit trail."""
    with open(EXP_ROOT / "test_evaluations.log", "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{exp_dir.name}\t{note}\n")
