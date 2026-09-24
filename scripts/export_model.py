"""Export a trained model for deployment: fp16 weights split into <=90 MB parts.

GitHub rejects files over 100 MB, so the weights are stored as half precision
(~260 MB instead of ~520 MB) and split into parts that are committed normally.
At startup `src.models.layout_tagger` joins the parts (checking the SHA-256)
and loads the weights as float32.

    python scripts/export_model.py --src experiments/exp_004_lilt/model --dst deploy/model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PART_BYTES = 90 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/exp_004_lilt/model")
    ap.add_argument("--dst", default="deploy/model")
    args = ap.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name != "model.safetensors":
            shutil.copy2(f, dst / f.name)
    state = load_file(src / "model.safetensors")
    half = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    tmp = dst / "model.fp16.safetensors"
    save_file(half, tmp, metadata={"format": "pt"})
    data = tmp.read_bytes()
    tmp.unlink()
    parts = []
    for i in range(0, len(data), PART_BYTES):
        name = f"model.fp16.safetensors.part{i // PART_BYTES:02d}"
        (dst / name).write_bytes(data[i:i + PART_BYTES])
        parts.append(name)
    manifest = {"file": "model.fp16.safetensors", "parts": parts, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "source": str(src).replace("\\", "/"),
                "dtype": "float16 on disk, loaded as float32"}
    (dst / "parts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(parts)} parts, {len(data) / 1e6:.0f} MB ->", dst)


if __name__ == "__main__":
    main()
