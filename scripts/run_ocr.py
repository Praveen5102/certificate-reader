"""Run OCR over all rendered pages; writes one JSON per page (resumable)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.common.io import load_yaml, read_json, resolve, write_json  # noqa: E402
from src.ocr.engine import OCRConfig, OCREngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--ocr-config", default="configs/ocr.yaml")
    ap.add_argument("--force", action="store_true", help="re-OCR pages that already have output")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    dcfg = load_yaml(args.dataset_config)
    ocfg = load_yaml(args.ocr_config)
    out_dir = resolve(ocfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    index = read_json(resolve(dcfg["rendering"]["output_dir"]) / "page_index.json")
    engine = OCREngine(OCRConfig(**{k: ocfg[k] for k in ("engine", "try_rotations",
                                                         "min_horizontal_fraction", "min_word_confidence")}))
    for pdf, pages in sorted(index.items()):
        if args.only and Path(pdf).stem not in args.only:
            continue
        for p in pages:
            out = out_dir / f"{Path(pdf).stem}_p{p['page']}.json"
            if out.exists() and not args.force:
                continue
            t0 = time.time()
            img = Image.open(resolve(p["path"]))
            res = engine.ocr_page(img, page=p["page"])
            res["file"] = pdf
            res["image"] = p["path"]
            res["elapsed_s"] = round(time.time() - t0, 2)
            write_json(out, res)
            print(f"{out.name}: {len(res['words'])} words, rot={res['rotation']}, {res['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
