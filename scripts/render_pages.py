"""Render every PDF in the raw directory to page images + a page index."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import PROJECT_ROOT, load_yaml, resolve, write_json  # noqa: E402
from src.preprocessing.render import render_pdf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--only", nargs="*", help="Optional list of PDF stems to render")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rcfg = cfg["rendering"]
    raw = resolve(cfg["raw_pdf_dir"])
    out_dir = resolve(rcfg["output_dir"])
    index = {}
    pdfs = sorted(raw.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.stem in set(args.only)]
    for pdf in pdfs:
        pages = render_pdf(pdf, out_dir, rcfg["max_long_side_px"], rcfg["min_long_side_px"],
                           rcfg.get("image_format", "png"), PROJECT_ROOT)
        index[pdf.name] = [p.to_dict() for p in pages]
        print(f"{pdf.name}: {len(pages)} page(s) {[(p.width, p.height) for p in pages]}")
    write_json(out_dir / "page_index.json", index)


if __name__ == "__main__":
    main()
