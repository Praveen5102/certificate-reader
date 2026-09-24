"""Extract structured JSON from certificate PDFs / images.

    python scripts/infer.py certificate.pdf                          # rule baseline, prints flat JSON
    python scripts/infer.py a.pdf b.jpg --out results/ --full        # full traceable JSON per file
    python scripts/infer.py a.pdf --tagger model --model-dir experiments/exp_004_lilt/model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import write_json  # noqa: E402
from src.inference.pipeline import CertificateExtractor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="PDF or image files")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--tagger", choices=["rule_baseline", "model"], default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default=None, help="directory for one JSON per input")
    ap.add_argument("--full", action="store_true", help="write the full traceable result (evidence, confidence, review)")
    args = ap.parse_args()
    ex = CertificateExtractor(args.config, tagger=args.tagger, model_dir=args.model_dir)
    for inp in args.inputs:
        res = ex.extract_file(inp)
        payload = res.model_dump() if args.full else res.to_flat()
        if args.out:
            out = Path(args.out) / f"{Path(inp).stem}.json"
            write_json(out, payload)
            print(f"{inp} -> {out}  [{res.document_type}; review: {res.review.status}; "
                  f"fields to check: {res.review.fields_requiring_review}]")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
