"""Start the local certificate-extraction website.

    python scripts/serve.py              # http://127.0.0.1:8000
    python scripts/serve.py --port 8080
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # configs and model paths are relative to the project root


def main() -> None:
    ap = argparse.ArgumentParser()
    # Railway (and most hosts) pass the port in $PORT and need 0.0.0.0
    ap.add_argument("--host", default="0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    import uvicorn
    from src.web.server import app, extractor
    print("Loading OCR + model (first start takes ~20 s)...", flush=True)
    extractor()
    print(f"Open http://{args.host}:{args.port} in your browser", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
