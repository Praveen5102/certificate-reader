"""Local web app: upload a certificate, see the extracted data.

    python scripts/serve.py            # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import base64
import io
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..inference.pipeline import SUPPORTED_SUFFIXES, CertificateExtractor
from ..preprocessing.render import load_input_pages

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Certificate Extractor")
_extractor: CertificateExtractor | None = None
_lock = threading.Lock()          # one extraction at a time (model + OCR are CPU-heavy)


def extractor() -> CertificateExtractor:
    global _extractor
    if _extractor is None:
        _extractor = CertificateExtractor()
    return _extractor


def _preview(path: Path) -> str | None:
    """First page as a small JPEG data URI, for display next to the results."""
    try:
        img = load_input_pages(path, 1400, 600)[0]
        img.thumbnail((900, 900))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _extractor is not None}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Upload a PDF or an image (JPG, PNG).")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "File is larger than 25 MB.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        t0 = time.time()
        with _lock:
            result = extractor().extract_file(path)
        preview = _preview(path)
    full = result.model_dump()
    full["file"] = file.filename
    flat = result.to_flat()
    flat["file"] = file.filename
    return JSONResponse({"flat": flat, "full": full, "preview": preview, "seconds": round(time.time() - t0, 1)})
