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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..inference.pipeline import SUPPORTED_SUFFIXES, CertificateExtractor
from ..passport.extract import extract_passport_images, flat as passport_flat
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
    from ..common.cpu import cpu_info
    return {"status": "ok", "model_loaded": _extractor is not None, "cpu": cpu_info()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/extract")
async def extract(files: list[UploadFile] = File(...), doc_type: str = Form("certificate")) -> JSONResponse:
    """doc_type: 'certificate' (12th / Intermediate) or 'passport'.
    Passports may be uploaded as up to 4 files (photo page + last page)."""
    if doc_type not in ("certificate", "passport"):
        raise HTTPException(400, "Unknown document type.")
    if not files:
        raise HTTPException(400, "No file uploaded.")
    if doc_type == "certificate" and len(files) > 1:
        raise HTTPException(400, "Upload one certificate at a time.")
    if len(files) > 4:
        raise HTTPException(400, "Upload at most 4 files.")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, f in enumerate(files):
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(400, f"Unsupported file type '{suffix}'. Upload a PDF or an image (JPG, PNG).")
            data = await f.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(400, f"{f.filename} is larger than 25 MB.")
            path = Path(tmp) / f"upload{i}{suffix}"
            path.write_bytes(data)
            paths.append(path)
        names = ", ".join(f.filename or "" for f in files)
        t0 = time.time()
        with _lock:
            if doc_type == "certificate":
                result = extractor().extract_file(paths[0])
            else:
                ex = extractor()
                images = [img for path in paths
                          for img in load_input_pages(path, ex.cfg["rendering"]["max_long_side_px"],
                                                      ex.cfg["rendering"]["min_long_side_px"])]
                result = extract_passport_images(images[:8], ex.ocr)
        preview = _preview(paths[0])
    if doc_type == "certificate":
        full = result.model_dump()
        full["file"] = names
        flat = result.to_flat()
        flat["file"] = names
        return JSONResponse({"kind": "certificate", "flat": flat, "full": full, "preview": preview,
                             "seconds": round(time.time() - t0, 1), "file": names})
    flat = passport_flat(result) | {"file": names}
    return JSONResponse({"kind": "passport", "flat": flat, "full": result, "preview": preview,
                         "seconds": round(time.time() - t0, 1), "file": names})
