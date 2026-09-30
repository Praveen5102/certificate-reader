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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..idcards.aadhaar import extract_aadhaar_images
from ..idcards.common import flat as id_flat
from ..idcards.pan import extract_pan_images
from ..inference.pipeline import SUPPORTED_SUFFIXES, CertificateExtractor
from ..passport.extract import extract_passport_images, flat as passport_flat
from ..preprocessing.render import load_input_pages

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Certificate Extractor")
_extractor: CertificateExtractor | None = None
_lock = threading.Lock()          # one extraction at a time (model + OCR are CPU-heavy)


_ocr = None
model_error: str | None = None     # set when the certificate model cannot load (e.g. torch blocked)


def extractor() -> CertificateExtractor:
    global _extractor
    if _extractor is None:
        _extractor = CertificateExtractor()
    return _extractor


def ocr_engine():
    """OCR only (passport / Aadhaar / PAN need no trained model)."""
    global _ocr
    if _extractor is not None:
        return _extractor.ocr
    if _ocr is None:
        from ..common.io import load_yaml
        from ..ocr.engine import OCRConfig, OCREngine
        cfg = load_yaml("configs/ocr.yaml")
        keys = ("engine", "try_rotations", "min_horizontal_fraction", "min_word_confidence", "use_angle_cls")
        _ocr = OCREngine(OCRConfig(**{k: cfg[k] for k in keys}))
    return _ocr


AUDIT_LOG = Path("logs/aadhaar_audit.jsonl")


def _audit(event: str, request: Request, result: dict) -> None:
    """One line per full-number view: when, from where, the operator's consent, and
    the last 4 digits only - the log itself never holds a full Aadhaar number."""
    import datetime as dt
    import json
    number = result["fields"]["aadhaar_number"]["value"] or ""
    entry = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "event": event,
             "client": request.client.host if request.client else None, "consent_confirmed": True,
             "number_last4": number.replace(" ", "")[-4:] or None,
             "check_digit_ok": result["checks"].get("number_check_digit")}
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _render_cfg() -> dict:
    from ..common.io import load_yaml
    return load_yaml("configs/inference.yaml")["rendering"]


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
    from .bench import RESULTS
    return {"status": "ok", "model_loaded": _extractor is not None, "model_error": model_error,
            "cpu": cpu_info(), "bench": RESULTS}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/extract")
async def extract(request: Request, files: list[UploadFile] = File(...), doc_type: str = Form("certificate"),
                  consent: str = Form("")) -> JSONResponse:
    """doc_type: 'certificate' (12th / Intermediate), 'passport', 'aadhaar' or 'pan'.
    ID documents may be uploaded as up to 4 files (e.g. card front + back)."""
    if doc_type not in ("certificate", "passport", "aadhaar", "pan"):
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
                try:
                    ex = extractor()
                except Exception as e:  # e.g. torch DLL blocked on this PC
                    raise HTTPException(503, f"The certificate model could not be loaded here ({e}). "
                                             "Passport, Aadhaar and PAN still work.")
                result = ex.extract_file(paths[0])
            else:
                rc = _render_cfg()
                images = [img for path in paths
                          for img in load_input_pages(path, rc["max_long_side_px"], rc["min_long_side_px"])][:8]
                ocr = ocr_engine()
                if doc_type == "passport":
                    result = extract_passport_images(images, ocr)
                elif doc_type == "aadhaar":
                    # full number only with the card holder's consent; otherwise masked
                    show_full = consent == "yes"
                    result = extract_aadhaar_images(images, ocr, show_full_number=show_full)
                    if show_full:
                        _audit("aadhaar_full_number_shown", request, result)
                else:
                    result = extract_pan_images(images, ocr)
        preview = _preview(paths[0])
    if doc_type == "certificate":
        full = result.model_dump()
        full["file"] = names
        flat = result.to_flat()
        flat["file"] = names
        return JSONResponse({"kind": "certificate", "flat": flat, "full": full, "preview": preview,
                             "seconds": round(time.time() - t0, 1), "file": names})
    flat = (passport_flat(result) if doc_type == "passport" else id_flat(result)) | {"file": names}
    return JSONResponse({"kind": doc_type, "flat": flat, "full": result, "preview": preview,
                         "seconds": round(time.time() - t0, 1), "file": names})
