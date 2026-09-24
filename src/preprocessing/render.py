"""PDF -> page images.

Every source PDF in this dataset is a scan or phone photo wrapped in a PDF, so
rendering resolution is chosen from the embedded image's native resolution
rather than a fixed DPI: rendering a 1920x2550pt "pdf-lib" page at 300 DPI
would produce an 8000px image of a 1920px photo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


@dataclass
class RenderedPage:
    page: int                 # 1-based page number
    path: str                 # image path, relative to the project root when possible
    width: int
    height: int
    zoom: float               # render scale relative to PDF points
    native_long_side: int     # longest side of the largest embedded image (0 if none)
    has_text_layer: bool      # PDF already carries an (unreliable) OCR text layer

    def to_dict(self) -> dict:
        return asdict(self)


def _native_long_side(doc: fitz.Document, page: fitz.Page) -> int:
    best = 0
    for img in page.get_images(full=True):
        w, h = img[2], img[3]
        best = max(best, w, h)
    return best


def page_zoom(page: fitz.Page, native_long: int, max_long: int, min_long: int) -> float:
    long_pts = max(page.rect.width, page.rect.height)
    target = native_long if native_long > 0 else max_long
    target = max(min(target, max_long), min_long)
    return target / long_pts


def render_page(page: fitz.Page, zoom: float) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return Image.fromarray(arr.copy())


def render_pdf(pdf_path: Path, out_dir: Path, max_long: int, min_long: int,
               image_format: str = "png", project_root: Path | None = None) -> list[RenderedPage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[RenderedPage] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            native = _native_long_side(doc, page)
            zoom = page_zoom(page, native, max_long, min_long)
            img = render_page(page, zoom)
            out = out_dir / f"{pdf_path.stem}_p{i + 1}.{image_format}"
            img.save(out)
            rel = out
            if project_root is not None:
                try:
                    rel = out.relative_to(project_root)
                except ValueError:
                    pass
            pages.append(RenderedPage(
                page=i + 1, path=rel.as_posix(), width=img.width, height=img.height,
                zoom=round(zoom, 4), native_long_side=native,
                has_text_layer=bool(page.get_text("text").strip()),
            ))
    return pages


def load_input_pages(path: Path, max_long: int, min_long: int) -> list[Image.Image]:
    """Inference-time loader: accept a PDF or a single image file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        out = []
        with fitz.open(path) as doc:
            for page in doc:
                out.append(render_page(page, page_zoom(page, _native_long_side(doc, page), max_long, min_long)))
        return out
    img = Image.open(path).convert("RGB")
    long_side = max(img.size)
    if long_side > max_long:
        scale = max_long / long_side
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return [img]
