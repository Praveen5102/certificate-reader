"""OCR layer: page image -> words + lines with boxes and confidences.

Engine: RapidOCR (PaddleOCR PP-OCR detection/recognition models exported to
ONNX). Chosen because it (a) is pretrained and needs no system install,
(b) returns word-level boxes and recognition confidences, (c) runs at a few
seconds per page on CPU, and (d) handled the fine print on this dataset's
scans well in spot checks (see README "OCR engine choice").

Output format (one JSON file per page) is engine-agnostic so the engine can be
swapped without touching annotation / model code:

    {"page": 1, "width": W, "height": H, "rotation": 0, "engine": {...},
     "lines": [{"id": 0, "text": "...", "bbox": [x0,y0,x1,y1], "confidence": c,
                "words": [{"id": 0, "text": "...", "bbox": [...], "confidence": c}]}],
     "words": [{"id": 0, "line_id": 0, "text": "...", "bbox": [...], "confidence": c}]}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _quad_to_bbox(quad) -> list[int]:
    arr = np.asarray(quad, dtype=float)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


@dataclass
class OCRConfig:
    engine: str = "rapidocr"
    try_rotations: bool = True
    # A page is considered mis-rotated if fewer than this fraction of detected
    # text boxes are wider than tall.
    min_horizontal_fraction: float = 0.5
    min_word_confidence: float = 0.0   # keep everything; downstream decides
    # Per-text-line 180-degree classifier. OFF: it flips short numeric crops
    # ('098' -> '860', '069' -> '690'); whole-page rotation is handled above.
    use_angle_cls: bool = False


class OCREngine:
    def __init__(self, cfg: OCRConfig | None = None):
        self.cfg = cfg or OCRConfig()
        if self.cfg.engine != "rapidocr":
            raise ValueError(f"Unsupported OCR engine: {self.cfg.engine}")
        from rapidocr import RapidOCR  # imported lazily: heavy

        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        params = {"Global.log_level": "warning", "Global.use_cls": self.cfg.use_angle_cls}
        self._engine = RapidOCR(params=params)
        self.engine_info = {"name": "rapidocr", "version": self._version(),
                            "models": "PP-OCR det/rec (rapidocr defaults)",
                            "use_angle_cls": self.cfg.use_angle_cls}

    @staticmethod
    def _version() -> str:
        try:
            from importlib.metadata import version
            return version("rapidocr")
        except Exception:  # pragma: no cover
            return "unknown"

    def _run(self, img: Image.Image) -> tuple[list[dict], list[dict]]:
        res = self._engine(np.asarray(img.convert("RGB"))[:, :, ::-1], return_word_box=True)
        lines, words = [], []
        if res.boxes is None:
            return lines, words
        word_results = res.word_results or [()] * len(res.txts)
        for li, (box, txt, score, wres) in enumerate(zip(res.boxes, res.txts, res.scores, word_results)):
            line = {"id": li, "text": txt, "bbox": _quad_to_bbox(box),
                    "confidence": round(float(score), 4), "words": []}
            for wtxt, wscore, wquad in (wres or [(txt, score, box)]):
                if not str(wtxt).strip() or float(wscore) < self.cfg.min_word_confidence:
                    continue
                w = {"id": len(words), "line_id": li, "text": str(wtxt),
                     "bbox": _quad_to_bbox(wquad), "confidence": round(float(wscore), 4)}
                words.append(w)
                line["words"].append({k: w[k] for k in ("id", "text", "bbox", "confidence")})
            lines.append(line)
        return lines, words

    @staticmethod
    def _horizontal_fraction(lines: list[dict]) -> float:
        if not lines:
            return 0.0
        horiz = sum(1 for l in lines if (l["bbox"][2] - l["bbox"][0]) >= (l["bbox"][3] - l["bbox"][1]))
        return horiz / len(lines)

    def ocr_page(self, img: Image.Image, page: int = 1) -> dict[str, Any]:
        rotation = 0
        lines, words = self._run(img)
        if self.cfg.try_rotations and self._horizontal_fraction(lines) < self.cfg.min_horizontal_fraction:
            best = (self._mean_conf_len(lines), 0, lines, words, img)
            for rot in (90, 270, 180):
                cand = img.rotate(rot, expand=True)
                l2, w2 = self._run(cand)
                score = self._mean_conf_len(l2)
                if score > best[0]:
                    best = (score, rot, l2, w2, cand)
            _, rotation, lines, words, img = best
            if rotation:
                log.info("page rotated by %d degrees", rotation)
        return {"page": page, "width": img.width, "height": img.height, "rotation": rotation,
                "engine": self.engine_info, "lines": lines, "words": words}

    @staticmethod
    def _mean_conf_len(lines: list[dict]) -> float:
        # Confidence-weighted character count: prefers the orientation that
        # yields the most confidently-read text.
        return sum(l["confidence"] * len(l["text"]) for l in lines)


def page_text(page_ocr: dict) -> str:
    return "\n".join(l["text"] for l in page_ocr["lines"])
