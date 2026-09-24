"""End-to-end extraction: PDF/image -> ExtractionResult.

    validate/classify -> render pages -> OCR -> tag (rules | model)
    -> table reconstruction -> normalization -> validation -> confidence/review
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..annotation.anchors import TemplateIndex
from ..annotation.layout import load_pages
from ..common.io import load_yaml
from ..preprocessing.classify import classify_document
from ..preprocessing.render import load_input_pages
from .postprocess import build_result
from .schema import ExtractionResult, ReviewSummary

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class CertificateExtractor:
    def __init__(self, inference_config: str | Path = "configs/inference.yaml", tagger: str | None = None,
                 model_dir: str | None = None):
        self.cfg = load_yaml(inference_config)
        self.fields_cfg = load_yaml(self.cfg["fields_config"])
        self.ocr_cfg = load_yaml(self.cfg["ocr_config"])
        kind = tagger or self.cfg["tagger"]
        if kind == "rule_baseline":
            from ..models.rule_baseline import RuleTagger
            self.tagger = RuleTagger(self.fields_cfg)
            self.conf_source = "rule evidence: min OCR word confidence x anchor match (uncalibrated)"
        elif kind == "model":
            from ..models.layout_tagger import LayoutTagger
            self.tagger = LayoutTagger(model_dir or self.cfg["model_dir"])
            self.conf_source = "model: min token probability x min OCR word confidence (uncalibrated)"
        else:
            raise ValueError(f"unknown tagger {kind!r}")
        # Table fallback: rule tagger, used only when the model's subject table does
        # not add up to the printed total and the rule table does.
        self.fallback = None
        if kind == "model" and self.cfg.get("table_fallback") == "rule_baseline":
            from ..models.rule_baseline import RuleTagger
            self.fallback = RuleTagger(self.fields_cfg)
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            from ..ocr.engine import OCRConfig, OCREngine
            keys = ("engine", "try_rotations", "min_horizontal_fraction", "min_word_confidence", "use_angle_cls")
            self._ocr = OCREngine(OCRConfig(**{k: self.ocr_cfg[k] for k in keys}))
        return self._ocr

    def provenance(self, extra: dict | None = None) -> dict[str, Any]:
        p = {"tagger": getattr(self.tagger, "name", type(self.tagger).__name__),
             "confidence_source": self.conf_source, "ocr": self.ocr_cfg.get("engine")}
        p.update(extra or {})
        return p

    # ------------------------------------------------------------------------------------
    def extract_file(self, path: str | Path) -> ExtractionResult:
        path = Path(path)
        if not path.exists() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return _invalid(path.name, f"unsupported or missing input: {path}")
        t0 = time.time()
        try:
            images = load_input_pages(path, self.cfg["rendering"]["max_long_side_px"],
                                      self.cfg["rendering"]["min_long_side_px"])
        except Exception as e:  # corrupt / encrypted PDF etc.
            return _invalid(path.name, f"could not open document: {e}")
        if not images:
            return _invalid(path.name, "document has no pages")
        ocr_pages = [self.ocr.ocr_page(img, page=i + 1) for i, img in enumerate(images)]
        res = self.extract_from_ocr(path.name, ocr_pages)
        res.provenance["elapsed_s"] = round(time.time() - t0, 2)
        return res

    def extract_from_ocr(self, file: str, ocr_pages: list[dict]) -> ExtractionResult:
        cls = classify_document([[l["text"] for l in p["lines"]] for p in ocr_pages])
        pages = load_pages(ocr_pages, [r["role"] for r in cls["page_roles"]])
        tmpls = {p.page: TemplateIndex(p, self.fields_cfg["template_phrases"]) for p in pages}
        tagged = self.tagger.tag(pages, tmpls) if cls["supported"] else {}
        fallback = self.fallback.tag(pages, tmpls) if (self.fallback and cls["supported"]) else None
        return build_result(file, cls, pages, tagged, self.fields_cfg, self.cfg,
                            self.provenance({"page_roles": [r["role"] for r in cls["page_roles"]],
                                             "ocr_rotations": [p.get("rotation", 0) for p in ocr_pages],
                                             "fallback_confidence_source":
                                                 "rule evidence: min OCR word confidence x anchor match"}),
                            table_fallback=fallback)


def _invalid(file: str, reason: str) -> ExtractionResult:
    return ExtractionResult(file=file, document_type="invalid", supported=False, pages=0,
                            review=ReviewSummary(status="manual_required", fields_requiring_review=["file"],
                                                 reasons={"file": reason}), notes=reason)
