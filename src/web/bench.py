"""TEMPORARY: measure OCR / model speed settings on the host itself (results on /health).

Runs once in the background after startup, on a synthetic text page (no user data).
"""
from __future__ import annotations

import contextlib
import threading
import time

import numpy as np

RESULTS: dict = {"status": "pending"}


def _page() -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (2000, 2800), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=34)
    for i in range(40):
        d.text((90, 60 + i * 38), f"{i:02d} ENGLISH PAPER-I 100 0{i:02d} SECOND LANGUAGE SANSKRIT PASSED A+",
               fill="black", font=font)
    return np.asarray(img)[:, :, ::-1].copy()


@contextlib.contextmanager
def _no_spinning():
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession
    orig = OrtInferSession._init_sess_opts

    def patched(cfg):
        so = orig(cfg)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return so
    OrtInferSession._init_sess_opts = staticmethod(patched)
    try:
        yield
    finally:
        OrtInferSession._init_sess_opts = staticmethod(orig)


def _time_ocr(params: dict, spin: bool, img) -> float:
    from rapidocr import RapidOCR
    base = {"Global.log_level": "warning", "Global.use_cls": False}
    with (contextlib.nullcontext() if spin else _no_spinning()):
        eng = RapidOCR(params=base | params)
    eng(img, return_word_box=True)
    t = time.time()
    for _ in range(1):
        eng(img, return_word_box=True)
    return round(time.time() - t, 2)


def _run(extractor) -> None:
    try:
        from rapidocr import EngineType
        img = _page()
        ocr = {}
        for n in (8, 4, 2):
            ocr[f"ort_t{n}_spin"] = _time_ocr({"EngineConfig.onnxruntime.intra_op_num_threads": n}, True, img)
            ocr[f"ort_t{n}_nospin"] = _time_ocr({"EngineConfig.onnxruntime.intra_op_num_threads": n}, False, img)
        ocr["ort_t8_nospin_arena"] = _time_ocr({"EngineConfig.onnxruntime.intra_op_num_threads": 8,
                                               "EngineConfig.onnxruntime.enable_cpu_mem_arena": True}, False, img)
        try:
            for n in (8, 4):
                ocr[f"openvino_t{n}"] = _time_ocr({"Det.engine_type": EngineType.OPENVINO,
                                                   "Rec.engine_type": EngineType.OPENVINO,
                                                   "EngineConfig.openvino.inference_num_threads": n}, True, img)
        except Exception as e:  # openvino missing
            ocr["openvino"] = f"error: {e}"
        RESULTS["ocr_s_per_page"] = ocr

        import torch
        model = extractor.tagger.model
        ids = torch.randint(5, 1000, (4, 512))
        bbox = torch.randint(0, 1000, (4, 512, 4)).sort(dim=-1).values
        mask = torch.ones(4, 512, dtype=torch.long)
        prev = torch.get_num_threads()
        tm = {}
        with torch.no_grad():
            for n in (8, 4, 2):
                torch.set_num_threads(n)
                model(input_ids=ids, bbox=bbox, attention_mask=mask)
                t = time.time()
                model(input_ids=ids, bbox=bbox, attention_mask=mask)
                tm[f"torch_t{n}"] = round(time.time() - t, 2)
        torch.set_num_threads(prev)
        RESULTS["model_s_per_batch4"] = tm
        RESULTS["status"] = "done"
    except Exception as e:
        RESULTS["status"] = f"error: {e!r}"


def start(extractor) -> None:
    threading.Thread(target=_run, args=(extractor,), daemon=True).start()
