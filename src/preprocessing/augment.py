"""Safe image augmentations for scanned certificates.

Every transform changes *image quality only* - never text, digits, layout
order or which marks are printed. Magnitudes are small so the page remains
legible; each augmented page is re-OCRed and re-aligned against the same
truth, and pages whose alignment gets worse than the original are dropped.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DEFAULTS = {
    "brightness": (0.8, 1.2),
    "contrast": (0.8, 1.25),
    "rotation_deg": (-2.0, 2.0),
    "blur_radius": (0.0, 0.8),
    "noise_sigma": (0.0, 6.0),
    "jpeg_quality": (45, 90),
    "resolution_scale": (0.6, 1.0),
}


def augment(img: Image.Image, rng: random.Random, cfg: dict | None = None) -> tuple[Image.Image, dict]:
    c = {**DEFAULTS, **(cfg or {})}
    rec = {}
    img = img.convert("RGB")
    rec["brightness"] = b = rng.uniform(*c["brightness"])
    img = ImageEnhance.Brightness(img).enhance(b)
    rec["contrast"] = k = rng.uniform(*c["contrast"])
    img = ImageEnhance.Contrast(img).enhance(k)
    rec["rotation_deg"] = r = rng.uniform(*c["rotation_deg"])
    img = img.rotate(r, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    rec["blur_radius"] = br = rng.uniform(*c["blur_radius"])
    if br > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(br))
    rec["noise_sigma"] = s = rng.uniform(*c["noise_sigma"])
    if s > 0.1:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(rng.randrange(2**31)).normal(0, s, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    rec["resolution_scale"] = sc = rng.uniform(*c["resolution_scale"])
    if sc < 0.99:
        w, h = img.size
        img = img.resize((int(w * sc), int(h * sc)), Image.BILINEAR)
    rec["jpeg_quality"] = q = rng.randint(*c["jpeg_quality"])
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}
