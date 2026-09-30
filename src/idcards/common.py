"""Shared helpers for ID-card extraction: page orientation, lines, dates, result shape."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

# OCR confusions, applied only where the field format says a digit / a letter must be
TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "1", "|": "1",
                          "Z": "2", "S": "5", "B": "8", "G": "6"})
TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"})


class Line:
    __slots__ = ("text", "x0", "y0", "x1", "y1", "conf", "page")

    def __init__(self, d: dict, page: int):
        self.text = d["text"].strip()
        self.x0, self.y0, self.x1, self.y1 = d["bbox"]
        self.conf = d["confidence"]
        self.page = page

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def __repr__(self) -> str:  # debugging aid
        return f"Line({self.text!r}, p{self.page}, y={self.y0})"


def lines_of(ocr_pages: list[dict]) -> list[Line]:
    out = [Line(l, p["page"]) for p in ocr_pages for l in p["lines"] if l["text"].strip()]
    return sorted(out, key=lambda l: (l.page, l.y0, l.x0))


def keyword_hits(lines: list[dict], keywords: list[str]) -> int:
    text = " ".join(l["text"].upper() for l in lines)
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    return sum(1 for k in keywords if k in text)


def ocr_upright(img, ocr_engine, page: int, keywords: list[str], enough: int = 3) -> dict:
    """OCR a card photo in the orientation where its printed keywords read.
    Phone photos come sideways or upside down; other paper in the photo can read
    better than the card itself, so orientation is judged by the card's own words."""
    p = ocr_engine.ocr_page(img, page=page)

    # horizontal text first: the recogniser also reads a sideways page, but then
    # the label/value positions are meaningless
    def quality(lines):
        return (ocr_engine._horizontal_fraction(lines) >= 0.5, keyword_hits(lines, keywords),
                ocr_engine._mean_conf_len(lines))
    q = quality(p["lines"])
    if q[0] and q[1] >= enough:
        return p
    for rot in (0, 90, 270, 180):
        if rot == p["rotation"]:
            continue
        cand = img.rotate(rot, expand=True) if rot else img
        lines, words = ocr_engine._run(cand)
        cq = quality(lines)
        if cq > q:
            q = cq
            p = {**p, "width": cand.width, "height": cand.height, "rotation": rot, "lines": lines, "words": words}
    return p


DATE = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})")


def parse_date(text: str) -> str:
    """'DOB: 26/06/2000' or '08-05-1998' -> ISO date; '' if none or impossible."""
    s = text or ""
    # digits misread as letters only inside date-shaped runs
    s = re.sub(r"[0-9OIlSB]{1,2}\s*[/\-.]\s*[0-9OIlSB]{1,2}\s*[/\-.]\s*[0-9OIlSB]{4}",
               lambda m: m.group(0).upper().translate(TO_DIGIT), s)
    m = DATE.search(s)
    if not m:
        return ""
    try:
        d = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return ""
    return d.isoformat() if 1900 <= d.year <= dt.date.today().year else ""


NAME_STOP = {"GOVERNMENT", "INDIA", "AADHAAR", "AADHAR", "AUTHORITY", "UNIQUE", "IDENTIFICATION", "DOB",
             "MALE", "FEMALE", "ENROLMENT", "ENROLLMENT", "ADDRESS", "INCOME", "TAX", "DEPARTMENT", "GOVT",
             "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "SIGNATURE", "NAME", "FATHER", "FATHERS", "BIRTH",
             "DATE", "VID", "ISSUE", "DOWNLOAD", "INFORMATION", "PROOF", "IDENTITY", "CITIZENSHIP",
             "MINOR", "YOUR", "VERIFIED", "VALID", "MOBILE", "HELP", "TO", "OF"}


def name_like(text: str, min_words: int = 1) -> bool:
    """A person's name in Latin letters: letters/spaces/dots only, no template words."""
    t = text.strip(" ,.:;")
    if not t or re.search(r"\d", t) or not re.fullmatch(r"[A-Za-z .']+", t):
        return False
    words = [w for w in re.split(r"[ .]+", t) if w]
    if len(words) < min_words or sum(len(w) for w in words) < 4:
        return False
    return not any(w.upper() in NAME_STOP for w in words)


def clean_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" ,.:;")).strip()


def field(value, source: str, conf=None, verified=False, raw=None) -> dict:
    return {"value": value, "source": source, "confidence": conf, "verified": verified, "raw": raw}


def finish(document_type: str, fields: dict, flags: dict, required: list[str], checks: dict,
           notes: str = "") -> dict[str, Any]:
    for k in required:
        if not fields[k]["value"]:
            flags.setdefault(k, "not found - enter manually")
    blocking = [k for k in flags if k in required]
    status = "auto" if not flags else ("manual_required" if blocking else "review_recommended")
    return {"document_type": document_type, "fields": fields, "checks": checks,
            "review": {"status": status, "fields_requiring_review": list(flags), "reasons": flags},
            "notes": notes}


def flat(result: dict) -> dict:
    return {k: v["value"] for k, v in result["fields"].items()} | {"document_type": result["document_type"]}
