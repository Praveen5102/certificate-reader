"""PAN card extraction (Income Tax Department, India).

PAN format: 5 letters + 4 digits + 1 letter (ABCPE1234F). The 4th letter is the
holder type (P = individual), and for individuals the 5th letter is the first
letter of the surname - a built-in cross-check against the printed name.
Newer cards print bilingual labels ('नाम / Name', 'पिता का नाम / Father's Name',
'जन्म की तारीख / Date of Birth') with the value below; older cards print the
name, father's name and date of birth as plain lines under the header.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from rapidfuzz import fuzz

from .common import (TO_ALPHA, TO_DIGIT, Line, clean_name, field, finish, lines_of, name_like, ocr_upright,
                     parse_date)

KEYWORDS = ["INCOME TAX", "DEPARTMENT", "GOVT OF INDIA", "PERMANENT ACCOUNT", "NUMBER CARD", "NAME",
            "DATE OF BIRTH", "SIGNATURE"]
HOLDER_TYPES = {"P": "individual", "C": "company", "H": "Hindu undivided family", "F": "firm",
                "A": "association of persons", "T": "trust", "B": "body of individuals",
                "L": "local authority", "J": "artificial juridical person", "G": "government"}
_PAN = re.compile(r"[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]")


def _pan_candidates(lines: list[Line]):
    out = []
    for l in lines:
        s = re.sub(r"[^A-Z0-9]", "", l.text.upper())
        for i in range(0, max(0, len(s) - 9)):
            w = s[i:i + 10]
            if _PAN.fullmatch(w):
                out.append((w, l, False))
                continue
            # repair by position: letters at 1-5 and 10, digits at 6-9
            fixed = w[:5].translate(TO_ALPHA) + w[5:9].translate(TO_DIGIT) + w[9].translate(TO_ALPHA)
            if _PAN.fullmatch(fixed) and len(s) <= 12:
                out.append((fixed, l, True))
    return out


def _label(text: str) -> str | None:
    t = re.sub(r"[^A-Za-z' ]", " ", text.rsplit("/", 1)[-1]).upper()
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    for key, ph in (("father_name", "FATHER'S NAME"), ("father_name", "FATHERS NAME"),
                    ("father_name", "FATHER NAME"), ("date_of_birth", "DATE OF BIRTH"), ("name", "NAME")):
        if t == ph or (len(t) >= 4 and fuzz.ratio(t, ph) >= 85):
            return key
    return None


def _below(label: Line, lines: list[Line], want) -> Line | None:
    c = [l for l in lines if l.page == label.page and l is not label and l.cy > label.cy + 0.4 * label.h
         and l.y0 <= label.y1 + 2.2 * label.h and abs(l.x0 - label.x0) <= 3 * label.h and want(l)]
    return min(c, key=lambda l: l.y0) if c else None


def _upper_name(l: Line) -> bool:
    return name_like(l.text) and _label(l.text) is None


def extract_pan(ocr_pages: list[dict]) -> dict[str, Any]:
    lines = lines_of(ocr_pages)
    fields: dict[str, dict] = {}
    flags: dict[str, str] = {}
    checks: dict[str, Any] = {}

    cands = _pan_candidates(lines)
    votes = Counter(c[0] for c in cands)
    pan, pline, repaired = "", None, False
    if votes:
        pan = max(votes, key=lambda p: (votes[p], not any(r for q, _, r in cands if q == p)))
        pline, repaired = next((l, r) for q, l, r in cands if q == pan)
    fields["pan_number"] = field(pan, "printed number" if pan else "not found", pline.conf if pline else None,
                                 raw=pline.text if pline else None)
    if repaired:
        checks["pan_repair"] = f"OCR letter/digit confusion corrected: '{pline.text}' -> '{pan}'"

    got: dict[str, Line] = {}
    for l in lines:
        key = _label(l.text)
        if key and key not in got:
            want = (lambda x: parse_date(x.text) != "") if key == "date_of_birth" else _upper_name
            v = _below(l, lines, want)
            if v is not None:
                got[key] = v
    # older cards: no labels - name, father's name, date of birth as the first lines under the header
    if "name" not in got or "father_name" not in got:
        head = [l for l in lines if re.search(r"INCOME\s*TAX|GOVT", l.text.upper())]
        top = max((l.y1 for l in head), default=0)
        seq = [l for l in lines if l.y0 > top and _upper_name(l) and l.text.upper() == l.text
               and l not in got.values()]
        for key, l in zip(("name", "father_name"), seq):
            got.setdefault(key, l)
            checks.setdefault("layout", "older card without labels (values by position)")
    if "date_of_birth" not in got:
        dates = [l for l in lines if parse_date(l.text)]
        if dates:
            got["date_of_birth"] = dates[0]

    for key in ("name", "father_name"):
        l = got.get(key)
        fields[key] = field(clean_name(l.text).upper() if l else "", "printed text" if l else "not found",
                            l.conf if l else None, raw=l.text if l else None)
    l = got.get("date_of_birth")
    fields["date_of_birth"] = field(parse_date(l.text) if l else "", "printed text" if l else "not found",
                                    l.conf if l else None, raw=l.text if l else None)

    # built-in checks
    if pan:
        checks["holder_type"] = HOLDER_TYPES.get(pan[3], "unknown")
        name = fields["name"]["value"]
        if pan[3] == "P" and name:
            # usually the surname's initial; names are printed in either order, and
            # some holders registered another part of the name as the surname
            ok = pan[4] in {w[0] for w in name.split()}
            checks["surname_letter_matches"] = ok
            fields["pan_number"]["verified"] = ok and not repaired
            if not ok:
                flags["pan_number"] = (f"5th letter '{pan[4]}' should be the first letter of the surname "
                                       f"('{name}') - check the number or the name")
        elif pan[3] != "P":
            flags["pan_number"] = f"not an individual's PAN (holder type: {checks['holder_type']})"
    for key in ("name", "father_name", "date_of_birth"):
        c = fields[key]["confidence"]
        if fields[key]["value"] and c is not None and c < 0.85:
            flags.setdefault(key, f"low OCR confidence ({c:.2f})")
    return finish("pan" if (pan or fields["name"]["value"]) else "unknown", fields, flags,
                  ["pan_number", "name", "date_of_birth"], checks)


def extract_pan_images(images: list, ocr_engine) -> dict[str, Any]:
    pages = [ocr_upright(img, ocr_engine, i + 1, KEYWORDS) for i, img in enumerate(images)]
    r = extract_pan(pages)
    r["pages"] = len(images)
    return r
