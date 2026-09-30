"""Aadhaar card / e-Aadhaar letter extraction.

Layouts handled: the PVC/laminated card front (name, DOB, gender, number), the
card back ("Address:" block), and the e-Aadhaar / letter print ("To <name>
S/O ... PIN Code" block plus the cut-out card at the bottom). Each field is read
from the English text; the regional-language lines are OCR noise here.

The 12-digit number carries a Verhoeff check digit, so a misread (or a number
that is not a genuine Aadhaar number) is detected. PRIVACY: the full number is
used only for that check; results show it masked (XXXX XXXX 1234), as UIDAI
rules require for storage outside an Aadhaar Data Vault. The full number is
shown only when the operator confirms the card holder's consent (the web app
records each such view in an audit log, without the number).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .common import (TO_DIGIT, Line, clean_name, field, finish, lines_of, name_like, ocr_upright,
                     parse_date)

KEYWORDS = ["GOVERNMENT OF INDIA", "AADHAAR", "UNIQUE IDENTIFICATION", "AUTHORITY OF INDIA", "DOB", "MALE",
            "ENROLMENT", "ADDRESS", "VID", "UIDAI"]

_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 7, 8, 6, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 5, 2, 8, 3]]


def verhoeff_ok(number: str) -> bool:
    if not re.fullmatch(r"\d{12}", number or ""):
        return False
    c = 0
    for i, d in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(d)]]
    return c == 0


def mask(number: str) -> str:
    n = re.sub(r"\D", "", number or "")[-4:] if number else ""
    return f"XXXX XXXX {n}" if len(n) == 4 else ""


_NUM = re.compile(r"(?<![0-9X])([0-9X]{4}) ?([0-9X]{4}) ?([0-9]{4})(?![0-9])")
_SKIP_NUM = re.compile(r"VID|ENROL|MOBILE|/|:\s*\d{4}/|PHONE|TEL", re.I)


def _numbers(lines: list[Line]) -> list[tuple[str, Line]]:
    """12-digit Aadhaar-number candidates (full, or masked 'XXXX XXXX 1234')."""
    out = []
    for l in lines:
        t = l.text.upper()
        if _SKIP_NUM.search(t):
            continue
        digits = re.sub(r"[^0-9X]", "", t)
        if len(re.sub(r"\D", "", t)) > 12 or len(digits) < 12:      # VID (16 digits) etc.
            continue
        s = re.sub(r"[^0-9X ]", "", t.translate(TO_DIGIT) if re.search(r"\d{4}", t) else t).strip()
        m = _NUM.search(s)
        if not m:
            continue
        n = "".join(m.groups())
        if n[0] in "01":                                           # Aadhaar never starts with 0/1
            continue
        out.append((n, l))
    return out


def _pick_number(cands):
    """Most frequent reading; one that passes the check digit wins over one that does not."""
    if not cands:
        return None, 0, None
    votes = Counter(n for n, _ in cands)
    best = max(votes, key=lambda n: (verhoeff_ok(n) or n.startswith("X"), votes[n],
                                     max(l.conf for m, l in cands if m == n)))
    line = max((l for m, l in cands if m == best), key=lambda l: l.conf)
    return best, votes[best], line


def _dob(lines: list[Line]) -> tuple[str, Line | None, str]:
    labelled = [l for l in lines if re.search(r"DOB|BIRTH|D\.O\.B", l.text.upper())]
    for l in labelled:
        d = parse_date(l.text)
        if d:
            return d, l, "date"
        m = re.search(r"(?:YOB|YEAR OF BIRTH)\D*((?:19|20)\d{2})", l.text.upper())
        if m:
            return m.group(1), l, "year"
    return "", None, ""


def _gender(lines: list[Line], near: Line | None) -> tuple[str, Line | None]:
    cands = []
    for l in lines:
        m = re.search(r"\b(FEMALE|MALE|TRANSGENDER)\b", l.text.upper())
        if m and len(re.sub(r"[^A-Z]", "", l.text.upper())) <= 14:      # 'पुरुष / MALE', not a sentence
            cands.append((m.group(1), l))
    if not cands:
        return "", None
    if near is not None:
        cands.sort(key=lambda c: (c[1].page != near.page, abs(c[1].y0 - near.y1)))
    return cands[0]


def _name_above(anchor: Line, lines: list[Line]) -> Line | None:
    """English name line just above the DOB line (card front / cut-out card)."""
    above = [l for l in lines if l.page == anchor.page and l.cy < anchor.cy - 0.4 * anchor.h
             and anchor.y0 - l.y1 <= 4 * anchor.h and abs(l.x0 - anchor.x0) <= 4 * anchor.h
             and name_like(l.text, min_words=1)]
    return max(above, key=lambda l: l.y0) if above else None


_CARE = re.compile(r"\b([SDWC])\s*[/\\]\s*[O0]\b\s*[:.,]?\s*(.*)", re.I)


def _care_of(lines: list[Line]):
    """'S/O: Ramesh Kumar,' -> ('S/O', 'Ramesh Kumar', rest-of-line, line)."""
    for l in lines:
        m = _CARE.search(l.text)
        if not m:
            continue
        name, _, rest = m.group(2).partition(",")
        name = clean_name(name)
        if name_like(name):
            return f"{m.group(1).upper()}/O", name, rest.strip(" ,"), l
    return None


_PIN = re.compile(r"(?<!\d)([1-9]\d{2}) ?(\d{3})(?!\d)")


def _pin_of(text: str) -> str:
    hits = _PIN.findall(text or "")
    return "".join(hits[-1]) if hits else ""


def _block_below(start: Line, lines: list[Line], first_rest: str = "") -> list[str]:
    """Address lines under `start` in its column, down to the line with the PIN code."""
    parts = [first_rest] if first_rest else []
    prev = start
    col = [l for l in lines if l.page == start.page and l.y0 > start.y0 + 0.4 * start.h
           and abs(l.x0 - start.x0) <= 3 * start.h]
    for l in sorted(col, key=lambda l: l.y0):
        if l.y0 - prev.y1 > 1.8 * prev.h:
            break
        t = l.text.strip()
        if re.match(r"(?i)mobile|phone|signature|vid\b", t) or _numbers([l]):
            break
        parts.append(t.strip(" ,"))
        prev = l
        if re.search(r"(?i)pin\s*code|\b[1-9]\d{5}\b\.?$", t):
            break
    return [p for p in parts if p]


def _address(lines: list[Line], care):
    """Prefer the English 'Address:' block (card back / letter back); else the
    letter's 'To ... S/O ...' block."""
    for l in lines:
        if re.fullmatch(r"(?i)\W*address\W*", l.text) or re.match(r"(?i)^address\s*:", l.text):
            rest = re.sub(r"(?i)^\W*address\W*", "", l.text)
            parts = _block_below(l, lines, rest)
            if parts and re.match(_CARE, parts[0]):                 # drop 'S/O <name>,' prefix
                parts[0] = _CARE.match(parts[0]).group(2).partition(",")[2].strip(" ,")
            parts = [p for p in parts if p]
            if parts and _pin_of(parts[-1]):
                return ", ".join(parts), "'Address:' block"
    if care:
        _, _, rest, l = care
        parts = _block_below(l, lines, rest)
        if parts and _pin_of(" ".join(parts)):
            return ", ".join(parts), "letter address block"
    return "", ""


def extract_aadhaar(ocr_pages: list[dict], show_full_number: bool = False) -> dict[str, Any]:
    lines = lines_of(ocr_pages)
    fields: dict[str, dict] = {}
    flags: dict[str, str] = {}
    checks: dict[str, Any] = {}

    number, votes, nline = _pick_number(_numbers(lines))
    masked = bool(number) and number.startswith("X")
    ok = bool(number) and not masked and verhoeff_ok(number)
    checks["number_check_digit"] = "masked on the document" if masked else ok
    checks["number_times_read"] = votes
    full = show_full_number and number and not masked
    shown = f"{number[:4]} {number[4:8]} {number[8:]}" if full else mask(number or "")
    checks["number_display"] = "full (holder's consent recorded)" if full else "masked"
    fields["aadhaar_number"] = field(shown, "printed number", nline.conf if nline else None, ok)
    if number and not masked and not ok:
        flags["aadhaar_number"] = ("the number fails the Aadhaar check digit - it was misread, or the "
                                   "document is not genuine; please check the original")

    dob, dline, kind = _dob(lines)
    fields["date_of_birth"] = field(dob, "printed text" if dob else "not found",
                                    dline.conf if dline else None, raw=dline.text if dline else None)
    if kind == "year":
        flags["date_of_birth"] = "only the year of birth is printed"

    g, gline = _gender(lines, dline)
    fields["gender"] = field(g, "printed text" if g else "not found", gline.conf if gline else None)

    care = _care_of(lines)
    # the name is printed above the DOB (card) and, on letters, above the S/O line too
    cands: list[tuple[Line, str]] = []
    if dline:
        l = _name_above(dline, lines)
        if l:
            cands.append((l, "above the date of birth"))
    if care:
        cl = care[3]
        prev = [l for l in lines if l.page == cl.page and l.cy < cl.cy - 0.4 * cl.h
                and cl.y0 - l.y1 <= 2 * cl.h and name_like(l.text)]
        if prev:
            cands.append((max(prev, key=lambda l: l.y0), "above the S/O line"))
    name_line, name_src, name = None, "not found", ""
    if cands:
        # same name printed again elsewhere (letter + cut-out card) confirms a reading
        def support(l):
            return sum(1 for x in lines if clean_name(x.text).upper() == clean_name(l.text).upper())
        name_line, name_src = max(cands, key=lambda c: (support(c[0]), c[0].conf))
        name = clean_name(name_line.text)
    fields["name"] = field(name, name_src, name_line.conf if name_line else None,
                           raw=name_line.text if name_line else None)
    if name:
        fields["name"]["verified"] = support(name_line) >= 2
        other = {clean_name(l.text) for l, _ in cands} - {name}
        if other and not fields["name"]["verified"]:
            flags["name"] = f"read as '{name}' and '{other.pop()}' in two places - please check"

    fields["care_of"] = field(care[1] if care else "", f"{care[0]} line" if care else "not printed",
                              care[3].conf if care else None, raw=care[3].text if care else None)
    fields["relation"] = field(care[0] if care else "", "printed text" if care else "not printed")

    addr, how = _address(lines, care)
    fields["address"] = field(addr, how or "not found",
                              raw=None)
    pin = _pin_of(addr)
    fields["pincode"] = field(pin, "from the address" if pin else "not found")
    notes = []
    if masked:
        notes.append("Masked Aadhaar: only the last 4 digits are printed on this document.")
    if not addr and not care:
        notes.append("The address is on the back of the card - upload the back side too.")
    notes = " ".join(notes)

    if name and not fields["name"]["verified"] and (name_line.conf or 0) < 0.9:
        flags.setdefault("name", f"low OCR confidence ({name_line.conf:.2f})")
    r = finish("aadhaar" if (number or dob or name) else "unknown", fields, flags,
               ["aadhaar_number", "name", "date_of_birth", "gender"], checks, notes)
    if dline is not None:
        r["_dob_line"] = (dline.page, (dline.x0, dline.y0, dline.x1, dline.y1))
    return r


def _reread_gender(images: list, pages: list[dict], dob_line, ocr_engine) -> str:
    """The gender line ('पुरुष / MALE') sits right under the DOB; small and faint on
    phone photos, the full-page pass can drop it. Re-read just that strip."""
    page, (x0, y0, x1, y1) = dob_line
    p = pages[page - 1]
    img = images[page - 1]
    img = img.rotate(p["rotation"], expand=True) if p["rotation"] else img
    h = y1 - y0
    box = (max(0, int(x0 - 2 * h)), int(y1), min(img.width, int(x1 + 4 * h)), min(img.height, int(y1 + 2.5 * h)))
    for t in ocr_engine.read_lines_low_threshold(img.crop(box)):
        m = re.search(r"\b(FEMALE|MALE|TRANSGENDER)\b", t.upper())
        if m:
            return m.group(1)
    return ""


def extract_aadhaar_images(images: list, ocr_engine, show_full_number: bool = False) -> dict[str, Any]:
    pages = [ocr_upright(img, ocr_engine, i + 1, KEYWORDS) for i, img in enumerate(images)]
    r = extract_aadhaar(pages, show_full_number)
    dob_line = r.pop("_dob_line", None)
    if not r["fields"]["gender"]["value"] and dob_line:
        g = _reread_gender(images, pages, dob_line, ocr_engine)
        if g:
            r["fields"]["gender"] = field(g, "re-read below the date of birth", None)
            reasons = r["review"]["reasons"]
            reasons.pop("gender", None)
            r["review"]["fields_requiring_review"] = list(reasons)
            blocking = [k for k in reasons if k in ("aadhaar_number", "name", "date_of_birth", "gender")]
            r["review"]["status"] = "auto" if not reasons else ("manual_required" if blocking
                                                                else "review_recommended")
    r["pages"] = len(images)
    return r
