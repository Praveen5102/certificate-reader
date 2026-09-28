"""Indian passport extraction: MRZ (verified by check digits) + printed labels.

Labels are bilingual ("जन्म तिथि / Date of Birth"); OCR turns the Hindi part
into noise ("afe/ Date of Birth"), so labels are matched on the English text
after the last '/'. Values are printed BELOW their label, starting inside the
label's horizontal span (several labels share a row: Place of Issue / Date of
Issue ... Date of Expiry).

MRZ values are authoritative when their check digits pass; printed values
cross-check them. Disagreements are flagged, never silently resolved.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from rapidfuzz import fuzz

from .mrz import TO_DIGIT, MRZResult, read_mrz

LABELS: dict[str, list[str]] = {
    "type": ["Type"],
    "code": ["Code", "Country Code"],
    "nationality": ["Nationality"],
    "passport_number": ["Passport No", "Passport No.", "Passport Number"],
    "surname": ["Surname"],
    "given_names": ["Given Name(s)", "Given Names", "Given Name"],
    "sex": ["Sex"],
    "date_of_birth": ["Date of Birth"],
    "place_of_birth": ["Place of Birth"],
    "place_of_issue": ["Place of Issue"],
    "date_of_issue": ["Date of Issue"],
    "date_of_expiry": ["Date of Expiry"],
    "father_name": ["Name of Father / Legal Guardian", "Legal Guardian", "Name of Father"],
    "mother_name": ["Name of Mother"],
    "spouse_name": ["Name of Spouse"],
    "address": ["Address"],
    "old_passport": ["Old Passport No. with Date and Place of Issue", "Date and Place of Issue",
                     "Old Passport No"],
    "file_number": ["File No", "File No.", "File Number"],
}

OUTPUT_FIELDS = ["passport_number", "surname", "given_names", "nationality", "sex", "date_of_birth",
                 "place_of_birth", "place_of_issue", "date_of_issue", "date_of_expiry", "father_name",
                 "mother_name", "spouse_name", "address", "file_number"]
# 1 letter + 7 digits (older) or 2 letters + 6 digits (2025+ e-passports)
PASSPORT_NO = re.compile(r"\b([A-Z][0-9OIlSBZ]{7}|[A-Z]{2}[0-9OIlSBZ]{6})\b")


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _english(text: str) -> str:
    return text.rsplit("/", 1)[-1].strip() if "/" in text else text.strip()


def _label_key(text: str) -> tuple[str | None, float]:
    """Best label for an OCR line (English part), with its score."""
    cands = {_norm(_english(text)), _norm(text)}
    if "/" in text:                              # 'Name of Father / Legal Guardian'
        cands.add(_norm(text.split("/", 1)[-1]))
    best, score = None, 0.0
    prefix_of: set[str] = set()
    for key, phrases in LABELS.items():
        for ph in phrases:
            for t in {_norm(ph), _norm(ph).replace("RN", "M")}:      # 'Sumame': rn read as m
                for c in cands:
                    if not c:
                        continue
                    s = fuzz.ratio(c, t)
                    if len(c) > len(t) + 3 and len(t) >= 6:     # label + trailing noise
                        s = max(s, fuzz.partial_ratio(t, c) - 5)
                    if s > score:
                        best, score = key, s
                    if len(c) >= 7 and t.startswith(c) and len(c) < len(t):
                        prefix_of.add(key)
    if score < 82 and len(prefix_of) == 1:      # label cut short: 'Place of I' -> Place of Issue
        return prefix_of.pop(), 82.0
    return (best, score) if score >= 82 else (None, score)


class _Line:
    __slots__ = ("text", "x0", "y0", "x1", "y1", "conf", "page", "label", "label_score")

    def __init__(self, d: dict, page: int):
        self.text = d["text"]
        self.x0, self.y0, self.x1, self.y1 = d["bbox"]
        self.conf = d["confidence"]
        self.page = page
        self.label, self.label_score = _label_key(self.text)

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def boundary(self) -> bool:
        """A label (even a badly read one) or an MRZ line: never part of a value."""
        return bool(self.label) or self.label_score >= 70 or _is_mrz(self.text)


def _is_mrz(text: str) -> bool:
    return text.count("<") >= 3 and len(text.replace(" ", "")) >= 30


def _in_column(label: _Line, l: _Line) -> bool:
    h = label.h
    return label.x0 - 2.5 * h <= l.x0 <= max(label.x1, label.x0 + 4 * h)


def _value_lines(label: _Line, lines: list[_Line], max_lines: int = 1) -> list[_Line]:
    """Lines printed below `label` in its column, and above the next label in that
    column - so an empty field (e.g. no spouse) stays empty instead of taking the
    next field's value."""
    h = label.h
    page = [l for l in lines if l.page == label.page and l is not label]
    stop = min((l.y0 for l in page if l.boundary and _in_column(label, l) and l.y0 > label.y0 + 0.5 * h),
               default=float("inf"))
    # below the label (by vertical centre: slanted phone photos give tall, overlapping
    # boxes), close to it, and above the next label
    cands = [l for l in page if not l.boundary and _in_column(label, l)
             and (l.y0 + l.y1) / 2 > (label.y0 + label.y1) / 2 + 0.3 * h
             and l.y0 <= label.y1 + 1.5 * h and l.y0 < stop - 0.2 * h]
    if not cands:
        return []
    first = min(cands, key=lambda l: (l.y0, abs(l.x0 - label.x0)))
    out = [first]
    while len(out) < max_lines:          # address: continuation lines up to the next label
        prev = out[-1]
        nxt = [l for l in page if not l.boundary and l not in out and l.y0 < stop - 0.2 * h
               and prev.y1 - 0.3 * prev.h <= l.y0 <= prev.y1 + 2.5 * prev.h and abs(l.x0 - first.x0) <= 3 * h]
        if not nxt:
            break
        out.append(min(nxt, key=lambda l: l.y0))
    return out


def _value_above(label: _Line, lines: list[_Line]) -> list[_Line]:
    """The value line just above `label` (used when the previous field's own label
    was too faint to read: its value still sits right above the next label)."""
    h = label.h
    c = [l for l in lines if l.page == label.page and not l.boundary and _in_column(label, l)
         and label.y0 - 2.5 * h <= l.y1 <= label.y0 + 0.4 * h and l.y0 < label.y0
         and not any(b.boundary and b.page == label.page and b is not label and _in_column(label, b)
                     and l.y0 < b.y0 < label.y0 for b in lines)]
    return [max(c, key=lambda l: l.y0)] if c else []


def _date(text: str) -> str:
    s = re.sub(r"[^0-9/]", "", (text or "").translate(TO_DIGIT) if re.search(r"\d", text or "") else "")
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", s)
    if not m:
        return ""
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
    except ValueError:
        return ""


def _clean_value(key: str, text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip(" ,.:;-")
    if key == "nationality":
        t = _english(t)
        if _norm(t) in ("IN", "IND", "INDIA") or fuzz.ratio(_norm(t), "INDIAN") >= 75:
            return "INDIAN"
    if key in ("surname", "given_names", "father_name", "mother_name", "spouse_name", "place_of_birth",
               "place_of_issue", "address"):
        # '0' inside a word of letters is the letter O ('R0AD' -> 'ROAD')
        t = re.sub(r"\b(?=[A-Z0]*[A-Z])[A-Z0]+\b", lambda m: m.group(0).replace("0", "O"), t)
    if key == "address":
        t = re.sub(r"PIN\W*([0-9OSIB]{6})\b", lambda m: "PIN:" + m.group(1).translate(TO_DIGIT), t)
    if key == "sex":
        m = re.search(r"\b([MFX])\b", t.upper())
        return m.group(1) if m else ""
    if key.startswith("date_"):
        return _date(t)
    if key == "passport_number":
        m = PASSPORT_NO.search(t.upper().replace(" ", ""))
        if not m:
            return ""
        v = m.group(1)
        n_letters = 2 if v[1].isalpha() else 1      # 2025+ numbers start with 2 letters, e.g. 'AB123456'
        return v[:n_letters] + v[n_letters:].translate(TO_DIGIT)
    if key == "file_number":
        return re.sub(r"[^A-Z0-9]", "", t.upper())
    return t


def extract_passport(ocr_pages: list[dict], extra_lines: list[str] | None = None) -> dict[str, Any]:
    """ocr_pages: OCR output per page. extra_lines: optional text from a second,
    MRZ-focused OCR pass (bottom strip read with a lower confidence cut-off)."""
    lines = [_Line(l, p["page"]) for p in ocr_pages for l in p["lines"]]
    mrz: MRZResult | None = read_mrz([l.text for l in lines] + list(extra_lines or []))

    visual: dict[str, dict] = {}

    def keep(key, vals, how="below its label"):
        visual[key] = {"text": ", ".join(v.text for v in vals), "confidence": round(min(v.conf for v in vals), 4),
                       "page": vals[0].page, "found": how,
                       "bbox": [min(v.x0 for v in vals), min(v.y0 for v in vals),
                                max(v.x1 for v in vals), max(v.y1 for v in vals)]}

    for key in LABELS:
        labels = [l for l in lines if l.label == key]
        for lab in sorted(labels, key=lambda l: -l.label_score):
            vals = _value_lines(lab, lines, max_lines=5 if key == "address" else 1)
            if key == "sex" and vals and not re.search(r"\b[MFX]\b", vals[0].text.upper()):
                continue
            if vals:
                keep(key, vals)
                break
    # a faint label that OCR missed: its value still sits right above the next label
    for key, nxt in (("father_name", "mother_name"), ("mother_name", "spouse_name")):
        if key not in visual and not any(l.label == key for l in lines):
            for lab in (l for l in lines if l.label == nxt):
                vals = _value_above(lab, lines)
                if vals and not (nxt in visual and vals[0].text == visual[nxt]["text"]):
                    keep(key, vals, how="above the next label (own label unreadable)")
                    break
    if "address" not in visual and not any(l.label == "address" for l in lines):
        used = {v["text"] for v in visual.values()}
        for lab in (l for l in lines if l.label in ("old_passport", "file_number")):
            block, cur = [], lab
            while len(block) < 5 and (up := _value_above(cur, lines)) and up[0].text not in used:
                block.insert(0, up[0])
                cur = up[0]
            if block:
                keep("address", block, how="above 'Old Passport No.' (address label unreadable)")
                break
    # passport number printed without a readable label: the pattern itself identifies it
    if "passport_number" not in visual:
        hits = [(l, m.group(1)) for l in lines if not _is_mrz(l.text) and not l.label
                for m in [PASSPORT_NO.search(l.text.upper().replace(" ", ""))] if m and len(_norm(l.text)) <= 10]
        if hits:
            l, _ = hits[0]
            visual["passport_number"] = {"text": l.text, "confidence": l.conf, "page": l.page,
                                         "bbox": [l.x0, l.y0, l.x1, l.y1]}

    fields: dict[str, dict] = {}
    flags: dict[str, str] = {}

    def put(key, value, source, conf=None, verified=False, raw=None):
        fields[key] = {"value": value, "source": source, "confidence": conf, "verified": verified, "raw": raw}

    # the date of issue has no MRZ copy; if its label was unreadable, it is the one
    # printed date that is neither the birth nor the expiry date and lies between them
    if "date_of_issue" not in visual:
        known = {_clean_value(k, visual[k]["text"]) for k in ("date_of_birth", "date_of_expiry") if k in visual}
        if mrz:
            known |= {mrz.date_of_birth, mrz.date_of_expiry}
        lo, hi = min(known - {""}, default=""), max(known - {""}, default="")
        others = {}
        for l in lines:
            d = _date(l.text) if not l.boundary else ""
            if d and d not in known and lo < d < hi:
                others.setdefault(d, l)
        if len(others) == 1 and lo and hi:
            keep("date_of_issue", list(others.values()), how="the remaining printed date (label unreadable)")

    mrz_map = {}
    if mrz:
        comp = mrz.checks.get("composite")
        nat_ok = comp and len(mrz.nationality) == 3 and mrz.nationality.isalpha()
        mrz_map = {"passport_number": (mrz.passport_number, mrz.checks.get("passport_number")),
                   "surname": (mrz.surname, comp), "given_names": (mrz.given_names, comp),
                   "nationality": ("INDIAN" if mrz.nationality == "IND" else mrz.nationality, nat_ok),
                   "sex": (mrz.sex, comp),
                   "date_of_birth": (mrz.date_of_birth, mrz.checks.get("date_of_birth")),
                   "date_of_expiry": (mrz.date_of_expiry, mrz.checks.get("date_of_expiry"))}

    for key in OUTPUT_FIELDS:
        vis = visual.get(key)
        vtext = vis["text"] if vis else ""
        vval = _clean_value(key, vtext) if vis else ""
        m = mrz_map.get(key)
        if m and m[0] and m[1]:
            agrees = bool(vval) and (_norm(vval) == _norm(m[0]) or fuzz.ratio(_norm(vval), _norm(m[0])) >= 85)
            unprotected = key in ("surname", "given_names", "nationality", "sex")
            put(key, m[0], "mrz", 1.0, agrees if unprotected else True, raw=vtext or None)
            if vval and _norm(vval) != _norm(m[0]) and fuzz.ratio(_norm(vval), _norm(m[0])) < 85 \
                    and key not in ("nationality",):
                flags[key] = f"printed '{vtext}' differs from the passport code lines - please check"
        elif vval:
            put(key, vval, "printed text", vis["confidence"], False, raw=vtext)
            if m and m[0]:
                if _norm(vval) != _norm(m[0]):
                    flags[key] = f"code line reads '{m[0]}' (not confirmed by its check digits); printed value used - check"
            elif (vis["confidence"] or 0) < 0.8:
                flags[key] = f"low OCR confidence ({vis['confidence']:.2f})"
        elif m and m[0]:
            put(key, m[0], "mrz", None, False)
            flags[key] = "passport code-line check digit failed - verify"
        else:
            put(key, "", "not found")

    # Indian passports carry the file number's digits in the MRZ optional-data field
    # (e.g. file HY1012345678901 <-> MRZ ...1012345678901). A match verifies it.
    fn = fields["file_number"]
    if mrz and mrz.personal_number and len(mrz.personal_number) >= 8:
        pn = _norm(mrz.personal_number)
        if fn["value"] and _norm(fn["value"]).endswith(pn):
            fn["verified"] = True
            fn["source"] = "printed text + code lines"
            flags.pop("file_number", None)
        elif fn["value"] and fuzz.ratio(_norm(fn["value"])[-len(pn):], pn) >= 80:
            flags["file_number"] = f"differs from the code lines ({pn}) - please check"

    # cross-field checks (flag only)
    dob, doi, doe = (fields[k]["value"] for k in ("date_of_birth", "date_of_issue", "date_of_expiry"))
    if dob and doi and doi <= dob:
        flags["date_of_issue"] = "date of issue is not after date of birth"
    if doi and doe:
        years = (dt.date.fromisoformat(doe) - dt.date.fromisoformat(doi)).days / 365.25
        if doe <= doi or not (4.5 <= years <= 10.1):
            flags["date_of_expiry"] = f"expiry is {years:.1f} years after issue (Indian passports: 5 or 10)"
    if doe and doe < dt.date.today().isoformat():
        flags.setdefault("date_of_expiry", "passport has expired")

    required = ["passport_number", "surname", "given_names", "date_of_birth", "date_of_expiry"]
    no_surname = bool(mrz and mrz.valid and not mrz.surname and mrz.given_names
                      and not fields["surname"]["value"])
    if no_surname:
        put("surname", "", "mrz (no surname on this passport)", 1.0, False)
    for k in required:
        if not fields[k]["value"] and not (k == "surname" and no_surname):
            flags[k] = "not found - enter manually"
    if not any(fields[k]["value"] for k in ("father_name", "mother_name", "address")):
        flags.setdefault("address", "last page not found - upload the last page for parents' names and address")
    blocking = [k for k in flags if k in required and not (fields[k]["value"] and fields[k]["verified"])]
    status = "auto" if not flags else ("manual_required" if blocking else "review_recommended")
    return {
        "document_type": "passport" if (mrz or visual) else "unknown",
        "mrz_found": mrz is not None,
        "mrz_valid": bool(mrz and mrz.valid),
        "mrz_lines": list(mrz.raw) if mrz else [],
        "mrz_checks": mrz.checks if mrz else {},
        "mrz_repairs": mrz.repairs if mrz else [],
        "fields": fields,
        "review": {"status": status, "fields_requiring_review": list(flags), "reasons": flags},
        "notes": "" if mrz else "The two code lines at the bottom of the photo page could not be read; "
                               "values come from the printed text only.",
    }


def _page_quality(lines: list[dict], ocr_engine) -> tuple:
    """How 'upright' an OCR result looks: horizontal text boxes, passport labels
    found (or MRZ lines), then confidently-read text."""
    labels = sum(1 for l in lines if _label_key(l["text"])[0]) + 3 * sum(1 for l in lines if _is_mrz(l["text"]))
    return (ocr_engine._horizontal_fraction(lines) >= 0.5, labels, ocr_engine._mean_conf_len(lines))


def _ocr_upright(img, ocr_engine, page: int) -> dict:
    """OCR a passport page in the orientation where the label layout makes sense.
    Phone scans come sideways or upside down; the recogniser still reads the text
    of a sideways page, but label/value positions would then be meaningless, and
    an upside-down page reads as noise while its boxes still look horizontal."""
    p = ocr_engine.ocr_page(img, page=page)
    q = _page_quality(p["lines"], ocr_engine)
    if q[0] and q[1] >= 4:
        return p
    for rot in (0, 90, 270, 180):
        if rot == p["rotation"]:
            continue
        cand = img.rotate(rot, expand=True) if rot else img
        lines, words = ocr_engine._run(cand)
        cq = _page_quality(lines, ocr_engine)
        if cq > q:
            q = cq
            p = {**p, "width": cand.width, "height": cand.height, "rotation": rot, "lines": lines, "words": words}
    return p


def _mrz_second_pass(images: list, ocr_pages: list[dict], ocr_engine) -> list[str]:
    """Re-read the MRZ when the full-page pass missed it: the text detector often
    skips the widely spaced code lines on a whole page, but reads them from a
    narrow horizontal strip. Photo pages (most labels) first; strips slide over
    the lower half of the upright page; stops once the check digits pass."""
    order = sorted(range(len(images)), key=lambda i: -sum(1 for l in ocr_pages[i]["lines"] if _label_key(l["text"])[0]))
    extra: list[str] = []
    for i in order:
        p = ocr_pages[i]
        up = images[i].rotate(p["rotation"], expand=True) if p["rotation"] else images[i]
        w, h = up.size
        crops = [(0, int(h * 0.5), w, h)]
        crops += [(0, int(h * y), w, min(h, int(h * (y + 0.15))))
                  for y in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)]
        for box in crops:
            # two scales: the recogniser can merge repeated characters ('0011100' ->
            # '00110') at one scale and not the other; the check digits pick the right read
            for scale in (1.0, 0.5):
                c = up.crop(box)
                if scale != 1.0:
                    c = c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))))
                extra += [t for t in ocr_engine.read_lines_low_threshold(c) if len(t.replace(" ", "")) >= 30]
                found = read_mrz(extra)
                if found is not None and found.valid:
                    return extra
    return extra


def extract_passport_images(images: list, ocr_engine) -> dict[str, Any]:
    """Full flow from page images: OCR every page; if no MRZ was read, OCR the
    lower half of each (upright) page again with a low score cut-off, because
    the '<<<<' filler lines are often dropped as low-confidence text."""
    ocr_pages = [_ocr_upright(img, ocr_engine, page=i + 1) for i, img in enumerate(images)]
    extra: list[str] = []
    mrz = read_mrz([l["text"] for p in ocr_pages for l in p["lines"]])
    if mrz is None or not mrz.valid:
        extra = _mrz_second_pass(images, ocr_pages, ocr_engine)
    result = extract_passport(ocr_pages, extra)
    result["pages"] = len(images)
    return result


def flat(result: dict) -> dict:
    return {k: v["value"] for k, v in result["fields"].items()} | {"document_type": result["document_type"]}
