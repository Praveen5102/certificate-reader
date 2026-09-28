"""Indian passport extraction: MRZ (verified by check digits) + printed labels.

Photo page labels (bilingual; the English part is what OCR reads):
    Passport No.  Surname  Given Name(s)  Nationality  Sex  Date of Birth
    Place of Birth  Place of Issue  Date of Issue  Date of Expiry
Last (address) page:
    Name of Father / Legal Guardian   Name of Mother   Name of Spouse
    Address   Old Passport No. ...   File No.

Values are printed BELOW their label, left-aligned with it. MRZ values are
authoritative when their check digits pass; printed values are then used to
cross-check them, and any disagreement is flagged, never silently resolved.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from rapidfuzz import fuzz

from ..annotation.layout import PageWords, Word, load_pages, norm
from .mrz import MRZResult, read_mrz

# label key -> phrases (normalized matching, tolerant to OCR noise)
LABELS: dict[str, list[str]] = {
    "passport_number": ["Passport No", "Passport Number"],
    "surname": ["Surname"],
    "given_names": ["Given Name(s)", "Given Names", "Given Name"],
    "nationality": ["Nationality"],
    "sex": ["Sex"],
    "date_of_birth": ["Date of Birth"],
    "place_of_birth": ["Place of Birth"],
    "place_of_issue": ["Place of Issue"],
    "date_of_issue": ["Date of Issue"],
    "date_of_expiry": ["Date of Expiry"],
    "father_name": ["Name of Father / Legal Guardian", "Name of Father", "Father / Legal Guardian"],
    "mother_name": ["Name of Mother"],
    "spouse_name": ["Name of Spouse"],
    "address": ["Address"],
    "file_number": ["File No", "File Number"],
    "old_passport": ["Old Passport No", "Old Passport Number"],
}
ALL_LABEL_WORDS = {norm(w) for phs in LABELS.values() for p in phs for w in p.split()} | {
    "TYPE", "COUNTRY", "CODE", "REPUBLIC", "INDIA", "OF", "NAME", "WITH", "AND", "PLACE", "DATE"}

OUTPUT_FIELDS = ["passport_number", "surname", "given_names", "nationality", "sex", "date_of_birth",
                 "place_of_birth", "place_of_issue", "date_of_issue", "date_of_expiry", "father_name",
                 "mother_name", "spouse_name", "address", "file_number"]


def _to_iso(text: str) -> str:
    m = re.search(r"(\d{2})[/\-. ](\d{2})[/\-. ](\d{4})", text or "")
    if not m:
        return ""
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
    except ValueError:
        return ""


def _find_label(page: PageWords, phrases: list[str]) -> list[Word] | None:
    """Best fuzzy match of a label phrase within one OCR line."""
    best = None
    by_line: dict[int, list[Word]] = {}
    for w in page.words:
        by_line.setdefault(w.line_id, []).append(w)
    for ws in by_line.values():
        ws = sorted(ws, key=lambda w: w.bbox[0])
        for ph in phrases:
            target = norm(ph)
            n = len(ph.split())
            for i in range(len(ws)):
                for j in range(i, min(i + n + 2, len(ws))):
                    span = ws[i:j + 1]
                    s = fuzz.ratio(norm("".join(w.text for w in span)), target)
                    if s >= 85 and (best is None or s > best[0]):
                        best = (s, span)
    return best[1] if best else None


def _value_below(page: PageWords, label: list[Word], max_lines: int = 1) -> list[list[Word]]:
    """Text lines directly below the label and left-aligned with it."""
    x0 = min(w.bbox[0] for w in label)
    y1 = max(w.bbox[3] for w in label)
    h = max(w.h for w in label)
    by_line: dict[int, list[Word]] = {}
    for w in page.words:
        by_line.setdefault(w.line_id, []).append(w)
    lines = []
    for ws in by_line.values():
        ws = sorted(ws, key=lambda w: w.bbox[0])
        top = min(w.bbox[1] for w in ws)
        if y1 - 0.3 * h <= top <= y1 + (2.2 + 1.6 * (max_lines - 1)) * h and abs(ws[0].bbox[0] - x0) <= 3 * h:
            # stop at the next label line
            if all(norm(w.text) in ALL_LABEL_WORDS or not norm(w.text) for w in ws):
                continue
            lines.append((top, ws))
    lines.sort(key=lambda t: t[0])
    out = []
    for _, ws in lines[:max_lines]:
        if out and any(norm(w.text) in {"FILE", "OLD", "NAME"} for w in ws[:2]):
            break
        out.append(ws)
    return out


def _text(lines: list[list[Word]]) -> str:
    return ", ".join(" ".join(w.text for w in ws) for ws in lines).strip(" ,:")


def _conf(lines: list[list[Word]]) -> float | None:
    confs = [w.conf for ws in lines for w in ws]
    return round(min(confs), 4) if confs else None


def extract_passport(ocr_pages: list[dict]) -> dict[str, Any]:
    pages = load_pages(ocr_pages)
    all_lines = [l["text"] for p in ocr_pages for l in p["lines"]]
    mrz: MRZResult | None = read_mrz(all_lines)

    visual: dict[str, dict] = {}
    for key, phrases in LABELS.items():
        for p in pages:
            lab = _find_label(p, phrases)
            if not lab:
                continue
            lines = _value_below(p, lab, max_lines=3 if key == "address" else 1)
            if lines:
                visual[key] = {"text": _text(lines), "confidence": _conf(lines), "page": p.page,
                               "bbox": [min(w.bbox[0] for ws in lines for w in ws),
                                        min(w.bbox[1] for ws in lines for w in ws),
                                        max(w.bbox[2] for ws in lines for w in ws),
                                        max(w.bbox[3] for ws in lines for w in ws)]}
                break

    fields: dict[str, dict] = {}
    flags: dict[str, str] = {}

    def put(key, value, source, conf=None, verified=False, raw=None):
        fields[key] = {"value": value, "source": source, "confidence": conf, "verified": verified, "raw": raw}

    mrz_map = {}
    if mrz:
        mrz_map = {"passport_number": (mrz.passport_number, mrz.checks.get("passport_number")),
                   "surname": (mrz.surname, mrz.checks.get("composite")),
                   "given_names": (mrz.given_names, mrz.checks.get("composite")),
                   "nationality": (mrz.nationality, mrz.checks.get("composite")),
                   "sex": (mrz.sex, mrz.checks.get("composite")),
                   "date_of_birth": (mrz.date_of_birth, mrz.checks.get("date_of_birth")),
                   "date_of_expiry": (mrz.date_of_expiry, mrz.checks.get("date_of_expiry"))}

    for key in OUTPUT_FIELDS:
        vis = visual.get(key)
        vtext = vis["text"] if vis else ""
        if key.startswith("date_"):
            vval = _to_iso(vtext)
        elif key == "passport_number":
            vval = re.sub(r"[^A-Z0-9]", "", vtext.upper())
        elif key == "sex":
            vval = (re.search(r"\b([MF])\b", vtext.upper()) or [None, ""])[1] if vtext else ""
        else:
            vval = re.sub(r"\s+", " ", vtext).strip()
        if key in mrz_map and mrz_map[key][0]:
            mval, ok = mrz_map[key]
            put(key, mval, "mrz", 1.0 if ok else None, bool(ok), raw=vtext or None)
            if not ok:
                flags[key] = "MRZ check digit failed - verify against the passport"
            elif vval and key != "nationality" and norm(vval) != norm(mval) and \
                    fuzz.ratio(norm(vval), norm(mval)) < 90:
                flags[key] = f"printed value '{vtext}' differs from the MRZ value"
        elif vval:
            put(key, vval, "printed text", vis["confidence"], False, raw=vtext)
            if (vis["confidence"] or 0) < 0.85:
                flags[key] = f"low OCR confidence ({vis['confidence']})"
        else:
            put(key, "", "not found")

    # cross-field checks (flag only)
    dob, doi, doe = (fields[k]["value"] for k in ("date_of_birth", "date_of_issue", "date_of_expiry"))
    if dob and doi and doi <= dob:
        flags["date_of_issue"] = "date of issue is not after date of birth"
    if doi and doe and doe <= doi:
        flags["date_of_expiry"] = "date of expiry is not after date of issue"
    if doe and doe < dt.date.today().isoformat():
        flags.setdefault("date_of_expiry", "passport has expired")

    required = ["passport_number", "surname", "given_names", "date_of_birth", "date_of_expiry"]
    for k in required:
        if not fields[k]["value"]:
            flags[k] = "not found - enter manually"
    status = "auto"
    if flags:
        status = "manual_required" if any(not fields[k]["value"] or (fields[k]["source"] == "mrz" and
                                                                    not fields[k]["verified"]) for k in flags
                                          if k in fields) else "review_recommended"
    return {
        "document_type": "passport" if mrz or visual else "unknown",
        "mrz_found": mrz is not None,
        "mrz_valid": bool(mrz and mrz.valid),
        "mrz_lines": list(mrz.raw) if mrz else [],
        "mrz_checks": mrz.checks if mrz else {},
        "mrz_repairs": mrz.repairs if mrz else [],
        "fields": fields,
        "review": {"status": status, "fields_requiring_review": list(flags), "reasons": flags},
        "notes": "" if mrz else "No passport machine-readable zone found. Upload the photo page "
                               "(and the last page for address / parents' names).",
    }


def flat(result: dict) -> dict:
    return {k: v["value"] for k, v in result["fields"].items()} | {"document_type": result["document_type"]}
