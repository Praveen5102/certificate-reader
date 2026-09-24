"""AP / TS Open School marks tables.

    Sub. Code | Subject     | [Maximum Marks] | Theory | Practical | Total | Total Marks in Words
    302       | ENGLISH     | [100]           | 83     |           | 83    | EIGHT THREE
    312       | PHYSICS     | [100]           | 45     | 15        | 60    | SIX ZERO

Marks = the TOTAL column (theory + practical); the per-subject maximum is read
only when a 'Maximum' column is printed. Open-school exams are a single sitting,
so every row is recorded as paper I (same convention as the truth file).
Parsed from geometry only; activated when the header row is detected.
"""
from __future__ import annotations

import re

from ..annotation.layout import PageWords, Word, norm, visual_rows
from ..validation.normalize import to_int
from .table import Cell, TableRow, estimate_skew

_CODE = re.compile(r"^[1-4]\d\d$")


def _y(w: Word, slope: float) -> float:
    return w.cy - slope * w.cx


def _header(page: PageWords, slope: float) -> dict | None:
    for row in visual_rows(page.words):
        names = [norm(w.text) for w in row]
        if "THEORY" in names and "PRACTICAL" in names and "TOTAL" in names:
            prac = row[names.index("PRACTICAL")]
            totals = [w for w in row if norm(w.text) == "TOTAL" and w.cx > prac.cx]
            if not totals:
                continue
            total = min(totals, key=lambda w: w.cx)                 # first 'Total' right of 'Practical'
            theory = row[names.index("THEORY")]
            maxw = [w for w in page.words if norm(w.text) == "MAXIMUM" and w.cx < theory.cx
                    and abs(_y(w, slope) - _y(theory, slope)) < 4 * page.text_h]
            return {"y": _y(theory, slope), "theory_x": theory.bbox[0], "total_x": total.cx,
                    "max_x": maxw[0].cx if maxw else None}
    return None


def _code_rows(page: PageWords, slope: float, hdr: dict) -> list[Word]:
    """Subject codes: 3-digit numbers in the leftmost column (left of every marks column)."""
    right_limit = min(hdr["theory_x"], hdr["max_x"] if hdr["max_x"] is not None else 1e9) - 3 * page.text_h
    return sorted([w for w in page.words if _CODE.match(w.text.strip()) and w.cx < right_limit
                   and _y(w, slope) > hdr["y"] - page.text_h], key=lambda w: _y(w, slope))


def detect(page: PageWords) -> bool:
    slope = estimate_skew(page)
    hdr = _header(page, slope)
    return hdr is not None and len(_code_rows(page, slope, hdr)) >= 3


def parse(page: PageWords) -> list[TableRow]:
    slope = estimate_skew(page)
    h = page.text_h
    hdr = _header(page, slope)
    if hdr is None:
        return []
    codes = _code_rows(page, slope, hdr)
    rows: list[TableRow] = []
    for k, code in enumerate(codes):
        y0 = _y(code, slope) - 0.6 * h
        y1 = (_y(codes[k + 1], slope) - 0.6 * h) if k + 1 < len(codes) else _y(code, slope) + 2.5 * h
        band = [w for w in page.words if y0 < _y(w, slope) < y1]
        name_ws = sorted([w for w in band if w.bbox[0] > code.bbox[2] and w.bbox[2] < hdr["theory_x"]
                          and abs(_y(w, slope) - _y(code, slope)) < 0.7 * h
                          and to_int(w.text)[0] is None and any(c.isalpha() for c in w.text)
                          and norm(w.text) not in {"MARKS", "MAXIMUM"}], key=lambda w: w.bbox[0])
        nums = [w for w in band if to_int(w.text)[0] is not None and w.bbox[0] > code.bbox[2]]
        if not name_ws or not nums:
            continue
        total = min(nums, key=lambda w: abs(w.cx - hdr["total_x"]))
        if abs(total.cx - hdr["total_x"]) > 3 * h:
            continue
        mx = None
        if hdr["max_x"] is not None:
            cand = min(nums, key=lambda w: abs(w.cx - hdr["max_x"]))
            if abs(cand.cx - hdr["max_x"]) <= 3 * h and cand is not total:
                mx = cand
        rows.append(TableRow(Cell(name_ws, "SUBJECT", min(w.conf for w in name_ws)), "I",
                             "open school table (single sitting; recorded as paper I)",
                             Cell([total], "MARKS", total.conf),
                             Cell([mx], "MAX_MARKS", mx.conf) if mx else None))
    return rows


def plausible(rows: list[TableRow]) -> bool:
    if len(rows) < 3:
        return False
    for r in rows:
        mk = to_int(r.marks.text)[0]
        mx = to_int(r.maximum.text)[0] if r.maximum else 100
        if mk is None or mx is None or mk > mx:
            return False
    return True
