"""Transposed ("sideways") marks tables.

Used by the TS online results memo and the older TS "Memorandum of Marks":
subjects are COLUMN headers, and each year is a block of two rows
("Max Marks" / "Marks Secured") labelled "Ist YEAR", "2nd YEAR", "PRACTICALS".

    YEAR  SUBJECT        ENGLISH           SANSKRIT  MATHS-(A) ...
                     THEORY  PRACTICAL
    Ist   Max Marks      80      20           100       075
    YEAR  Marks Secured  47* P   15* P        58* P     62* P
    2nd   Max Marks      ...

Parsed purely from geometry (no tagger needed), so it works whether the model
or the rules are in use. Only activated when the layout is detected.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..annotation.layout import PageWords, Word, norm, visual_rows
from ..validation.normalize import to_int
from .table import Cell, TableRow, estimate_skew

_SKIP_HEADER = re.compile(r"^(PAR[TE]\W*(I{1,3}|[1-4])?\W*|-?\(?(I{1,3}|[123])\)?:?|YEAR|SUBJECT|:)$", re.I)


def _y(w: Word, slope: float) -> float:
    return w.cy - slope * w.cx


def _label_rows(page: PageWords, slope: float, first: str, second: str) -> list[tuple[float, list[Word]]]:
    """Left-side two-word labels ('Marks Secured', 'Max Marks') -> (deskewed y, words)."""
    out = []
    for row in visual_rows(page.words):
        for a, b in zip(row, row[1:]):
            # tolerant: OCR reads 'Max Marias', 'Marka Secured'
            if fuzz.ratio(norm(a.text), first) >= 75 and fuzz.ratio(norm(b.text), second) >= 70                     and a.bbox[0] < page.width * 0.45:
                out.append(((_y(a, slope) + _y(b, slope)) / 2, [a, b]))
    return out


def _local_h(labels: list[tuple[float, list[Word]]], default: float) -> float:
    """Text height of the table itself (its print can be much smaller than the page's)."""
    hs = sorted(w.h for _, ws in labels for w in ws if w.h > 0)
    return hs[len(hs) // 2] if hs else default


def detect(page: PageWords) -> bool:
    """>= 2 'Marks Secured' labels stacked vertically at the left = transposed table.
    ('Max Marks' labels are not required: OCR often drops the small 'Max'.)
    The standard layout prints 'Marks Secured' twice side by side, so it never matches."""
    slope = estimate_skew(page)
    secured = _label_rows(page, slope, "MARKS", "SECURED")
    if len(secured) < 2:
        return False
    h = _local_h(secured, page.text_h)
    xs = [s[1][0].bbox[0] for s in secured]
    ys = sorted(s[0] for s in secured)
    return (max(xs) - min(xs)) < page.width * 0.1 and (ys[-1] - ys[0]) > 2 * h


_YEAR_RE = {"PRACTICALS": ("II", "practicals"), "PRACTICAL": ("II", "practicals"), "IST": ("I", "year"),
            "1ST": ("I", "year"), "FIRST": ("I", "year"), "2ND": ("II", "year"), "IIND": ("II", "year"),
            "SECOND": ("II", "year")}


def _year_label(page: PageWords, slope: float, y_mid: float, x_max: float) -> tuple[str, str]:
    """Nearest year label ('Ist', '2nd', 'PRACTICALS') left of the row labels."""
    cands = [(abs(_y(w, slope) - y_mid), _YEAR_RE[norm(w.text)]) for w in page.words
             if w.bbox[2] <= x_max and norm(w.text) in _YEAR_RE]
    cands = [c for c in cands if c[0] <= 3 * page.text_h]
    return min(cands)[1] if cands else ("", "")


def parse(page: PageWords) -> list[TableRow]:
    slope = estimate_skew(page)
    secured = sorted(_label_rows(page, slope, "MARKS", "SECURED"), key=lambda s: s[0])
    maxes = sorted(_label_rows(page, slope, "MAX", "MARKS"), key=lambda s: s[0])
    h = _local_h(secured, page.text_h)
    label_right = max(w.bbox[2] for _, ws in secured + maxes for w in ws)

    def numbers_between(y0: float, y1: float) -> list[Word]:
        ws = [w for w in page.words if w.bbox[0] > label_right and y0 < _y(w, slope) < y1
              and to_int(w.text)[0] is not None]
        return sorted(ws, key=lambda w: w.cx)

    # the first data row: first 'Max Marks' label, or (when OCR missed it) just above the first secured row
    first_max_y = min(maxes[0][0], secured[0][0] - 1.2 * h) if maxes else secured[0][0] - 1.2 * h
    # header band: from the 'PART-I' / 'SUBJECT' header row down to the first 'Max Marks' row
    tops = [_y(w, slope) for w in page.words if first_max_y - 6 * h < _y(w, slope) < first_max_y
            and (re.match(r"(?i)^PART", w.text) or norm(w.text) == "SUBJECT")]
    top = (min(tops) - 0.6 * h) if tops else first_max_y - 3 * h
    header_words = [w for w in page.words if w.bbox[0] > label_right - h
                    and top < _y(w, slope) < first_max_y - 0.6 * h
                    and not _SKIP_HEADER.match(w.text.strip())
                    and to_int(w.text)[0] is None and norm(w.text) != "P"]
    rows: list[TableRow] = []
    max_ys = [m[0] for m in maxes]
    blocks = []
    for sy, _ in secured:
        above = [y for y in max_ys if sy - 3 * h < y < sy]
        my = above[-1] if above else sy - 1.2 * h
        nxt = [y for y in max_ys if y > sy]
        # marks secured: numeric row(s) after the max row and before the next block;
        # on the old memo the values sit slightly above their label
        vals = numbers_between(my + 0.5 * h, min(nxt[0] - 0.5 * h, sy + 1.0 * h) if nxt else sy + 1.0 * h)
        max_row = numbers_between(my - 0.7 * h, my + 0.5 * h)
        _, kind = _year_label(page, slope, (my + sy) / 2, label_right)
        blocks.append((kind, vals, max_row))
    # Year blocks are numbered top to bottom (I, II); labels only identify PRACTICALS
    # (OCR misses/misreads the small 'Ist'/'2nd' labels).
    year_blocks = [b for b in blocks if b[0] != "practicals"]
    if not year_blocks:
        return rows
    ref = max((b[1] for b in year_blocks), key=len)          # the block with every column
    centers = [v.cx for v in ref]
    cols = _column_names(header_words, centers, slope, h)
    for kind, vals, max_row in blocks:
        paper = "II" if kind == "practicals" else ("I" if year_blocks.index((kind, vals, max_row)) == 0 else "II")
        for v in vals:
            ci = min(range(len(centers)), key=lambda k: abs(centers[k] - v.cx))
            if abs(centers[ci] - v.cx) > 3 * h or not cols[ci]:
                continue
            name_ws = cols[ci]
            mx = min(max_row, key=lambda m: abs(m.cx - v.cx), default=None)
            if mx is not None and abs(mx.cx - v.cx) > 3 * h:
                mx = None
            name = _join(name_ws, h)
            if kind == "practicals":
                name = f"{name} PRACTICAL"
            src = "transposed table (practicals block)" if kind == "practicals" else                 "transposed table (year blocks in top-to-bottom order)"
            subj = Cell(name_ws, "SUBJECT", min(w.conf for w in name_ws))
            rows.append(TableRow(subj, paper, src, Cell([v], "MARKS", v.conf),
                                 Cell([mx], "MAX_MARKS", mx.conf) if mx else None, name_override=name))
    return rows


PLAUSIBLE_MAXIMA = {20, 25, 30, 40, 50, 60, 70, 75, 80, 100}


def plausible(rows: list[TableRow]) -> bool:
    """Sanity gate: most rows must have a plausible maximum and marks <= maximum.
    A garbled parse (tiny/warped photo) fails and the table goes to review
    instead of being reported."""
    if not rows:
        return False
    bad = 0
    for r in rows:
        mk = to_int(r.marks.text)[0]
        mx = to_int(r.maximum.text)[0] if r.maximum else None
        if mk is None or mx is None or mx not in PLAUSIBLE_MAXIMA or mk > mx:
            bad += 1
    return bad / len(rows) <= 0.2


def _column_names(header_words: list[Word], centers: list[float], slope: float, h: float) -> list[list[Word]]:
    """Assign each header word to the column(s) it covers. Column cells are the
    intervals between midpoints of neighbouring value centres; a word goes to
    every column holding >= 30% of its width (so 'ENGLISH' spanning THEORY and
    PRACTICAL names both), otherwise to the nearest column only."""
    if not centers:
        return []
    bounds = [(-1e9 if i == 0 else (centers[i - 1] + c) / 2, 1e9 if i == len(centers) - 1 else (c + centers[i + 1]) / 2)
              for i, c in enumerate(centers)]
    out: list[list[Word]] = [[] for _ in centers]
    for w in header_words:
        width = max(1.0, w.bbox[2] - w.bbox[0])
        shares = [(max(0.0, min(w.bbox[2], hi) - max(w.bbox[0], lo)) / width, i) for i, (lo, hi) in enumerate(bounds)]
        big = [i for f, i in shares if f >= 0.3]
        if len(big) >= 2:
            for i in big:
                out[i].append(w)
        else:
            i = min(range(len(centers)), key=lambda k: abs(centers[k] - w.cx))
            if abs(centers[i] - w.cx) <= max(4 * h, (bounds[i][1] - bounds[i][0]) / 2 if bounds[i][1] < 1e8 else 4 * h):
                out[i].append(w)
    return [sorted(ws, key=lambda w: (round(_y(w, slope) / (0.8 * h)), w.bbox[0])) for ws in out]


def _join(words: list[Word], h: float) -> str:
    """Join header words; fragments of one word split by OCR ('CHEM' 'ISTRY',
    touching boxes on the same line) are joined without a space."""
    out = ""
    for k, w in enumerate(words):
        if k and w.line_id == words[k - 1].line_id and 0 <= w.bbox[0] - words[k - 1].bbox[2] < 0.25 * h                 and w.text[:1].isalpha() and words[k - 1].text[-1:].isalpha():
            out += w.text
        else:
            out += (" " if out else "") + w.text
    return out
