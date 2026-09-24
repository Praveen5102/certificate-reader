"""Subject table reconstruction: tagged cells -> (subject, paper, marks, maximum) rows.

The tagger (rules or model) only says *what* a cell is (SUBJECT / MAX_MARKS /
MARKS). Structure comes from geometry, never from reading order alone:
  * a subject's cells are the MAX/MARKS words on its visual row to the right
    (row-oriented tables), or below it in its column (column-oriented tables,
    e.g. the TS online memo);
  * each MARKS cell is paired with the nearest MAX cell before it;
  * the paper (I / II) comes from the detected year headers ("I Year" / "II Year",
    "FIRST YEAR" / "SECOND YEAR") when present, else from left-to-right order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..annotation.layout import PageWords, Word, h_overlap, union_bbox, v_overlap
from ..validation.normalize import to_int


@dataclass
class Cell:
    words: list[Word]
    label: str
    confidence: float

    @property
    def bbox(self):
        return union_bbox(self.words)

    @property
    def text(self):
        return " ".join(w.text for w in self.words)


@dataclass
class TableRow:
    subject: Cell
    paper: str
    paper_source: str
    marks: Cell
    maximum: Cell | None
    notes: list[str] = field(default_factory=list)
    name_override: str | None = None   # when the name is not a plain join of words (transposed tables)


def _cx(c: Cell) -> float:
    b = c.bbox
    return (b[0] + b[2]) / 2


def _cy(c: Cell) -> float:
    b = c.bbox
    return (b[1] + b[3]) / 2


def estimate_skew(page: PageWords) -> float:
    """Page slope dy/dx from OCR text lines (each line's words lie on one
    baseline). Median over lines with >= 3 words; 0 when undeterminable."""
    by_line: dict[int, list[Word]] = {}
    for w in page.words:
        by_line.setdefault(w.line_id, []).append(w)
    slopes = []
    for ws in by_line.values():
        if len(ws) < 3:
            continue
        xs = [w.cx for w in ws]
        ys = [w.cy for w in ws]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var <= 0 or (max(xs) - min(xs)) < page.width * 0.15:
            continue
        slopes.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var)
    if len(slopes) < 3:
        return 0.0
    slopes.sort()
    s = slopes[len(slopes) // 2]
    return max(-0.1, min(0.1, s))


def clean_subject_cells(page: PageWords, cells: list[Cell], slope: float) -> list[Cell]:
    """Drop punctuation-only words (':' '-' OCR'd from 'Part-II :'), drop cells
    without letters, and merge name fragments on the same row ('MATHEMATICS' + 'A')."""
    out = []
    for c in cells:
        ws = [w for w in c.words if any(ch.isalnum() for ch in w.text)]
        if ws and any(ch.isalpha() for w in ws for ch in w.text):
            out.append(Cell(ws, c.label, c.confidence))
    out.sort(key=lambda c: (_ydesk(c, slope), c.bbox[0]))
    merged: list[Cell] = []
    for c in out:
        prev = merged[-1] if merged else None
        if prev is not None and _same_row(prev, c, slope) and 0 <= c.bbox[0] - prev.bbox[2] < 2.0 * page.text_h:
            merged[-1] = Cell(prev.words + c.words, prev.label, min(prev.confidence, c.confidence))
        else:
            merged.append(c)
    return merged


def _ydesk(c: Cell, slope: float) -> float:
    return _cy(c) - slope * _cx(c)


def _height(c: Cell) -> float:
    b = c.bbox
    return max(1.0, b[3] - b[1])


def _same_row(a: Cell, b: Cell, slope: float, tol: float = 0.6) -> bool:
    return abs(_ydesk(a, slope) - _ydesk(b, slope)) <= tol * max(_height(a), _height(b))


def reconstruct(page: PageWords, subjects: list[Cell], maxes: list[Cell], marks: list[Cell],
                boundary: tuple[str, float] | None, practical_markers: list[str]) -> list[TableRow]:
    """Row-oriented tables: every MARKS cell is assigned to the nearest SUBJECT
    to its left on the same (deskewed) row - a global assignment, so one subject
    cannot absorb a neighbouring row's marks. Marks with no row subject fall
    back to column-oriented matching (online memos: subjects are column headers)."""
    slope = estimate_skew(page)
    subjects = clean_subject_cells(page, subjects, slope)
    rows: list[TableRow] = []
    used_marks: set[int] = set()
    used_max: set[int] = set()
    by_subject: dict[int, list[tuple[int, Cell]]] = {}
    for i, m in enumerate(marks):
        cands = [(abs(_ydesk(s, slope) - _ydesk(m, slope)), k) for k, s in enumerate(subjects)
                 if m.bbox[0] > s.bbox[2] and _same_row(s, m, slope)]
        if cands:
            by_subject.setdefault(min(cands)[1], []).append((i, m))
    for k, subj in sorted(enumerate(subjects), key=lambda ks: (_ydesk(ks[1], slope), _cx(ks[1]))):
        sb = subj.bbox
        row_marks = [(i, m) for i, m in by_subject.get(k, []) if i not in used_marks]
        orientation = "row"
        if not row_marks and not by_subject:
            # column-oriented table: values below the subject header, same column
            row_marks = [(i, m) for i, m in enumerate(marks) if i not in used_marks and _cy(m) > sb[3]
                         and h_overlap(sb, m.bbox) >= 0.3]
            orientation = "column"
        row_marks.sort(key=lambda im: (_cx(im[1]) if orientation == "row" else _cy(im[1])))
        for order, (i, mk) in enumerate(row_marks):
            # nearest unused MAX before this MARKS cell on the same row / column
            if orientation == "row":
                cands = [(j, mx) for j, mx in enumerate(maxes) if j not in used_max and _same_row(mk, mx, slope)
                         and mx.bbox[2] <= mk.bbox[0] + page.text_h and mx.bbox[0] > sb[2] - page.text_h]
                cands.sort(key=lambda jm: mk.bbox[0] - jm[1].bbox[2])
            else:
                cands = [(j, mx) for j, mx in enumerate(maxes) if j not in used_max and h_overlap(mk.bbox, mx.bbox) >= 0.3
                         and mx.bbox[3] <= mk.bbox[1] + page.text_h and _cy(mx) > sb[3]]
                cands.sort(key=lambda jm: mk.bbox[1] - jm[1].bbox[3])
            mx = None
            if cands:
                j, mx = cands[0]
                used_max.add(j)
            used_marks.add(i)
            paper, src = _paper(mk, boundary, order, len(row_marks))
            name_cell = subj
            notes = []
            if orientation == "column" and _is_practical_row(page, mk, practical_markers):
                paper, src = "II", "practicals row"
                notes.append("practicals row in column-oriented table")
            rows.append(TableRow(name_cell, paper, src, mk, mx, notes))
    return rows


def _paper(mk: Cell, boundary, order: int, n: int) -> tuple[str, str]:
    if boundary is not None:
        axis, b = boundary
        coord = _cx(mk) if axis == "x" else _cy(mk)
        return ("I" if coord < b else "II"), f"year header boundary ({axis})"
    if n == 1:
        return "II", "single value; no year headers found (assumed II) - review"
    return ("I" if order == 0 else "II"), "reading order; no year headers found"


def _is_practical_row(page: PageWords, mk: Cell, markers: list[str]) -> bool:
    mb = mk.bbox
    for w in page.words:
        if w.bbox[2] < mb[0] and v_overlap(w.bbox, mb) >= 0.3 and any(m in w.text.upper() for m in markers):
            return True
    return False


def row_values(r: TableRow) -> dict:
    marks, _ = to_int(r.marks.text)
    mx = to_int(r.maximum.text)[0] if r.maximum else None
    return {"subject_name": r.subject.text, "paper": r.paper, "marks": marks, "maximum_marks": mx}
