"""Post-extraction validation. Checks *flag*; they never change a value."""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any

from ..inference.schema import FieldValue, SubjectRow, ValidationIssue

MAX_YEAR = _dt.date.today().year + 1


def _v(fields: dict[str, FieldValue], name: str) -> Any:
    fv = fields.get(name)
    return fv.value if fv is not None else None


def run_checks(fields: dict[str, FieldValue], subjects: list[SubjectRow],
               extra: dict[str, Any] | None = None) -> list[ValidationIssue]:
    extra = extra or {}
    out: list[ValidationIssue] = []

    def add(field, check, status, reason):
        out.append(ValidationIssue(field=field, check=check, status=status, reason=reason))

    total, gmax = _v(fields, "total_marks"), _v(fields, "maximum_marks")
    if total is not None and not isinstance(total, int):
        add("total_marks", "numeric", "error", f"total_marks is not an integer: {total!r}")
    if isinstance(total, int) and isinstance(gmax, int) and total > gmax:
        add("total_marks", "total_le_maximum", "error", f"total {total} exceeds maximum {gmax}")

    rows = [(s.subject_name.value, s.paper.value, s.marks.value, s.maximum_marks.value) for s in subjects]
    for i, (name, paper, marks, mx) in enumerate(rows):
        if marks is None:
            add(f"subjects[{i}].marks", "numeric", "error", f"marks for {name!r} not an integer")
        if mx is None:
            add(f"subjects[{i}].maximum_marks", "missing", "warning", f"maximum marks for {name!r} not found")
        if isinstance(marks, int) and isinstance(mx, int) and marks > mx:
            add(f"subjects[{i}].marks", "marks_le_maximum", "error", f"{name!r} marks {marks} > maximum {mx}")
    seen = {}
    for i, (name, paper, *_rest) in enumerate(rows):
        key = (re.sub(r"[^A-Z]", "", str(name).upper()), paper)
        if key in seen:
            add(f"subjects[{i}]", "duplicate_row", "warning", f"row {name!r} paper {paper} appears twice")
        seen[key] = i
    if subjects and isinstance(total, int) and all(isinstance(r[2], int) for r in rows):
        s = sum(r[2] for r in rows)
        if s != total:
            add("total_marks", "subject_sum", "warning",
                f"sum of extracted subject marks ({s}) != declared total ({total})")
    words_total = extra.get("total_in_words")
    digits_total = extra.get("total_in_digits")
    if words_total is not None and digits_total is not None and words_total != digits_total:
        add("total_marks", "digits_vs_words", "error",
            f"total in figures ({digits_total}) != total in words ({words_total})")

    y = _v(fields, "year_of_pass")
    if y is not None and not (1980 <= y <= MAX_YEAR):
        add("year_of_pass", "plausible_year", "error", f"year {y} outside 1980-{MAX_YEAR}")
    ht = _v(fields, "hall_ticket_number")
    if ht and not re.fullmatch(r"\d{10}", str(ht)):
        add("hall_ticket_number", "format", "warning",
            f"{ht!r} is not 10 digits (TS/AP registration numbers observed in this dataset are 10 digits)")
    if ht and y and re.fullmatch(r"\d{10}", str(ht)) and int(str(ht)[:2]) != y % 100:
        # TS/AP registration numbers start with the 2-digit exam year in this
        # dataset; a mismatch is common for supplementary exams, so warn only.
        add("hall_ticket_number", "year_prefix", "info",
            f"registration number prefix {str(ht)[:2]} differs from year {y}")
    cg = _v(fields, "cgpa")
    if cg is not None and not (0 < cg <= 10):
        add("cgpa", "range", "error", f"CGPA {cg} outside (0, 10]")
    for req in ("student_name", "total_marks"):
        if _v(fields, req) in (None, ""):
            add(req, "required", "warning", f"{req} not extracted")
    return out
