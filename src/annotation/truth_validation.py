"""Structural + plausibility checks on the canonical truth file.

Every check here only *flags*. Nothing in this module (or anywhere else in the
project) writes to the truth file. Issues are emitted as dicts:

    {"file", "field", "check", "severity", "truth_value", "detail"}

severity:
    error    -> value is internally inconsistent; do not train/evaluate on it
                without human review
    warning  -> value is suspicious (probable OCR residue, non-standard form)
    info     -> value is missing / derived; affects coverage, not correctness
"""
from __future__ import annotations

import re
from typing import Any

EXPECTED_KEYS = [
    "file", "document_type", "state", "board", "student_name", "father_name", "mother_name",
    "hall_ticket_number", "year_of_pass", "month_of_pass", "course", "group", "medium",
    "subjects", "total_marks", "maximum_marks", "percentage", "result", "cgpa",
]
SUBJECT_KEYS = ["subject_name", "paper", "marks", "maximum_marks"]
NAME_FIELDS = ["student_name", "father_name", "mother_name"]
MONTHS = {"JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
          "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"}
# Result vocabulary observed on TS/AP Intermediate certificates.
RESULT_VOCAB = {"PASSED", "QUALIFIED", "COMPARTMENTAL", "COMPARTMENTALLY", "A GRADE", "B GRADE",
                "C GRADE", "D GRADE", "FIRST DIVISION", "SECOND DIVISION", "THIRD DIVISION", "FAILED"}
MEDIUM_VOCAB = {"ENGLISH", "TELUGU", "URDU", "HINDI", "MARATHI", "KANNADA", "TAMIL", "ORIYA", "ODIA"}
# Phrases that belong to the certificate template, not to a person's name.
TEMPLATE_RESIDUE = re.compile(
    r"\b(THIS|THISIS|CERTIFY|CERTY|CER|CEIY|CEVILY|THAT|THT|TAT|HA|NAME|FATHER|MOTHER)\b", re.I)
PAPER_VOCAB = {"I", "II"}
VALID_YEAR = (1980, 2030)


def _issue(rec: dict, field: str, check: str, severity: str, detail: str, value: Any = None) -> dict:
    return {"file": rec.get("file"), "field": field, "check": check, "severity": severity,
            "truth_value": rec.get(field) if value is None and field in rec else value, "detail": detail}


def is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v)


def name_suspicion(value: str) -> list[str]:
    """Return reasons a person-name truth value looks like OCR residue."""
    reasons = []
    if not value:
        return reasons
    if TEMPLATE_RESIDUE.search(value):
        reasons.append("contains certificate template words (e.g. 'THIS IS TO CERTIFY THAT')")
    if re.search(r"[a-z]", value) and re.search(r"[A-Z]{3,}", value):
        reasons.append("mixed case inside an otherwise upper-case name")
    if re.search(r"[0-9()\[\]{}|/\\:;,]", value):
        reasons.append("contains digits or punctuation")
    if re.search(r"(^|\s)\.(\s|$)|\.\s*\.", value):
        reasons.append("contains stray dots")
    tokens = value.split()
    if tokens and len(tokens[-1]) <= 2 and len(tokens) > 1 and tokens[-1] not in {"MD", "SK"}:
        reasons.append(f"ends with a very short token '{tokens[-1]}' (possible OCR tail noise)")
    if len(value.replace(" ", "")) < 4:
        reasons.append("implausibly short")
    return reasons


def validate_record(rec: dict) -> list[dict]:
    issues: list[dict] = []
    keys = list(rec.keys())
    if keys != EXPECTED_KEYS:
        missing = [k for k in EXPECTED_KEYS if k not in rec]
        extra = [k for k in rec if k not in EXPECTED_KEYS]
        if missing or extra:
            issues.append(_issue(rec, "*", "schema_keys", "error",
                                 f"missing={missing} extra={extra}", value=None))

    # ---- missing values (coverage) -------------------------------------------------
    for f in ["state", "board", "student_name", "father_name", "mother_name", "hall_ticket_number",
              "year_of_pass", "month_of_pass", "medium", "total_marks", "result", "subjects"]:
        if is_blank(rec.get(f)):
            issues.append(_issue(rec, f, "missing_value", "info",
                                 "blank in truth: treated as UNANNOTATED (not as 'absent from document')"))

    # ---- names ----------------------------------------------------------------------
    for f in NAME_FIELDS:
        v = rec.get(f) or ""
        for reason in name_suspicion(v):
            issues.append(_issue(rec, f, "name_ocr_residue", "warning", reason))
    sn, fn = rec.get("student_name"), rec.get("father_name")
    if sn and fn and sn.strip() == fn.strip():
        issues.append(_issue(rec, "father_name", "student_equals_father", "warning",
                             "student_name and father_name are identical"))

    # ---- dates ----------------------------------------------------------------------
    y = rec.get("year_of_pass")
    if y is not None and (not isinstance(y, int) or not VALID_YEAR[0] <= y <= VALID_YEAR[1]):
        issues.append(_issue(rec, "year_of_pass", "implausible_year", "error", f"expected int in {VALID_YEAR}"))
    m = rec.get("month_of_pass") or ""
    if m and m.upper() not in MONTHS:
        issues.append(_issue(rec, "month_of_pass", "nonstandard_month", "warning", "not a month name"))
    if m and y is None:
        issues.append(_issue(rec, "year_of_pass", "month_without_year", "info", "month present, year missing"))

    # ---- categorical ----------------------------------------------------------------
    r = (rec.get("result") or "").strip()
    if r and r.upper() not in RESULT_VOCAB:
        norm = r.rstrip(". ").upper()
        detail = ("trailing punctuation only; canonical form would be " + repr(norm)) if norm in RESULT_VOCAB \
            else "not in known result vocabulary; probable OCR residue"
        issues.append(_issue(rec, "result", "nonstandard_result",
                             "info" if norm in RESULT_VOCAB else "warning", detail))
    med = (rec.get("medium") or "").strip()
    if med and med.upper() not in MEDIUM_VOCAB:
        issues.append(_issue(rec, "medium", "nonstandard_medium", "warning", "not a known medium"))
    ht = (rec.get("hall_ticket_number") or "").strip()
    if ht and not re.fullmatch(r"[0-9A-Z]{6,14}", ht):
        issues.append(_issue(rec, "hall_ticket_number", "hall_ticket_format", "warning",
                             "unexpected characters/length"))

    # ---- subjects -------------------------------------------------------------------
    subjects = rec.get("subjects") or []
    for i, s in enumerate(subjects):
        if list(s.keys()) != SUBJECT_KEYS:
            issues.append(_issue(rec, f"subjects[{i}]", "subject_schema", "error", f"keys={list(s.keys())}", s))
            continue
        if s["paper"] not in PAPER_VOCAB:
            issues.append(_issue(rec, f"subjects[{i}].paper", "nonstandard_paper", "warning",
                                 f"paper={s['paper']!r}", s))
        if not isinstance(s["marks"], (int, float)):
            issues.append(_issue(rec, f"subjects[{i}]", "non_numeric_marks", "error", "marks not numeric", s))
        elif s["maximum_marks"] is None:
            # e.g. AP Open School memos print no per-subject maximum
            issues.append(_issue(rec, f"subjects[{i}].maximum_marks", "maximum_not_printed", "info",
                                 "per-subject maximum not printed on this certificate", s))
        elif not isinstance(s["maximum_marks"], (int, float)):
            issues.append(_issue(rec, f"subjects[{i}]", "non_numeric_marks", "error", "maximum not numeric", s))
        elif s["marks"] > s["maximum_marks"]:
            issues.append(_issue(rec, f"subjects[{i}].marks", "marks_exceed_max", "error",
                                 f"{s['marks']} > {s['maximum_marks']}", s))
    keys_seen = [(s.get("subject_name"), s.get("paper")) for s in subjects]
    dups = {k for k in keys_seen if keys_seen.count(k) > 1}
    for k in dups:
        issues.append(_issue(rec, "subjects", "duplicate_subject_row", "warning", f"repeated row {k}", None))

    # ---- totals ---------------------------------------------------------------------
    total, gmax, pct = rec.get("total_marks"), rec.get("maximum_marks"), rec.get("percentage")
    numeric_subjects = [s for s in subjects if isinstance(s.get("marks"), (int, float))
                        and isinstance(s.get("maximum_marks"), (int, float))]
    subj_sum = sum(s["marks"] for s in numeric_subjects)
    subj_max = sum(s["maximum_marks"] for s in numeric_subjects)
    if total is not None and numeric_subjects and subj_sum != total:
        issues.append(_issue(rec, "total_marks", "subject_sum_mismatch", "error",
                             f"sum(subject marks)={subj_sum} but total_marks={total}"))
    if gmax is not None and total is not None and gmax < total:
        issues.append(_issue(rec, "maximum_marks", "maximum_below_total", "error",
                             f"maximum_marks={gmax} < total_marks={total}; field appears to hold a per-subject "
                             f"maximum, not the grand maximum (sum of subject maxima={subj_max or 'n/a'})"))
    if pct is not None:
        if total is not None and subj_max:
            expected = round(100.0 * total / subj_max, 1)
            if abs(expected - pct) > 0.15:
                issues.append(_issue(rec, "percentage", "percentage_inconsistent", "error",
                                     f"100*total/sum(subject max) = {expected}, truth={pct}"))
            else:
                issues.append(_issue(rec, "percentage", "derived_value", "info",
                                     "equals 100*total/sum(subject maxima); not printed on TS/AP certificates"))
        elif total is None:
            issues.append(_issue(rec, "percentage", "percentage_without_total", "warning", "no total to verify"))
    if rec.get("cgpa") is not None and not (0 < rec["cgpa"] <= 10):
        issues.append(_issue(rec, "cgpa", "implausible_cgpa", "error", "expected 0-10"))
    return issues


def validate_all(records: list[dict]) -> list[dict]:
    out = []
    files = [r.get("file") for r in records]
    for f in {f for f in files if files.count(f) > 1}:
        out.append({"file": f, "field": "file", "check": "duplicate_file_record", "severity": "error",
                    "truth_value": f, "detail": "file appears more than once in truth"})
    for r in records:
        out.extend(validate_record(r))
    return out
