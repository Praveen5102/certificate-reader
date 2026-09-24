from src.annotation.align import year_boundary
from src.annotation.layout import PageWords, Word
from src.inference.schema import ExtractionResult, FieldValue, ReviewSummary, SubjectRow
from src.inference.table import Cell, reconstruct
from src.validation.checks import run_checks

YEAR_HEADERS = {"I": ["I Year"], "II": ["II Year"]}


def W(i, text, x0, y0, x1, y1, line=0):
    return Word(uid=f"1:{i}", page=1, idx=i, line_id=line, text=text, bbox=(x0, y0, x1, y1), conf=0.99)


def _page():
    words = [
        W(0, "I", 500, 50, 520, 70, 1), W(1, "Year", 525, 50, 580, 70, 1),
        W(2, "II", 800, 50, 825, 70, 2), W(3, "Year", 830, 50, 885, 70, 2),
        W(4, "ENGLISH", 100, 100, 250, 120, 3),
        W(5, "100", 450, 100, 490, 120, 4), W(6, "47*", 600, 100, 640, 120, 5),
        W(7, "100", 750, 100, 790, 120, 6), W(8, "71", 900, 100, 930, 120, 7),
        W(9, "PHYSICS", 100, 150, 240, 170, 8), W(10, "PRACTICALS", 245, 150, 400, 170, 8),
        W(11, "30", 750, 150, 780, 170, 9), W(12, "27", 900, 150, 930, 170, 10),
    ]
    return PageWords(page=1, width=1000, height=400, words=words)


def test_year_boundary_and_reconstruction():
    p = _page()
    b = year_boundary(p, YEAR_HEADERS)
    assert b[0] == "x" and 580 < b[1] < 800
    w = {x.idx: x for x in p.words}
    subj = [Cell([w[4]], "SUBJECT", 0.9), Cell([w[9], w[10]], "SUBJECT", 0.9)]
    mx = [Cell([w[5]], "MAX_MARKS", 0.9), Cell([w[7]], "MAX_MARKS", 0.9), Cell([w[11]], "MAX_MARKS", 0.9)]
    mk = [Cell([w[6]], "MARKS", 0.9), Cell([w[8]], "MARKS", 0.9), Cell([w[12]], "MARKS", 0.9)]
    rows = reconstruct(p, subj, mx, mk, b, ["PRACTICAL"])
    got = sorted((r.subject.text, r.paper, r.marks.text, r.maximum.text) for r in rows)
    # a single practicals value sitting in the II-year column must be paper II
    assert got == [("ENGLISH", "I", "47*", "100"), ("ENGLISH", "II", "71", "100"),
                   ("PHYSICS PRACTICALS", "II", "27", "30")]


def _fv(v):
    return FieldValue(value=v, confidence=0.99)


def test_checks_flag_but_do_not_change_values():
    fields = {"total_marks": _fv(576), "maximum_marks": _fv(500), "student_name": _fv("X")}
    subjects = [SubjectRow(subject_name=_fv("ENGLISH"), paper=_fv("I"), marks=_fv(120), maximum_marks=_fv(100))]
    issues = run_checks(fields, subjects, {"total_in_words": 575, "total_in_digits": 576})
    checks = {(i.field, i.check) for i in issues}
    assert ("total_marks", "total_le_maximum") in checks
    assert ("total_marks", "digits_vs_words") in checks
    assert ("subjects[0].marks", "marks_le_maximum") in checks
    assert fields["total_marks"].value == 576 and subjects[0].marks.value == 120


def test_flat_output_matches_spec_keys():
    r = ExtractionResult(file="c.pdf", document_type="intermediate_certificate",
                         review=ReviewSummary(status="auto"))
    keys = list(r.to_flat())
    spec = ["file", "document_type", "state", "board", "student_name", "father_name", "mother_name",
            "hall_ticket_number", "year_of_pass", "month_of_pass", "course", "group", "medium", "subjects",
            "total_marks", "maximum_marks", "percentage", "result"]
    assert keys[:len(spec)] == spec
    assert keys[-1] == "notes"
