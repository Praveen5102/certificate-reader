"""10th (SSC) rules on synthetic OCR lines (made-up student, no real data)."""
from src.idcards.ssc import extract_ssc


def _line(text, x0, y0, x1, y1, conf=0.99):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": conf}


HEADER = [
    _line("Board of Secondary Education", 300, 20, 900, 60),
    _line("TELANGANA STATE, INDIA", 420, 65, 760, 95),
    _line("SECONDARY SCHOOL CERTIFICATE", 350, 200, 850, 235),
    _line("REGULAR PC/22/00000/0000000/1", 400, 240, 800, 270),
    _line("CERTIFIED THAT", 100, 300, 260, 325), _line(": RAVI KUMAR TEST", 300, 300, 520, 325),
    _line("FATHER'S NAME", 100, 340, 260, 365), _line(": SURESH TEST", 300, 340, 480, 365),
    _line("MOTHER'S NAME", 100, 380, 260, 405), _line(": LATHA TEST", 300, 380, 470, 405),
    _line("ROLL NO.", 100, 420, 200, 445), _line(":2000000001", 300, 420, 440, 445),
    _line("DATE OF BIRTH", 100, 460, 260, 485), _line(":01/02/2005", 300, 460, 430, 485),
    _line("SCHOOL", 100, 500, 190, 525), _line(": TEST HIGH SCHOOL, HYDERABAD DISTRICT", 300, 500, 700, 525),
    _line("MEDIUM", 100, 540, 190, 565), _line(":ENGLISH", 300, 540, 400, 565),
    _line("Has Declared Passed SSC EXAMINATION held in MARCH-2021", 100, 580, 700, 605),
]


def _grade_rows(points):
    rows, y = [], 700
    areas = [("FIRST LANGUAGE : (", "TELUGU"), ("THIRD LANGUAGE", "ENGLISH"), ("MATHEMATICS", None),
             ("SCIENCE", None), ("SOCIAL STUDIES", None), ("SECOND LANGUAGE : (", "HINDI")]
    for (area, lang), (grade, gp) in zip(areas, points):
        rows.append(_line(area, 100, y, 290, y + 25))
        if lang:
            rows.append(_line(lang, 350, y, 450, y + 25))
        rows += [_line(grade, 580, y, 615, y + 25), _line(f"{gp:02d}", 700, y, 730, y + 25)]
        y += 40
    return rows, y


def test_grades_layout_with_cgpa_check():
    rows, y = _grade_rows([("A1", 10), ("A2", 9), ("A1", 10), ("B1", 8), ("A1", 10), ("A2", 9)])
    lines = HEADER + rows + [_line("Cumulative Grade Point Average (CGPA) :", 100, y + 20, 520, y + 50),
                             _line("9.3", 880, y + 20, 940, y + 52)]
    r = extract_ssc([{"page": 1, "lines": lines}])
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert (f["student_name"], f["father_name"], f["mother_name"]) == ("RAVI KUMAR TEST", "SURESH TEST", "LATHA TEST")
    assert (f["roll_number"], f["date_of_birth"], f["medium"], f["exam"]) == ("2000000001", "2005-02-01", "ENGLISH", "MARCH 2021")
    assert f["state"] == "TELANGANA" and f["result_type"] == "grades" and f["cgpa"] == 9.3
    assert [(s["subject"], s["grade"], s["grade_point"]) for s in r["subjects"]][0] == ("TELUGU", "A1", 10)
    assert r["checks"]["cgpa_matches_grade_points"] and r["fields"]["cgpa"]["verified"]
    assert r["review"]["status"] == "auto"


def test_grade_point_mismatch_is_flagged():
    rows, y = _grade_rows([("A1", 10), ("A2", 9), ("A1", 7), ("B1", 8), ("A1", 10), ("A2", 9)])   # A1 must be 10
    lines = HEADER + rows + [_line("Cumulative Grade Point Average (CGPA) :", 100, y + 20, 520, y + 50),
                             _line("8.8", 880, y + 20, 940, y + 52)]
    r = extract_ssc([{"page": 1, "lines": lines}])
    assert any("should carry 10 points" in v for v in r["review"]["reasons"].values())
    assert r["review"]["status"] != "auto"


def test_marks_layout_total_check():
    rows, y = [], 700
    for area, marks in [("FIRST LANGUAGE : ( TELUGU", 95), ("SECOND LANGUAGE : ( HINDI", 90), ("THIRD LANGUAGE : ENGLISH", 88),
                        ("MATHEMATICS", 99), ("GENERAL SCIENCE", 91), ("SOCIAL STUDIES", 93)]:
        rows += [_line(area, 100, y, 400, y + 25), _line(str(marks), 600, y, 640, y + 25)]
        y += 40
    lines = HEADER + rows + [_line("GRAND TOTAL", 100, y, 260, y + 25), _line("556", 600, y, 650, y + 25)]
    r = extract_ssc([{"page": 1, "lines": lines}])
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert f["result_type"] == "marks" and f["total_marks"] == 556 and f["maximum_marks"] == 600
    assert r["checks"]["total_matches_subjects"]


def test_intermediate_certificate_is_rejected():
    lines = [_line("Telangana State Board Of Intermediate Education", 100, 20, 900, 60),
             _line("PASS CERTIFICATE CUM MEMORANDUM OF MARKS", 200, 100, 800, 130)]
    r = extract_ssc([{"page": 1, "lines": lines}])
    assert r["document_type"] == "not_ssc" and "Intermediate" in r["notes"]
