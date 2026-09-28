"""Passport label/value layout rules (synthetic OCR lines, no real data)."""
from src.passport.extract import _clean_value, _label_key, extract_passport


def _line(text, x0, y0, x1, y1, conf=0.99):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": conf}


def test_label_matching_handles_ocr_noise():
    assert _label_key("afe/ Date of Birth")[0] == "date_of_birth"
    assert _label_key("/Sumame")[0] == "surname"                 # 'rn' read as 'm'
    assert _label_key("/Place ofI")[0] == "place_of_issue"       # label cut short
    assert _label_key("HYDERABAD")[0] is None


def test_value_cleaning():
    assert _clean_value("nationality", "A/NDIAN") == "INDIAN"
    assert _clean_value("address", "MAIN R0AD, PIN.S00001") == "MAIN ROAD, PIN:500001"
    assert _clean_value("date_of_expiry", "1.6/09/2031") == "2031-09-16"


def test_empty_spouse_does_not_take_the_address():
    lines = [
        _line("/Name of Father / Legal Guardian", 100, 100, 500, 130),
        _line("RAVI KUMAR", 100, 135, 300, 165),
        _line("/Name of Mother", 100, 175, 300, 205),
        _line("LATHA KUMAR", 100, 210, 300, 240),
        _line("/Name of Spouse", 100, 250, 300, 280),
        _line("/Address", 100, 330, 220, 360),
        _line("1-2-3, MAIN ROAD", 100, 365, 400, 395),
        _line("HYDERABAD", 100, 420, 300, 450),
        _line("PIN:500001,TELANGANA,INDIA", 100, 475, 500, 505),
        _line("/Old Passport No. with Date and Place of Issue", 100, 530, 700, 560),
        _line("/File No.", 100, 600, 220, 630),
        _line("HY1234567890123", 100, 635, 400, 665),
    ]
    r = extract_passport([{"page": 1, "lines": lines}])
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert f["father_name"] == "RAVI KUMAR"
    assert f["mother_name"] == "LATHA KUMAR"
    assert f["spouse_name"] == ""
    assert f["address"] == "1-2-3, MAIN ROAD, HYDERABAD, PIN:500001,TELANGANA,INDIA"
    assert f["file_number"] == "HY1234567890123"
