"""Aadhaar / PAN rules on synthetic OCR lines (made-up people and numbers, no real data)."""
from src.idcards.aadhaar import _D, _P, extract_aadhaar, mask, verhoeff_ok
from src.idcards.pan import extract_pan


def _with_check_digit(first11: str) -> str:
    inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    c = 0
    for i, d in enumerate(reversed(first11 + "0")):
        c = _D[c][_P[i % 8][int(d)]]
    return first11 + str(inv[c])


def _line(text, x0, y0, x1, y1, conf=0.99):
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": conf}


def test_verhoeff_and_mask():
    n = _with_check_digit("23456789012")
    assert verhoeff_ok(n)
    bad = n[:-1] + str((int(n[-1]) + 1) % 10)
    assert not verhoeff_ok(bad)
    assert mask(n) == f"XXXX XXXX {n[-4:]}"


def test_aadhaar_card_front_and_back():
    n = _with_check_digit("56781234567")
    spaced = f"{n[:4]} {n[4:8]} {n[8:]}"
    front = [_line("Government of India", 300, 50, 900, 90),
             _line("Ravi Kumar Test", 400, 200, 700, 240),
             _line("/DOB: 01/02/2003", 400, 245, 700, 285),
             _line("/ MALE", 400, 290, 560, 330),
             _line(spaced, 400, 500, 800, 550),
             _line("VID : 9123 4567 8901 2345", 400, 560, 900, 600)]
    back = [_line("Address:", 80, 100, 250, 140),
            _line("S/O Suresh Test, 1-2-3,", 80, 145, 600, 185),
            _line("Main Road, Hyderabad,", 80, 190, 600, 230),
            _line("Telangana - 500001", 80, 235, 500, 275),
            _line(spaced, 300, 320, 700, 370)]
    r = extract_aadhaar([{"page": 1, "lines": front}, {"page": 2, "lines": back}])
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert f["aadhaar_number"] == f"XXXX XXXX {n[-4:]}"          # masked by default
    assert r["fields"]["aadhaar_number"]["verified"]
    assert (f["name"], f["date_of_birth"], f["gender"]) == ("Ravi Kumar Test", "2003-02-01", "MALE")
    assert (f["relation"], f["care_of"], f["pincode"]) == ("S/O", "Suresh Test", "500001")
    assert f["address"] == "1-2-3, Main Road, Hyderabad, Telangana - 500001"
    assert r["review"]["status"] == "auto"


def test_aadhaar_bad_check_digit_is_flagged():
    n = _with_check_digit("56781234567")
    bad = n[:-1] + str((int(n[-1]) + 1) % 10)
    lines = [_line("Ravi Kumar Test", 400, 200, 700, 240), _line("DOB: 01/02/2003", 400, 245, 700, 285),
             _line("MALE", 400, 290, 560, 330), _line(f"{bad[:4]} {bad[4:8]} {bad[8:]}", 400, 500, 800, 550)]
    r = extract_aadhaar([{"page": 1, "lines": lines}])
    assert not r["fields"]["aadhaar_number"]["verified"]
    assert "aadhaar_number" in r["review"]["reasons"]
    assert r["review"]["status"] == "manual_required"


def test_pan_labels_repair_and_letter_check():
    lines = [_line("INCOME TAX DEPARTMENT", 100, 50, 700, 100),
             _line("Permanent Account Number Card", 400, 200, 1000, 250),
             _line("ABCPT1Z34K", 500, 260, 900, 320),               # OCR read 2 as Z
             _line("/Name", 100, 400, 300, 440),
             _line("RAVI KUMAR TEST", 100, 445, 600, 490),
             _line("/Father's Name", 100, 520, 500, 560),
             _line("SURESH TEST", 100, 565, 500, 610),
             _line("Date of Birth", 100, 640, 330, 680),
             _line("01/02/2003", 100, 685, 330, 730)]
    r = extract_pan([{"page": 1, "lines": lines}])
    f = {k: v["value"] for k, v in r["fields"].items()}
    assert f == {"pan_number": "ABCPT1234K", "name": "RAVI KUMAR TEST", "father_name": "SURESH TEST",
                 "date_of_birth": "2003-02-01"}
    assert r["checks"]["holder_type"] == "individual"
    assert r["checks"]["surname_letter_matches"]
    assert "pan_repair" in r["checks"]
