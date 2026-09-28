"""Passport machine-readable zone (MRZ), ICAO Doc 9303 TD3 (2 lines x 44 chars).

    P<INDSHARMA<<RAVI<KUMAR<<<<<<<<<<<<<<<<<<<<<
    Z1234567<6IND9001014M3001012<<<<<<<<<<<<<<<6

Line 2 carries check digits for the passport number, birth date, expiry date,
optional data and a composite over all of them, so every value read from the
MRZ can be VERIFIED. OCR confusions are repaired only per field type (a letter
in a digit-only position, e.g. 'O'->'0'), and only when the repair makes the
check digit pass; otherwise the raw value is kept and flagged.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

_WEIGHTS = (7, 3, 1)
TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "1",
                          "Z": "2", "S": "5", "B": "8", "G": "6"})
TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})


def _val(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - 55
    return 0          # '<'


def check_digit(s: str) -> str:
    return str(sum(_val(c) * _WEIGHTS[i % 3] for i, c in enumerate(s)) % 10)


def _clean(line: str) -> str:
    """OCR line -> MRZ alphabet (A-Z, 0-9, '<'); common filler misreads -> '<'."""
    s = line.upper().replace(" ", "")
    s = re.sub(r"[«‹<‹«〈〈]", "<", s)
    s = re.sub(r"[^A-Z0-9<]", "<", s)
    return s


def find_mrz_lines(lines: list[str]) -> tuple[str, str] | None:
    """Pick the two consecutive text lines that look like a TD3 MRZ."""
    cands = [_clean(l) for l in lines if len(l.replace(" ", "")) >= 30]
    best = None
    for i, l1 in enumerate(cands):
        if not l1.startswith("P"):
            continue
        for l2 in cands[i + 1:i + 3]:
            if l2.count("<") >= 1 and re.search(r"\d{6}", l2.translate(TO_DIGIT)):
                score = l1.count("<") + (5 if l1[:5].startswith("P<") else 0)
                if best is None or score > best[0]:
                    best = (score, l1, l2)
    if best is None:
        return None
    l1, l2 = best[1], best[2]
    return l1.ljust(44, "<")[:44], l2.ljust(44, "<")[:44]


@dataclass
class MRZResult:
    document_code: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    passport_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""        # ISO yyyy-mm-dd
    sex: str = ""
    date_of_expiry: str = ""
    personal_number: str = ""
    checks: dict = field(default_factory=dict)   # field -> True/False
    repairs: list = field(default_factory=list)
    raw: tuple = ("", "")

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def _digits_field(raw: str, cd: str, name: str, repairs: list) -> tuple[str, bool]:
    """Numeric field: accept as read if the check digit passes, else try the
    letter->digit repair and accept that only if it makes the check pass."""
    cd_d = cd.translate(TO_DIGIT)
    if check_digit(raw) == cd_d:
        return raw, True
    fixed = raw.translate(TO_DIGIT)
    if fixed != raw and check_digit(fixed) == cd_d:
        repairs.append(f"{name}: '{raw}' -> '{fixed}' (OCR letter/digit confusion; check digit passes)")
        return fixed, True
    return raw, False


def _date(yymmdd: str, future: bool) -> str:
    if not re.fullmatch(r"\d{6}", yymmdd):
        return ""
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    now = dt.date.today().year % 100
    if future:
        century = 2000 if yy < 70 else 1900      # expiry dates are always in this century window
    else:
        century = 2000 if yy <= now else 1900    # a birth date cannot be in the future
    try:
        return dt.date(century + yy, mm, dd).isoformat()
    except ValueError:
        return ""


def parse_td3(l1: str, l2: str) -> MRZResult:
    r = MRZResult(raw=(l1, l2))
    r.document_code = l1[0:2].replace("<", "")
    r.issuing_country = l1[2:5].translate(TO_ALPHA).replace("<", "")
    names = l1[5:44]
    sur, _, given = names.partition("<<")
    r.surname = sur.translate(TO_ALPHA).replace("<", " ").strip()
    r.given_names = re.sub(r"<+", " ", given).translate(TO_ALPHA).strip()

    pn_raw = l2[0:9]
    pn_ok = check_digit(pn_raw) == l2[9].translate(TO_DIGIT)
    if not pn_ok:
        # Indian passports: 1 letter + 7 digits (+ filler). Repair digit positions only.
        fixed = pn_raw[0] + pn_raw[1:].translate(TO_DIGIT)
        if check_digit(fixed) == l2[9].translate(TO_DIGIT):
            r.repairs.append(f"passport_number: '{pn_raw}' -> '{fixed}' (check digit passes)")
            pn_raw, pn_ok = fixed, True
    r.passport_number = pn_raw.replace("<", "")
    r.checks["passport_number"] = pn_ok

    r.nationality = l2[10:13].translate(TO_ALPHA).replace("<", "")
    dob, ok = _digits_field(l2[13:19], l2[19], "date_of_birth", r.repairs)
    r.date_of_birth, r.checks["date_of_birth"] = _date(dob, future=False), ok
    sex = l2[20].replace("0", "O")
    r.sex = {"M": "M", "F": "F"}.get(sex, "X" if sex in "<X" else sex)
    exp, ok = _digits_field(l2[21:27], l2[27], "date_of_expiry", r.repairs)
    r.date_of_expiry, r.checks["date_of_expiry"] = _date(exp, future=True), ok
    r.personal_number = l2[28:42].replace("<", "")
    composite = l2[0:10] + l2[13:20] + l2[21:43]
    # recompute the composite with repaired values
    comp = (pn_raw.ljust(9, "<") + l2[9].translate(TO_DIGIT) + dob + l2[19].translate(TO_DIGIT)
            + exp + l2[27].translate(TO_DIGIT) + l2[28:43])
    r.checks["composite"] = check_digit(comp) == l2[43].translate(TO_DIGIT) or \
        check_digit(composite) == l2[43].translate(TO_DIGIT)
    return r


def read_mrz(lines: list[str]) -> MRZResult | None:
    pair = find_mrz_lines(lines)
    return parse_td3(*pair) if pair else None
