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


def _line1_score(l: str) -> int:
    """How much a cleaned line looks like MRZ line 1 ('P<IND' + names + fillers)."""
    if not l.startswith("P") or "<<" not in l:
        return -1
    return l.count("<") + (10 if l.startswith("P<") else 0) + (5 if l[2:5].translate(TO_ALPHA).isalpha() else 0)


def _line2_score(l: str) -> int:
    """MRZ line 2: passport number, dates with check digits. Scored by how many
    of its check digits pass - the strongest evidence available."""
    l = l.ljust(44, "<")[:44]
    if not re.search(r"\d{6}", l.translate(TO_DIGIT)):
        return -1
    ok = 0
    ok += check_digit(l[0:9]) == l[9].translate(TO_DIGIT) or \
        check_digit(l[0] + l[1:9].translate(TO_DIGIT)) == l[9].translate(TO_DIGIT)
    ok += check_digit(l[13:19].translate(TO_DIGIT)) == l[19].translate(TO_DIGIT)
    ok += check_digit(l[21:27].translate(TO_DIGIT)) == l[27].translate(TO_DIGIT)
    return ok * 10 + l.count("<")


def find_mrz_lines(lines: list[str]) -> tuple[str, str] | None:
    """Pick the (line 1, line 2) pair that best looks like a TD3 MRZ. Lines may
    arrive in any order (rotated scans list line 2 first)."""
    cands = [_clean(l) for l in lines if len(l.replace(" ", "")) >= 30]
    ones = [(s, l) for l in cands if (s := _line1_score(l)) >= 0]
    twos = [(s, l) for l in cands if not l.startswith("P<") and (s := _line2_score(l)) >= 0]
    if not ones or not twos:
        return None
    l1 = max(ones)[1]
    l2 = max(twos)[1]
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
    personal_number: str = ""                     # Indian passports: the file number digits
    checks_optional: bool = False
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
        # Indian passports: 1 letter + 7 digits (older) or 2 letters + 6 digits (2025+).
        # Repair digit positions only, and only if the check digit then passes.
        for n_letters in (1, 2):
            fixed = pn_raw[:n_letters].translate(TO_ALPHA) + pn_raw[n_letters:].translate(TO_DIGIT)
            if check_digit(fixed) == l2[9].translate(TO_DIGIT):
                r.repairs.append(f"passport_number: '{pn_raw}' -> '{fixed}' (check digit passes)")
                pn_raw, pn_ok = fixed, True
                break
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
    r.checks_optional = check_digit(l2[28:42]) == l2[42].translate(TO_DIGIT) or l2[42] == "<"
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
