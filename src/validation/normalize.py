"""Safe, conservative normalization.

Rules:
  * numbers: strip leading zeros / '%' / trailing markers ('*', 'P') only when the
    remaining string is unambiguously numeric;
  * names: whitespace collapse only. Case is preserved (certificates print names
    in upper case and the truth keeps that form);
  * categorical values: mapped to a canonical vocabulary only on an exact or
    near-exact match; otherwise returned unchanged and flagged by the caller.
Every normalizer returns (value, note) where note explains any change, so the
change is traceable in the output.
"""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz, process

MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
          "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
MONTH_ABBR = {m[:3]: m for m in MONTHS} | {"SEPT": "SEPTEMBER"}
RESULTS = ["PASSED", "QUALIFIED", "COMPARTMENTAL", "COMPARTMENTALLY", "A GRADE", "B GRADE", "C GRADE",
           "D GRADE", "FIRST DIVISION", "SECOND DIVISION", "THIRD DIVISION", "FAILED"]
MEDIA = ["ENGLISH", "TELUGU", "URDU", "HINDI", "MARATHI", "KANNADA", "TAMIL", "ORIYA"]


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def to_int(raw: Any) -> tuple[int | None, str]:
    """'095' -> 95, '47*' -> 47 (marker '*' = marks obtained at earlier exam), '76 P' -> 76."""
    if raw is None:
        return None, "empty"
    if isinstance(raw, bool):
        return None, "boolean is not a mark"
    if isinstance(raw, int):
        return raw, ""
    if isinstance(raw, float):
        return (int(raw), "") if raw.is_integer() else (None, "non-integer")
    s = collapse_ws(str(raw))
    # A single trailing non-alphanumeric marker ('*' = marks from an earlier
    # attempt; OCR renders it as '*', '°', '•', '�', ...) or a pass flag 'P'.
    m = re.fullmatch(r"(0*)(\d{1,4})\s*([^\w\s]?\s*P|[^\w\s])?", s, flags=re.I)
    if not m or (m.group(1) and not m.group(2)):
        return None, f"not an unambiguous integer: {s!r}"
    note = []
    if m.group(1) and m.group(2):
        note.append("leading zeros stripped")
    if m.group(3):
        note.append(f"marker {m.group(3).strip()!r} stripped")
    return int(m.group(2)), "; ".join(note)


def to_float(raw: Any) -> tuple[float | None, str]:
    """'74.8%' -> 74.8, '7.94' -> 7.94."""
    if raw is None:
        return None, "empty"
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), ""
    s = collapse_ws(str(raw)).rstrip("%").strip()
    if re.fullmatch(r"\d{1,3}(\.\d{1,3})?", s):
        return float(s), ("'%' stripped" if "%" in str(raw) else "")
    return None, f"not an unambiguous number: {raw!r}"


def normalize_name(raw: str) -> tuple[str, str]:
    """Whitespace collapse only; leading/trailing punctuation that OCR attaches
    to a field separator (':' or '.') is removed and noted."""
    s = collapse_ws(raw)
    stripped = s.strip(" :;.,-_|")
    return stripped, ("separator punctuation stripped" if stripped != s else "")


def normalize_month_year(raw: str) -> tuple[str, int | None, str]:
    """'MARCH-2019' / 'JUNE 2024' / 'MARCH - 2019 (REG)' / 'MAY -2022' -> ('MARCH', 2019)."""
    s = collapse_ws(raw).upper()
    year = None
    ym = re.search(r"(19[89]\d|20[0-3]\d)", s)
    if ym:
        year = int(ym.group(1))
    month = ""
    for tok in re.findall(r"[A-Z]{3,}", s):
        if tok in MONTHS:
            month = tok
            break
        # OCR glues the template word onto the value: "inMARCH-2015"
        glued = [m for m in MONTHS if len(m) >= 4 and tok.endswith(m)] or \
                [m for m in MONTHS if len(m) == 3 and tok.endswith(m) and len(tok) <= 5]
        if glued:
            month = glued[0]
            break
        if tok[:4] in MONTH_ABBR or tok[:3] in MONTH_ABBR:
            month = MONTH_ABBR.get(tok[:4]) or MONTH_ABBR[tok[:3]]
            break
        cand = process.extractOne(tok, MONTHS, scorer=fuzz.ratio, score_cutoff=85)
        if cand:
            month = cand[0]
            break
    return month, year, ""


def normalize_categorical(raw: str, vocab: list[str], cutoff: int = 90) -> tuple[str, str]:
    s = collapse_ws(raw).upper().strip(" .,:;")
    if not s:
        return "", "empty"
    if s in vocab:
        return s, ""
    compact = s.replace(" ", "")
    for v in vocab:
        if compact == v.replace(" ", ""):
            return v, f"spacing normalized from {raw!r}"
    cand = process.extractOne(s, vocab, scorer=fuzz.ratio, score_cutoff=cutoff)
    if cand:
        return cand[0], f"fuzzy-mapped from {raw!r} (score {cand[1]:.0f})"
    return collapse_ws(raw), "not in vocabulary; kept verbatim"


def normalize_result(raw: str) -> tuple[str, str]:
    return normalize_categorical(raw, RESULTS)


def normalize_medium(raw: str) -> tuple[str, str]:
    return normalize_categorical(raw, MEDIA)


def normalize_id(raw: str) -> tuple[str, str]:
    s = re.sub(r"[\s.:\-]", "", raw or "").upper()
    return s, ("separators removed" if s != (raw or "") else "")


_DIGIT_WORDS = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7,
                "EIGHT": 8, "NINE": 9, "NIL": 0}


def parse_number_words(raw: str) -> int | None:
    """Digit-by-digit number words as printed on TS/AP memos:
    '*SEVEN**FIVE***TWO*' -> 752, 'THREE SEVEN FOUR' -> 374.
    Every token must be an exact digit word (no fuzzy repair: a misread word
    returns None rather than a guessed number)."""
    tokens = [t for t in re.split(r"[^A-Z]+", (raw or "").upper()) if t]
    if not tokens or len(tokens) > 4 or any(t not in _DIGIT_WORDS for t in tokens):
        return None
    return int("".join(str(_DIGIT_WORDS[t]) for t in tokens))
