"""Document-type, state/board and page-role classification from OCR text.

With 97 Intermediate documents and a single non-target example, a *trained*
document classifier would be learning from one negative. Instead this is an
explicit evidence-based classifier: each decision records the phrases that
triggered it, so a reviewer can see *why* a document was accepted or rejected.

The returned `score` is an evidence-strength heuristic in [0, 1], NOT a
calibrated probability; it is labelled as such in every output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", text.upper())).strip()


# (phrase, weight). Phrases are matched fuzzily against normalized OCR lines,
# because OCR regularly breaks words ("INTERMEDlATE", "MEMORANDUMOF").
DOC_TYPE_EVIDENCE: dict[str, list[tuple[str, float]]] = {
    "intermediate_certificate": [
        ("BOARD OF INTERMEDIATE EDUCATION", 3.0),
        ("INTERMEDIATE EDUCATION", 1.0),
        ("INTERMEDIATE PASS CERTIFICATE", 2.0),
        ("PASS CERTIFICATE CUM MEMORANDUM OF MARKS", 2.0),
        ("INTERMEDIATE PUBLIC EXAMINATION", 2.0),
        ("MEMORANDUM OF MARKS", 1.0),
        ("INTERMEDIATE", 1.0),
        ("OPEN SCHOOL SOCIETY", 1.0),
        ("VOCATIONAL COURSE", 0.5),
    ],
    "senior_school_certificate": [
        ("SENIOR SCHOOL CERTIFICATE", 4.0),
        ("CENTRAL BOARD OF SECONDARY EDUCATION", 3.0),
        ("ALL INDIA SENIOR SCHOOL CERTIFICATE EXAMINATION", 4.0),
        ("NATIONAL INSTITUTE OF OPEN SCHOOLING", 3.0),
    ],
    "secondary_school_certificate": [
        ("SECONDARY SCHOOL CERTIFICATE", 3.0),
        ("BOARD OF SECONDARY EDUCATION", 3.0),
        ("CLASS X", 1.0),
    ],
    "diploma": [
        ("STATE BOARD OF TECHNICAL EDUCATION", 4.0),
        ("DIPLOMA IN", 2.0),
        ("POLYTECHNIC", 1.0),
    ],
}

STATE_EVIDENCE: dict[str, list[tuple[str, float]]] = {
    "Telangana": [("TELANGANA", 3.0), ("TELANGANA STATE BOARD OF INTERMEDIATE EDUCATION", 2.0),
                  ("TELANGANA BOARD OF INTERMEDIATE EDUCATION", 2.0), ("TS B", 0.5)],
    "Andhra Pradesh": [("ANDHRA PRADESH", 3.0), ("BOARD OF INTERMEDIATE EDUCATION A P", 2.0),
                       ("BIE AP GOV IN", 1.0), ("AMARAVATI", 1.0), ("VIJAYAWADA", 0.5)],
}

BOARD_PATTERNS: list[tuple[str, str]] = [
    ("TELANGANA STATE BOARD OF INTERMEDIATE EDUCATION", "Telangana State Board of Intermediate Education"),
    ("TELANGANA BOARD OF INTERMEDIATE EDUCATION", "Telangana Board of Intermediate Education"),
    ("ANDHRA PRADESH OPEN SCHOOL SOCIETY", "Andhra Pradesh Open School Society"),
    ("TELANGANA OPEN SCHOOL SOCIETY", "Telangana Open School Society"),
    ("BOARD OF INTERMEDIATE EDUCATION ANDHRA PRADESH", "Board of Intermediate Education, Andhra Pradesh"),
    ("BOARD OF INTERMEDIATE EDUCATION A P", "Board of Intermediate Education, A.P."),
    ("CENTRAL BOARD OF SECONDARY EDUCATION", "Central Board of Secondary Education"),
]

BOARD_STATE = {
    "Telangana State Board of Intermediate Education": "Telangana",
    "Telangana Board of Intermediate Education": "Telangana",
    "Telangana Open School Society": "Telangana",
    "Andhra Pradesh Open School Society": "Andhra Pradesh",
    "Board of Intermediate Education, Andhra Pradesh": "Andhra Pradesh",
    "Board of Intermediate Education, A.P.": "Andhra Pradesh",
}

PAGE_ROLE_EVIDENCE: dict[str, list[tuple[str, float]]] = {
    # Front side carrying the student's data.
    "certificate": [("THIS IS TO CERTIFY THAT", 2.0), ("FATHER S NAME", 2.0), ("MOTHER S NAME", 2.0),
                    ("FATHER NAME", 2.0), ("MOTHER NAME", 2.0), ("REGISTERED NO", 1.5),
                    ("REGD NUMBER", 1.5), ("HALL TICKET NO", 1.5), ("TOTAL MARKS", 1.0),
                    ("MARKS SECURED", 1.0), ("GRAND TOTAL", 1.0)],
    # Reverse side: eligibility rules / explanatory notes.
    "notes": [("ELIGIBILITY RULES", 2.0), ("AWARDING OF GRADES", 2.0), ("EXPLANATORY NOTE", 2.0),
              ("RANGE OF MARKS PERCENTAGE", 2.0), ("A CANDIDATE IS DECLARED AS PASSED", 1.5),
              ("MINIMUM PASS MARKS", 1.0), ("DEAF DUMB", 1.0), ("NOTE", 0.3)],
}


@dataclass
class Evidence:
    label: str
    score: float
    matched: list[str] = field(default_factory=list)


def _match_phrases(lines_norm: list[str], phrases: list[tuple[str, float]], cutoff: int = 88) -> tuple[float, list[str]]:
    total, matched = 0.0, []
    for phrase, w in phrases:
        best = 0
        for ln in lines_norm:
            if len(ln) < len(phrase) * 0.5:
                continue
            s = fuzz.partial_ratio(phrase, ln) if len(ln) >= len(phrase) else fuzz.ratio(phrase, ln)
            best = max(best, s)
            if best >= 99:
                break
        if best >= cutoff:
            total += w
            matched.append(phrase)
    return total, matched


def _with_line_pairs(lines_norm: list[str]) -> list[str]:
    """Phrases are often split across two OCR lines ("...BOARD | INTERMEDIATE
    EDUCATION"); also match against concatenations of consecutive lines."""
    return lines_norm + [f"{a} {b}" for a, b in zip(lines_norm, lines_norm[1:])]


def _rank(lines_norm: list[str], table: dict[str, list[tuple[str, float]]]) -> list[Evidence]:
    lines_norm = _with_line_pairs(lines_norm)
    ev = []
    for label, phrases in table.items():
        s, m = _match_phrases(lines_norm, phrases)
        ev.append(Evidence(label, s, m))
    return sorted(ev, key=lambda e: -e.score)


def _evidence_score(best: float, second: float, saturation: float = 6.0) -> float:
    """Strength of evidence for the winning label, in [0, 1]. Uncalibrated."""
    if best <= 0:
        return 0.0
    margin = (best - second) / best
    return round(min(best / saturation, 1.0) * (0.5 + 0.5 * margin), 3)


def _has_word(text: str, word: str, cutoff: int = 85) -> bool:
    return any(fuzz.ratio(word, w) >= cutoff for w in text.split())


def _detect_board(lines_norm: list[str]) -> str:
    """Structured board detection.

    Headers are split over several OCR lines and qualifiers are short ("A.P."),
    so plain fuzzy matching of whole board names confuses the boards. Instead:
    locate the core phrase, then read the qualifier immediately before/after it.
    Returns "" when the qualifier is unreadable (e.g. gothic-font 'Telangana'
    OCR'd as 'EUEBUELD') rather than guessing.
    """
    text = " ".join(lines_norm)
    if fuzz.partial_ratio("OPEN SCHOOL SOCIETY", text) >= 90:
        if "ANDHRA PRADESH" in text:
            return "Andhra Pradesh Open School Society"
        if "TELANGANA" in text:
            return "Telangana Open School Society"
        return ""
    if fuzz.partial_ratio("CENTRAL BOARD OF SECONDARY EDUCATION", text) >= 92:
        return "Central Board of Secondary Education"
    core = "BOARD OF INTERMEDIATE EDUCATION"
    al = fuzz.partial_ratio_alignment(core, text)
    if al is None or al.score < 88:
        return _board_fallback(lines_norm, al.score if al is not None else 0.0)
    prefix = text[max(0, al.dest_start - 30):al.dest_start]
    suffix = text[al.dest_end:al.dest_end + 30]
    if _has_word(prefix, "TELANGANA"):
        return ("Telangana State Board of Intermediate Education" if _has_word(prefix, "STATE", 80)
                else "Telangana Board of Intermediate Education")
    if re.match(r"\s*ANDHRA\s*PRADESH", suffix) or fuzz.partial_ratio("ANDHRA PRADESH", suffix[:20]) >= 88:
        return "Board of Intermediate Education, Andhra Pradesh"
    if re.match(r"\s*A\s?P(\s|$)", suffix):
        return "Board of Intermediate Education, A.P."
    return _board_fallback(lines_norm, al.score)


def _board_fallback(lines_norm: list[str], core_score: float) -> str:
    """When the qualifier is not next to the core phrase (OCR scrambled the
    header lines, or the gothic 'Telangana' is unreadable), use evidence that
    still identifies exactly one board:
      * 'STATE BOARD OF INTERMEDIATE' - only the Telangana board is a *State* board;
      * the core phrase + 'TELANGANA' / 'ANDHRA PRADESH' anywhere in the header lines.
    Nothing is inferred from addresses, codes or layout."""
    header = " ".join(lines_norm[:15])
    if fuzz.partial_ratio("STATE BOARD OF INTERMEDIATE", header) >= 88:
        return "Telangana State Board of Intermediate Education"
    if core_score < 85:
        return ""
    words = header.split()
    has = lambda w: any(fuzz.ratio(w, x) >= 88 for x in words)  # noqa: E731
    ts = has("TELANGANA")
    ap = "ANDHRA PRADESH" in header or fuzz.partial_ratio("ANDHRA PRADESH", header) >= 90
    if ts and not ap:
        return "Telangana Board of Intermediate Education"
    if ap and not ts:
        return "Board of Intermediate Education, Andhra Pradesh"
    return ""


# Canonical board identity. The truth file writes both TS board names as
# "Telangana State Board of Intermediate Education" and both AP spellings as
# "Board of Intermediate Education, A.P."; comparisons use these IDs.
BOARD_ID = {
    "Telangana State Board of Intermediate Education": "TSBIE",
    "Telangana Board of Intermediate Education": "TSBIE",
    "Board of Intermediate Education, Andhra Pradesh": "BIEAP",
    "Board of Intermediate Education, A.P.": "BIEAP",
    "Andhra Pradesh Open School Society": "APOSS",
    "Telangana Open School Society": "TOSS",
    "Central Board of Secondary Education": "CBSE",
}


def board_id(name: str) -> str:
    return BOARD_ID.get((name or "").strip(), (name or "").strip())


def classify_page_role(lines: list[str]) -> dict:
    ln = [_norm(l) for l in lines if l.strip()]
    ranked = _rank(ln, PAGE_ROLE_EVIDENCE)
    best, second = ranked[0], ranked[1]
    role = best.label if best.score >= 2.0 else "other"
    return {"role": role, "score": _evidence_score(best.score, second.score, 4.0),
            "evidence": {e.label: e.matched for e in ranked if e.matched}}


def classify_document(pages_lines: list[list[str]]) -> dict:
    """pages_lines: OCR line texts per page."""
    all_lines = [_norm(l) for lines in pages_lines for l in lines if l.strip()]
    ranked = _rank(all_lines, DOC_TYPE_EVIDENCE)
    best, second = ranked[0], ranked[1]
    if best.score < 2.0:
        doc_type = "unknown"
    else:
        doc_type = best.label
    states = _rank(all_lines, STATE_EVIDENCE)
    state = states[0].label if states[0].score >= 3.0 and states[0].score > states[1].score else ""
    board = _detect_board(all_lines)
    state_source = "ocr_text" if state else ""
    if not state and board in BOARD_STATE:
        # The board name identifies the issuing state. Note: before June 2014
        # "Board of Intermediate Education, A.P." also covered today's Telangana;
        # the state reported is the issuing board's state, not the student's.
        state, state_source = BOARD_STATE[board], "board_name"
    subtype = ""
    if doc_type == "intermediate_certificate":
        text = " ".join(all_lines)
        if "OPEN SCHOOL SOCIETY" in text:
            subtype = "open_school_certificate"
        elif "RESULTS MEMO" in text or "COMPUTER GENERATED" in text or "DOWNLOAD" in text:
            subtype = "online_results_memo"
        elif "PASS CERTIFICATE" in text:
            subtype = "pass_certificate_cum_memorandum"
        elif "MEMORANDUM OF MARKS" in text:
            subtype = "memorandum_of_marks"
        if "VOCATIONAL COURSE" in text or re.search(r"SECOND YEAR VOCATIONAL|FIRST YEAR VOCATIONAL", text):
            subtype = (subtype + "+vocational").lstrip("+")
    page_roles = [classify_page_role(lines) for lines in pages_lines]
    return {
        "document_type": doc_type,
        "document_subtype": subtype,
        "score": _evidence_score(best.score, second.score),
        "score_kind": "keyword-evidence heuristic (uncalibrated)",
        "evidence": {e.label: e.matched for e in ranked if e.matched},
        "state": state,
        "state_evidence": {e.label: e.matched for e in states if e.matched},
        "state_source": state_source,
        "board": board,
        "page_roles": page_roles,
        # Supported for extraction = Intermediate document. A missing/unknown
        # state does not block extraction; it is surfaced as a review flag.
        "supported": doc_type == "intermediate_certificate",
        "in_initial_scope": doc_type == "intermediate_certificate" and state in {"Telangana", "Andhra Pradesh"},
    }
