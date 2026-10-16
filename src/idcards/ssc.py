"""10th class (SSC) certificate extraction - Board of Secondary Education,
Telangana / Andhra Pradesh.

Layouts handled (2005-2024): grades with grade points (with or without the FA /
SA columns), the older "with GPA 9.3 ... GRADE POINTS TOTAL" memos, and marks
memos (marks out of 100 per subject, GRAND TOTAL).

Header fields are read row by row (label and value on one printed row, e.g.
"FATHER'S NAME : R SURESH KUMAR"); subject rows are anchored on the printed
subject areas (FIRST LANGUAGE ... SECOND LANGUAGE). Built-in checks: each grade
must match its grade point, the CGPA must be the average of the grade points,
printed totals must equal the sum of the subjects, and numbers printed in words
must agree with the figures. Anything that disagrees is flagged, not corrected.
"""
from __future__ import annotations

import re
from typing import Any

from .common import (TO_DIGIT, Line, clean_name, field, finish, lines_of, name_like, ocr_upright,
                     parse_date)

KEYWORDS = ["SECONDARY", "SCHOOL CERTIFICATE", "BOARD", "GRADE", "DATE OF BIRTH", "MEDIUM", "LANGUAGE",
            "MATHEMATICS", "SOCIAL STUDIES", "FATHER", "MOTHER", "EXAMINATION"]
GRADE_POINTS = {"A1": 10, "A2": 9, "B1": 8, "B2": 7, "C1": 6, "C2": 5, "D1": 4, "D2": 3, "D": 4}
AREAS = [("L1", "First language", r"FIRST\s*LANGUAGE"), ("L2", "Second language", r"SECOND\s*LANGUAGE"),
         ("L3", "Third language", r"THIRD\s*LANGUAGE"), ("MA", "Mathematics", r"MATHEMATICS"),
         ("SC", "Science", r"(?:GENERAL\s*)?SCIENCE"), ("SS", "Social studies", r"SOCIAL\s*STUDIES")]
AREA_SUBJECT = {"MA": "MATHEMATICS", "SC": "SCIENCE", "SS": "SOCIAL STUDIES"}
WORD_DIGITS = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6",
               "SEVEN": "7", "EIGHT": "8", "NINE": "9"}
_GRADE = re.compile(r"^(A1|A2|B1|B2|C1|C2|D1|D2|D|E)\*?$")
_MONTHS = "JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER"


# ---------------------------------------------------------------- rows
class Row:
    """Lines printed on one row, left to right."""

    def __init__(self, first: Line):
        self.lines = [first]

    @property
    def cy(self) -> float:
        return sum(l.cy for l in self.lines) / len(self.lines)

    @property
    def h(self) -> float:
        return min(l.h for l in self.lines)

    @property
    def text(self) -> str:
        return " ".join(l.text for l in sorted(self.lines, key=lambda l: l.x0))

    @property
    def conf(self) -> float:
        return min(l.conf for l in self.lines)


def _overlap(a: Line, b: Line) -> float:
    return min(a.y1, b.y1) - max(a.y0, b.y0)


def _rows(lines: list[Line]) -> list[Row]:
    rows: list[Row] = []
    for l in sorted(lines, key=lambda l: (l.page, l.cy)):
        for r in rows[-4:]:
            a = max(r.lines, key=lambda x: x.h)
            if a.page == l.page and _overlap(a, l) >= 0.55 * min(a.h, l.h) and abs(a.cy - l.cy) < 0.6 * max(a.h, l.h):
                r.lines.append(l)
                break
        else:
            rows.append(Row(l))
    return rows


def _deskew(lines: list[Line]) -> float:
    """Phone photos are often slightly tilted: values at the right end of a row sit
    higher or lower than the label at its left. Estimate that slope from the subject
    table (grade / number cells against their subject labels) and straighten the
    line positions. Returns the slope used (0.0 when the page is level)."""
    anchors = [l for l in lines if any(re.match(rf"\W*{pat}\b", l.text.upper()) for _, _, pat in AREAS)]
    cells = [l for l in lines if re.fullmatch(r"(A1|A2|B1|B2|C1|C2|D1|D2|D|\d{1,3})\*?", l.text.upper().strip(" .:*"))]
    if len(anchors) < 3 or len(cells) < 6:
        return 0.0

    def score(k):
        n = 0
        for c in cells:
            cx = (c.x0 + c.x1) / 2
            for a in anchors:
                if a.page == c.page and c.x0 > a.x1 and abs((c.cy - k * (cx - a.x0)) - a.cy) < 0.3 * a.h:
                    n += 1
                    break
        return n
    best = max((score(k / 1000), -abs(k), k / 1000) for k in range(-60, 61, 2))
    k = best[2] if best[0] > score(0.0) else 0.0
    if k:
        for l in lines:
            d = k * ((l.x0 + l.x1) / 2)
            l.y0 -= d
            l.y1 -= d
    return k


def _after(text: str, label: str) -> str:
    """Text after a printed label, without the ':' / '(' punctuation around it."""
    m = re.search(label, text, re.I)
    if not m:
        return ""
    return text[m.end():].strip(" :;.,-(")


def _words_number(text: str) -> str:
    """'ZERO EIGHT' -> '08'; '' if the text is not entirely digit words."""
    ws = re.findall(r"[A-Z]+", text.upper())
    if not ws or any(w not in WORD_DIGITS for w in ws):
        return ""
    return "".join(WORD_DIGITS[w] for w in ws)


# ---------------------------------------------------------------- header
LABELS = {
    "student_name": r"CERTIFIED\s*THAT",
    "father_name": r"FATHER'?S?\s*NAME|FATHER\s*NAME",
    "mother_name": r"MOTHER'?S?\s*NAME|MOTHER\s*NAME",
    "roll_number": r"(?:BEARING\s*)?ROLL\s*NO\.?",
    "date_of_birth": r"DATE\s*OF\s*BIRTH(?:\s*OF\s*THE)?",
    "school": r"BELONG(?:ING|S)\s*TO|\bSCHOOL\b\s*:|^\W*SCHOOL\b",
    "medium": r"^\W*MEDIUM",
}


def _header(rows: list[Row], all_lines: list[Line]) -> tuple[dict[str, tuple[str, Row]], dict]:
    got: dict[str, tuple[str, Row]] = {}
    for r in rows:
        t = r.text
        for key, label in LABELS.items():
            if key in got or not re.search(label, t, re.I):
                continue
            v = _after(t, label)
            if key == "roll_number":
                m = re.search(r"\b(\d{6,12})\b", v.translate(TO_DIGIT) if re.search(r"\d{4}", v) else v)
                v = m.group(1) if m else ""
            elif key == "date_of_birth":
                v = parse_date(v)
            elif key in ("student_name", "father_name", "mother_name"):
                # 'RAVI KUMAR T' / old memos print 'S/O <father>' under the name instead of a label
                v = re.split(r"\s{2,}|\bBEARING\b|\bS\s*/\s*O\b", v, flags=re.I)[0]
                v = clean_name(v)
                if not name_like(v):
                    v = ""
            elif key == "school":
                v = clean_name(v)
            elif key == "medium":
                v = v.split()[0] if v else ""
            if not v and key in ("date_of_birth", "school", "medium", "roll_number"):
                lab = next((l for l in r.lines if re.search(label, l.text, re.I)), r.lines[0])
                near = sorted((l for l in all_lines if l.page == lab.page and l.x0 > lab.x1 - 5
                               and abs(l.cy - lab.cy) < (2.2 if key == "date_of_birth" else 1.0) * lab.h),
                              key=lambda l: (abs(l.cy - lab.cy), l.x0))
                for l in near:
                    t2 = l.text.strip(" :;.,-(")
                    if key == "date_of_birth":
                        v = parse_date(t2)
                    elif key == "roll_number":
                        m = re.fullmatch(r"\d{6,12}", t2)
                        v = m.group(0) if m else ""
                    elif key == "medium":
                        v = t2 if re.fullmatch(r"[A-Z]+", t2.upper()) else ""
                    else:
                        v = clean_name(t2) if len(t2) > 8 else ""
                    if v:
                        r = Row(l)
                        break
            if v:
                got[key] = (v, r)
    extra: dict = {}
    for r in rows:
        t = r.text.upper()
        m = re.search(r"\b[SD]\s*/\s*O\s+([A-Z][A-Z .]+?)(?:\s+BEARING\b|$)", t)
        if m and "father_name" not in got and name_like(m.group(1)):
            got["father_name"] = (clean_name(m.group(1)), r)
        m = re.search(rf"HELD\s*IN\s*((?:{_MONTHS})(?:\s*/\s*(?:{_MONTHS}))?)\W*((?:19|20)\d{{2}})", t)
        if m and "exam" not in extra:
            extra["exam"] = (f"{re.sub(r'\s+', '', m.group(1))} {m.group(2)}", r)
        m = re.search(r"\b(?:AND|WITH)\s+([A-Z]+)\s+AS\s+(?:THE\s+)?MEDIUM", t)
        if m and "medium" not in got:
            got["medium"] = (m.group(1), r)
        m = re.search(r"\b(REGULAR|PRIVATE)\b", t)
        if m and "appearance" not in extra:
            extra["appearance"] = (m.group(1), r)
        if "COMPARTMENTAL" in t:
            extra["compartmental"] = True
    # a faint, unreadable label: the parent's name still sits on its own row between the
    # candidate's row and the next labelled row
    order = ["student_name", "father_name", "mother_name"]
    for i in (1, 2):
        key, prev, nxt = order[i], order[i - 1], (order[i + 1] if i + 1 < len(order) else "roll_number")
        if key in got or prev not in got:
            continue
        p_row = got[prev][1]
        n_row = got[nxt][1] if nxt in got else None
        cands = [r for r in rows if r.cy > p_row.cy + 0.5 * p_row.h and (n_row is None or r.cy < n_row.cy - 0.5 * n_row.h)
                 and r.cy - p_row.cy < 4 * p_row.h]
        for r in cands:
            v = clean_name(re.sub(r"^.*?\bNAME\b\W*", "", r.text, flags=re.I))
            if name_like(v, min_words=2) and v.upper() == v:
                got[key] = (v, r)
                break
    return got, extra


# ---------------------------------------------------------------- subjects
def _subject_rows(lines: list[Line]) -> list[dict]:
    out = []
    for code, area_name, pat in AREAS:
        anchors = [l for l in lines if re.match(rf"\W*{pat}\b", l.text.upper())
                   and not re.search(r"EDUCATION|SKILLS", l.text.upper())]
        if not anchors:
            continue
        a = min(anchors, key=lambda l: (l.page, l.y0))
        same = [l for l in lines if l.page == a.page and l is not a and l.x0 > a.x0
                and abs(l.cy - a.cy) < 0.6 * max(a.h, l.h)]
        tokens = sorted(same, key=lambda l: l.x0)
        lang = re.sub(rf"^\W*{pat}\W*", "", a.text.upper()).strip(" :;(),.")
        grade = gp = marks = None
        words = ""
        conf = min([a.conf] + [t.conf for t in tokens])
        for t in tokens:
            s = re.sub(r"^[^A-Z0-9]+|[^A-Z0-9*]+$", "", t.text.upper())
            g = s.replace(" ", "")
            if _GRADE.match(g):
                grade = g.rstrip("*")          # rightmost grade = overall grade (FA / SA come first)
            elif re.fullmatch(r"\d{1,3}\*?", g.translate(TO_DIGIT)):
                n = int(g.translate(TO_DIGIT).rstrip("*"))
                if gp is None and marks is None:
                    (gp, marks) = (n, None) if n <= 10 and len(g.rstrip("*")) <= 2 else (None, n)
            elif _words_number(s):
                words = _words_number(s)
            elif not lang and re.fullmatch(r"[A-Z/ .]+", s) and len(s) >= 4:
                lang = s
        if grade is None and gp is not None and marks is None and gp > 10:
            gp, marks = None, gp
        lang = re.sub(r"\s+", " ", lang.replace(" / ", "/")).strip()
        out.append({"area": code, "area_name": area_name,
                    "subject": AREA_SUBJECT.get(code) or lang or ("ENGLISH" if code == "L3" else ""),
                    "grade": grade, "grade_point": gp, "marks": marks, "words": words,
                    "y": (a.page, a.y0), "confidence": round(conf, 4)})
    return sorted(out, key=lambda r: r["y"])


def _number_on_row(rows: list[Row], label: str, pattern: str) -> tuple[str, Row] | None:
    for r in rows:
        if re.search(label, r.text, re.I):
            vals = re.findall(pattern, _after(r.text, label))
            if vals:
                return vals[-1], r
    return None


# ---------------------------------------------------------------- main
def extract_ssc(ocr_pages: list[dict]) -> dict[str, Any]:
    lines = lines_of(ocr_pages)
    skew = _deskew(lines)
    rows = _rows(lines)
    alltext = " ".join(l.text.upper() for l in lines)
    fields: dict[str, dict] = {}
    flags: dict[str, str] = {}
    checks: dict[str, Any] = {}

    is_ssc = bool(re.search(r"SECONDARY\s*SCHOOL|SECONDARY\s*EDUCATION|\bSSC\b", alltext))
    if not is_ssc or re.search(r"BOARD\s*OF\s*INTERMEDIATE|INTERMEDIATE\s*EDUCATION", alltext):
        note = ("This looks like an Intermediate (12th) certificate - choose '12th / Intermediate' instead."
                if "INTERMEDIATE" in alltext else "This does not look like a 10th class (SSC) certificate.")
        for k in ("student_name", "date_of_birth", "roll_number"):
            fields[k] = field("", "not found")
        r = finish("not_ssc", fields, {"document": note}, [], {}, note)
        r["review"]["status"] = "manual_required"
        r["subjects"] = []
        return r

    state = "TELANGANA" if re.search(r"TELANGANA", alltext[:600]) else \
        ("ANDHRA PRADESH" if re.search(r"ANDHRA\s*PRADESH", alltext[:600]) else "")
    fields["board"] = field(f"Board of Secondary Education, {state.title()}" if state else "Board of Secondary Education",
                            "printed text")
    fields["state"] = field(state, "printed text" if state else "not found")

    got, extra = _header(rows, lines)
    def conf_of(v, r):
        """Confidence of the line(s) holding the value, not of everything else on its row."""
        if r is None:
            return None
        nv = re.sub(r"[^A-Z0-9]", "", str(v).upper())
        own = [l.conf for l in r.lines if (nl := re.sub(r"[^A-Z0-9]", "", l.text.upper())) and (nl in nv or nv in nl)]
        return round(min(own), 4) if own else r.conf

    for key in ("student_name", "father_name", "mother_name", "roll_number", "date_of_birth", "school", "medium"):
        v, r = got.get(key, ("", None))
        fields[key] = field(v, "printed text" if v else "not found", conf_of(v, r) if v else None,
                            raw=r.text if r else None)
    ex, er = extra.get("exam", ("", None))
    fields["exam"] = field(ex, "printed text" if ex else "not found", er.conf if er else None, raw=er.text if er else None)
    ap, _ = extra.get("appearance", ("", None))
    fields["appearance"] = field(ap + (" (compartmental)" if extra.get("compartmental") else ""), "printed text")

    # ---- subjects and results
    subs = _subject_rows(lines)
    graded = sum(1 for s in subs if s["grade"]) >= max(1, len(subs) // 2)
    fields["result_type"] = field("grades" if graded else "marks", "layout")
    for s in subs:
        if graded and s["grade_point"] is None and s["words"]:
            s["grade_point"] = int(s["words"])
        if not graded and s["marks"] is None and s["words"]:
            s["marks"] = int(s["words"])
    bad_rows = []
    for i, s in enumerate(subs):
        s["review"] = "auto"
        if graded:
            exp = GRADE_POINTS.get(s["grade"] or "")
            if s["grade"] is None or s["grade_point"] is None:
                s["review"] = "manual_required"
                flags[f"subjects[{i}]"] = f"{s['area_name']}: grade or grade point not read"
            elif exp is not None and exp != s["grade_point"]:
                s["review"] = "review_recommended"
                flags[f"subjects[{i}]"] = f"{s['area_name']}: grade {s['grade']} should carry {exp} points, read {s['grade_point']}"
            elif s["words"] and int(s["words"]) != s["grade_point"]:
                s["review"] = "review_recommended"
                flags[f"subjects[{i}]"] = f"{s['area_name']}: figures {s['grade_point']} and words {s['words']} differ"
        else:
            if s["marks"] is None:
                s["review"] = "manual_required"
                flags[f"subjects[{i}]"] = f"{s['area_name']}: marks not read"
            elif s["words"] and int(s["words"]) != s["marks"]:
                s["review"] = "review_recommended"
                flags[f"subjects[{i}]"] = f"{s['area_name']}: figures {s['marks']} and words {s['words']} differ"
        if s["review"] != "auto":
            bad_rows.append(i)
    if len(subs) < 6:
        flags["subjects"] = f"only {len(subs)} of 6 subject rows were read"

    cg = _number_on_row(rows, r"\bC?GPA\b\W*(?:\(CGPA\))?", r"(?<![\d.])(\d{1,2}\.\d)(?![\d.])")
    cgpa = float(cg[0]) if cg else None
    fields["cgpa"] = field(cgpa, "printed text" if cg else "not printed" if not graded else "not found",
                           cg[1].conf if cg else None, raw=cg[1].text if cg else None)
    tot = _number_on_row(rows, r"GRAND\s*TOTAL", r"\b(\d{2,3})\b") or _number_on_row(rows, r"^\W*TOTAL\b", r"\b(\d{2,3})\b")
    total = int(tot[0]) if tot and not graded else None
    gpt = _number_on_row(rows, r"GRADE\s*POINTS?\s*TOTAL", r"\b(\d{1,2})\b") if graded else None
    fields["total_marks"] = field(total, "printed text" if total is not None else "not printed",
                                  tot[1].conf if (tot and total is not None) else None)
    fields["maximum_marks"] = field(100 * len(subs) if (not graded and subs) else None, "100 per subject" if not graded else "not printed")
    pct = round(100 * total / (100 * len(subs)), 2) if (total is not None and subs) else None
    fields["percentage"] = field(pct, "derived: total / maximum" if pct is not None else "not printed")

    if graded:
        gps = [s["grade_point"] for s in subs if s["grade_point"] is not None]
        if len(gps) == len(subs) == 6:
            mean = sum(gps) / 6
            checks["grade_point_sum"] = sum(gps)
            if cgpa is not None:
                ok = abs(cgpa - round(mean + 1e-9, 1)) < 0.051
                checks["cgpa_matches_grade_points"] = ok
                fields["cgpa"]["verified"] = ok and not bad_rows
                if not ok:
                    flags["cgpa"] = f"printed {cgpa} but the subject grade points average {mean:.2f}"
            if gpt:
                okt = int(gpt[0]) == sum(gps)
                checks["grade_points_total_matches"] = okt
                if not okt:
                    flags["cgpa"] = f"printed grade points total {gpt[0]} but the subjects add up to {sum(gps)}"
        if cgpa is None:
            flags["cgpa"] = "CGPA / GPA not found"
    else:
        ms = [s["marks"] for s in subs if s["marks"] is not None]
        if total is not None and len(ms) == len(subs):
            ok = sum(ms) == total
            checks["total_matches_subjects"] = ok
            fields["total_marks"]["verified"] = ok
            if not ok:
                flags["total_marks"] = f"printed total {total} but the subjects add up to {sum(ms)}"
        elif total is None:
            flags["total_marks"] = "grand total not found"

    for k in ("student_name", "father_name", "mother_name", "roll_number", "date_of_birth", "school"):
        c = fields[k]["confidence"]
        if fields[k]["value"] and c is not None and c < 0.8:
            flags.setdefault(k, f"low OCR confidence ({c:.2f})")
    if fields["roll_number"]["value"] and not re.fullmatch(r"\d{7,10}", fields["roll_number"]["value"]):
        flags.setdefault("roll_number", "unusual roll number length")

    if skew:
        checks["tilt_corrected"] = f"{skew:+.3f}"
    r = finish("ssc", fields, flags, ["student_name", "date_of_birth", "roll_number"], checks)
    r["subjects"] = [{k: s[k] for k in ("area", "area_name", "subject", "grade", "grade_point", "marks", "confidence", "review")}
                     for s in subs]
    return r


def flat(result: dict) -> dict:
    out = {k: v["value"] for k, v in result["fields"].items()} | {"document_type": result["document_type"]}
    out["subjects"] = [{k: s[k] for k in ("area_name", "subject", "grade", "grade_point", "marks")}
                       for s in result.get("subjects", [])]
    return out


def extract_ssc_images(images: list, ocr_engine) -> dict[str, Any]:
    pages = [ocr_upright(img, ocr_engine, i + 1, KEYWORDS) for i, img in enumerate(images)]
    r = extract_ssc(pages)
    r["pages"] = len(images)
    return r
