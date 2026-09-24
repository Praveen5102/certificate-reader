"""Rule baseline tagger (exp_001).

The OCR/rule baseline whose numbers were quoted in the spec was not supplied
with the dataset, so this is a re-implementation built on the same idea:
printed field labels ("Father's Name", "Registered No.", ...) locate values;
subject rows are rows inside the marks table that start with a name and
continue with integer (max, marks) pairs.

Confidence here is an *evidence score* (min OCR word confidence x anchor match
quality), not a calibrated probability.
"""
from __future__ import annotations

import re

from ..annotation.anchors import TemplateIndex, best_candidate, table_band
from ..annotation.layout import PageWords, Word, visual_rows
from ..inference.table import Cell
from ..validation.normalize import to_int

SCALAR_FIELDS = ["STUDENT_NAME", "FATHER_NAME", "MOTHER_NAME", "HALL_TICKET", "EXAM_SESSION",
                 "RESULT", "MEDIUM", "TOTAL_MARKS", "TOTAL_IN_WORDS", "CGPA"]
TYPICAL_MAXIMA = {20, 25, 30, 40, 50, 60, 75, 80, 100}
NON_SUBJECT_WORDS = {"PART", "OPTIONAL", "SUBJECTS", "SUBJECT", "ENVIRONMENTAL", "ETHICS", "YEAR", "MAX",
                     "MARKS", "SECURED", "MAXIMUM", "GRADE", "POINT", "TOTAL"}
_SUBJ_TOKEN = re.compile(r"^[A-Z][A-Z&().\-]*$|^[&\-]$")


class RuleTagger:
    name = "rule_baseline_v1"

    def __init__(self, fields_cfg: dict):
        self.cfg = fields_cfg

    def tag(self, pages: list[PageWords], tmpls: dict[int, TemplateIndex]) -> dict[str, list[Cell]]:
        out: dict[str, list[Cell]] = {}
        for f in SCALAR_FIELDS:
            c = best_candidate(pages, f, self.cfg["anchors"][f], tmpls)
            if c is not None:
                out[f] = [Cell(c.words, f, round(c.ocr_conf * c.anchor_score / 100.0, 4))]
        subj, mx, mk = [], [], []
        for p in pages:
            if p.role == "notes":
                continue
            band = table_band(p, self.cfg["table"]["header_phrases"], self.cfg["table"]["end_phrases"])
            if band is None:
                continue
            top, bot = band
            words = [w for w in p.words if top <= w.cy <= bot]
            for row in visual_rows(words):
                s, a, b = self._row(row)
                subj += s
                mx += a
                mk += b
        out["SUBJECT"], out["MAX_MARKS"], out["MARKS"] = subj, mx, mk
        return out

    @staticmethod
    def _row(row: list[Word]):
        name: list[Word] = []
        i = 0
        # skip row prefixes such as "Part-I :" / "Part - 3:"
        while i < len(row) and (re.match(r"(?i)^part", row[i].text) or row[i].text.strip(" :") in {"", "I", "II", "III",
                                                                                                   "1", "2", "3", "-"}):
            i += 1
        while i < len(row) and _SUBJ_TOKEN.match(row[i].text.strip(" :")) and to_int(row[i].text)[0] is None:
            name.append(row[i])
            i += 1
        if not name or {w.text.strip(" :-").upper() for w in name} & NON_SUBJECT_WORDS:
            return [], [], []
        ints = [w for w in row[i:] if to_int(w.text)[0] is not None and "." not in w.text]
        vals = [to_int(w.text)[0] for w in ints]
        mx, mk = [], []
        k = 0
        while k + 1 < len(ints):
            if vals[k] in TYPICAL_MAXIMA and vals[k + 1] <= vals[k]:
                mx.append(Cell([ints[k]], "MAX_MARKS", ints[k].conf))
                mk.append(Cell([ints[k + 1]], "MARKS", ints[k + 1].conf))
                k += 2
            else:
                k += 1
        if not mk:
            return [], [], []
        conf = min(w.conf for w in name)
        return [Cell(name, "SUBJECT", conf)], mx, mk
