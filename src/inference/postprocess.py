"""Tagged cells -> normalized, validated, confidence-scored ExtractionResult.

Shared by the rule baseline and the fine-tuned model so both are evaluated
through identical normalization / validation / review logic.

Confidence (documented, not invented):
  * each tagger supplies a per-span score (model: min token softmax probability
    x min OCR word confidence; rules: min OCR confidence x anchor match quality);
  * two independent readings that agree (total in figures and in words) are
    combined by noisy-OR: 1 - (1-a)(1-b);
  * derived values (maximum_marks, percentage, group) take the minimum of their
    inputs' confidences.
Scores are uncalibrated until thresholds are fitted on the validation split.
Review level = f(confidence, validation issues touching the field).
"""
from __future__ import annotations

from typing import Any

from ..annotation.layout import PageWords
from ..preprocessing.classify import board_id
from ..validation.checks import run_checks
from ..validation.normalize import (normalize_id, normalize_medium, normalize_month_year, normalize_name,
                                    normalize_result, parse_number_words, to_float, to_int)
from .schema import Evidence, ExtractionResult, FieldValue, ReviewSummary, SubjectRow
from . import open_school_table, transposed_table
from .table import Cell, reconstruct

NAME_FIELDS = {"STUDENT_NAME": "student_name", "FATHER_NAME": "father_name", "MOTHER_NAME": "mother_name"}
GROUPS = {  # derived from subject set; only exact matches are reported
    "MPC": {"MATHEMATICS", "PHYSICS", "CHEMISTRY"},
    "BIPC": {"BOTANY", "ZOOLOGY", "PHYSICS", "CHEMISTRY"},
    "MEC": {"MATHEMATICS", "ECONOMICS", "COMMERCE"},
    "CEC": {"CIVICS", "ECONOMICS", "COMMERCE"},
    "HEC": {"HISTORY", "ECONOMICS", "CIVICS"},
}


def _evidence(cell: Cell) -> list[Evidence]:
    return [Evidence(page=cell.words[0].page, text=cell.text, bbox=list(cell.bbox),
                     ocr_confidence=round(min(w.conf for w in cell.words), 4))]


def _best(tagged: dict[str, list[Cell]], label: str) -> Cell | None:
    cells = tagged.get(label) or []
    return max(cells, key=lambda c: c.confidence) if cells else None


def _fv(value, cell: Cell | None, conf_source: str, normalization: str = "", derived: bool = False,
        confidence: float | None = None) -> FieldValue:
    return FieldValue(value=value, raw=cell.text if cell else None,
                      confidence=round(confidence if confidence is not None else (cell.confidence if cell else 0.0), 4)
                      if (cell or confidence is not None) else None,
                      confidence_source=conf_source, normalization=normalization, derived=derived,
                      evidence=_evidence(cell) if cell else [])


def _subject_key(name: str) -> str:
    n = "".join(ch for ch in (name or "").upper() if ch.isalpha() or ch == " ")
    if n.startswith("MATHEMATICS"):
        return "MATHEMATICS"
    if "CIVICS" in n or "POLITICAL SCIENCE" in n:
        return "CIVICS"
    return n.replace("PRACTICALS", "").replace("PRACTICAL", "").strip()


def build_result(file: str, classification: dict, pages: list[PageWords], tagged: dict[str, list[Cell]],
                 fields_cfg: dict, inference_cfg: dict, provenance: dict[str, Any],
                 table_fallback: dict[str, list[Cell]] | None = None) -> ExtractionResult:
    conf_src = provenance.get("confidence_source", "")
    doc_type = classification.get("document_type", "unknown")
    if not classification.get("supported", False):
        return ExtractionResult(
            file=file, document_type="unsupported" if doc_type != "unknown" else "unknown",
            document_type_score=classification.get("score"),
            document_type_score_kind=classification.get("score_kind", ""), supported=False, pages=len(pages),
            review=ReviewSummary(status="manual_required", fields_requiring_review=["document_type"],
                                 reasons={"document_type": f"classified as {doc_type}; extraction not attempted"}),
            notes=f"Document classified as {doc_type} (evidence: {classification.get('evidence')}); "
                  f"only Intermediate certificates are extracted.",
            provenance=provenance)

    fields: dict[str, FieldValue] = {}
    fields["state"] = FieldValue(value=classification.get("state", ""), confidence=classification.get("score"),
                                 confidence_source="document classifier evidence (uncalibrated)",
                                 normalization=f"from {classification.get('state_source') or 'n/a'}")
    fields["board"] = FieldValue(value=classification.get("board", ""), confidence=classification.get("score"),
                                 confidence_source="document classifier evidence (uncalibrated)",
                                 normalization=f"board id {board_id(classification.get('board', ''))}")
    fields["document_subtype"] = FieldValue(value=classification.get("document_subtype", ""),
                                            confidence=classification.get("score"),
                                            confidence_source="document classifier evidence (uncalibrated)")

    for label, key in NAME_FIELDS.items():
        c = _best(tagged, label)
        if c:
            v, note = normalize_name(c.text)
            fields[key] = _fv(v, c, conf_src, note)
        else:
            fields[key] = FieldValue(value="", confidence=None, confidence_source="not found")

    c = _best(tagged, "HALL_TICKET")
    if c:
        v, note = normalize_id(c.text)
        fields["hall_ticket_number"] = _fv(v, c, conf_src, note)
    else:
        fields["hall_ticket_number"] = FieldValue(value="", confidence_source="not found")

    c = _best(tagged, "EXAM_SESSION")
    if c:
        month, year, _ = normalize_month_year(c.text)
        fields["month_of_pass"] = _fv(month, c, conf_src, "parsed from exam session text")
        fields["year_of_pass"] = _fv(year, c, conf_src, "parsed from exam session text")
    else:
        fields["month_of_pass"] = FieldValue(value="", confidence_source="not found")
        fields["year_of_pass"] = FieldValue(value=None, confidence_source="not found")

    c = _best(tagged, "RESULT")
    if c:
        v, note = normalize_result(c.text)
        fields["result"] = _fv(v, c, conf_src, note)
    else:
        fields["result"] = FieldValue(value="", confidence_source="not found")

    c = _best(tagged, "MEDIUM")
    if c:
        v, note = normalize_medium(c.text)
        fields["medium"] = _fv(v, c, conf_src, note)
    else:
        fields["medium"] = FieldValue(value="", confidence_source="not found")

    c = _best(tagged, "CGPA")
    if c:
        v, note = to_float(c.text)
        fields["cgpa"] = _fv(v, c, conf_src, note)

    # ---- total: two independent readings -----------------------------------------------
    cd, cw = _best(tagged, "TOTAL_MARKS"), _best(tagged, "TOTAL_IN_WORDS")
    vd = to_int(cd.text)[0] if cd else None
    vw = parse_number_words(cw.text) if cw else None
    if vd is not None and vw is not None and vd == vw:
        conf = 1 - (1 - cd.confidence) * (1 - cw.confidence)
        fields["total_marks"] = _fv(vd, cd, conf_src + "; figures and words agree (noisy-OR)", "", confidence=conf)
        fields["total_marks"].evidence += _evidence(cw)
    elif vd is not None:
        fields["total_marks"] = _fv(vd, cd, conf_src, to_int(cd.text)[1])
        if cw is not None and vw is not None:
            fields["total_marks"].evidence += _evidence(cw)
    elif vw is not None:
        fields["total_marks"] = _fv(vw, cw, conf_src + "; read from total in words", "parsed number words")
    else:
        fields["total_marks"] = FieldValue(value=None, confidence_source="not found")

    # ---- subjects ---------------------------------------------------------------------------
    # Primary table from the main tagger. If its marks do not add up to the printed
    # total but the fallback tagger's table does, the fallback table is used: the
    # printed total is an independent check, so this selects - it never edits values.
    subjects = subject_rows(pages, tagged, fields_cfg, conf_src)
    table_source = "primary"
    total = fields["total_marks"].value
    if table_fallback and isinstance(total, int) and not _adds_up(subjects, total):
        alt = subject_rows(pages, table_fallback, fields_cfg, provenance.get("fallback_confidence_source", ""))
        if _adds_up(alt, total):
            subjects, table_source = alt, "fallback: primary table did not add up to the printed total"
    provenance = {**provenance, "table_source": table_source}

    # ---- derived fields ----------------------------------------------------------------------
    if subjects and all(s.maximum_marks.value is not None for s in subjects):
        gmax = sum(s.maximum_marks.value for s in subjects)
        conf = min(s.row_confidence or 0 for s in subjects)
        fields["maximum_marks"] = FieldValue(value=gmax, confidence=round(conf, 4),
                                             confidence_source="min over subject rows",
                                             normalization="sum of subject maximum marks", derived=True)
    else:
        fields["maximum_marks"] = FieldValue(value=None, confidence_source="not derivable (subject maxima missing)",
                                             derived=True)
    tv, gm = fields["total_marks"].value, fields["maximum_marks"].value
    if isinstance(tv, int) and isinstance(gm, int) and gm > 0:
        pct = round(100.0 * tv / gm, 2)
        conf = min(fields["total_marks"].confidence or 0, fields["maximum_marks"].confidence or 0)
        fields["percentage"] = FieldValue(value=pct, confidence=round(conf, 4), confidence_source="min of inputs",
                                          normalization="100 * total_marks / maximum_marks (not printed)",
                                          derived=True)
    else:
        fields["percentage"] = FieldValue(value=None, confidence_source="not derivable", derived=True)
    keys = {_subject_key(s.subject_name.value) for s in subjects}
    group = next((g for g, need in GROUPS.items() if need <= keys), "")
    fields["group"] = FieldValue(value=group, confidence=min((s.row_confidence or 0 for s in subjects), default=None)
                                 if group else None, confidence_source="derived from subject set",
                                 normalization="derived: exact subject-set match" if group else "", derived=True)
    fields["course"] = FieldValue(value="", confidence_source="not extracted in v1 (no truth labels)")

    extra = {"total_in_words": vw, "total_in_digits": vd}
    issues = run_checks(fields, subjects, extra)
    review = route_review(fields, subjects, issues, inference_cfg["review"])
    return ExtractionResult(
        file=file, document_type="intermediate_certificate", document_type_score=classification.get("score"),
        document_type_score_kind=classification.get("score_kind", ""), supported=True, pages=len(pages),
        fields=fields, subjects=subjects, validation=issues, review=review, provenance=provenance,
        notes="; ".join(i.reason for i in issues if i.status == "error"))


def _adds_up(subjects: list[SubjectRow], total: int) -> bool:
    marks = [s.marks.value for s in subjects]
    return bool(marks) and all(isinstance(m, int) for m in marks) and sum(marks) == total


def subject_rows(pages: list[PageWords], tagged: dict[str, list[Cell]], fields_cfg: dict,
                 conf_src: str) -> list[SubjectRow]:
    from ..annotation.align import year_boundary
    subjects: list[SubjectRow] = []
    tcfg = fields_cfg["table"]
    for p in pages:
        if p.role == "notes":
            continue
        on_page = lambda cells: [c for c in (cells or []) if c.words[0].page == p.page]  # noqa: E731
        # layout-specific readers first (each with a sanity gate); otherwise the
        # general tagged-cell reconstruction
        rows = transposed_table.parse(p) if transposed_table.detect(p) else []
        if not transposed_table.plausible(rows):
            rows = open_school_table.parse(p) if open_school_table.detect(p) else []
            if not open_school_table.plausible(rows):
                rows = []
        if not rows:
            rows = reconstruct(p, on_page(tagged.get("SUBJECT")), on_page(tagged.get("MAX_MARKS")),
                               on_page(tagged.get("MARKS")), year_boundary(p, tcfg["year_headers"]),
                               tcfg["practical_markers"])
        for r in rows:
            name, note = normalize_name(r.name_override or r.subject.text)
            mk, mk_note = to_int(r.marks.text)
            mx, mx_note = (to_int(r.maximum.text) if r.maximum else (None, "not found"))
            confs = [r.subject.confidence, r.marks.confidence] + ([r.maximum.confidence] if r.maximum else [])
            subjects.append(SubjectRow(
                subject_name=_fv(name, r.subject, conf_src, note),
                paper=FieldValue(value=r.paper, confidence=None, confidence_source=r.paper_source,
                                 derived=True),
                marks=_fv(mk, r.marks, conf_src, mk_note),
                maximum_marks=_fv(mx, r.maximum, conf_src, mx_note) if r.maximum else
                FieldValue(value=None, confidence_source="not found"),
                row_confidence=round(min(confs), 4)))
    return subjects


def route_review(fields: dict[str, FieldValue], subjects: list[SubjectRow], issues, rcfg: dict) -> ReviewSummary:
    hi, med = rcfg["high_confidence"], rcfg["medium_confidence"]
    required = set(rcfg.get("required_fields", []))
    recommended = set(rcfg.get("recommended_fields", []))
    bad = {}
    for i in issues:
        root = i.field.split(".")[0].split("[")[0]
        if i.status in ("error", "warning"):
            bad.setdefault(root, []).append(i)
    levels, reasons = {}, {}
    for name, fv in fields.items():
        if name in ("document_subtype",):
            continue
        if fv.value in (None, ""):
            lvl = "manual_required" if name in required else ("review_recommended" if name in recommended else "auto")
            if name in required:
                reasons[name] = "required field not extracted"
            elif name in recommended:
                reasons[name] = "could not be read from the certificate - please fill in"
        elif fv.confidence is None:
            lvl = "review_recommended"
            reasons[name] = "no confidence signal"
        elif fv.confidence >= hi:
            lvl = "auto"
        elif fv.confidence >= med:
            lvl = "review_recommended"
            reasons[name] = f"confidence {fv.confidence:.2f} < {hi}"
        else:
            lvl = "manual_required"
            reasons[name] = f"confidence {fv.confidence:.2f} < {med}"
        if name in bad:
            worst = "manual_required" if any(i.status == "error" for i in bad[name]) else "review_recommended"
            lvl = _worse(lvl, worst)
            reasons[name] = "; ".join(i.reason for i in bad[name])
        fv.review = lvl
        levels[name] = lvl
    for i, s in enumerate(subjects):
        c = s.row_confidence or 0
        lvl = "auto" if c >= hi else ("review_recommended" if c >= med else "manual_required")
        if "subjects" in bad and any(f"subjects[{i}]" in x.field for x in bad["subjects"]):
            lvl = _worse(lvl, "manual_required")
        if "year header" not in (s.paper.confidence_source or ""):
            lvl = _worse(lvl, "review_recommended")
        s.review = lvl
        levels[f"subjects[{i}]"] = lvl
        if lvl != "auto":
            reasons[f"subjects[{i}]"] = f"row confidence {c:.2f}; paper from {s.paper.confidence_source}"
    flagged = [k for k, v in levels.items() if v != "auto"]
    status = "auto"
    for v in levels.values():
        status = _worse(status, v)
    return ReviewSummary(status=status, fields_requiring_review=flagged,
                         reasons={k: reasons.get(k, "") for k in flagged})


_ORDER = {"auto": 0, "review_recommended": 1, "manual_required": 2}


def _worse(a: str, b: str) -> str:
    return a if _ORDER[a] >= _ORDER[b] else b
