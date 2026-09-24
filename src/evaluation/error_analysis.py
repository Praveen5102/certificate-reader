"""Categorize every evaluation error using the available evidence.

Categories (spec §19):
  ocr_error                 truth value is not (cleanly) present in the OCR text
  layout_error              truth value is present in OCR but nothing was extracted
  field_association_error   a value was extracted from the wrong place
  table_extraction_error    subject row missing / mis-assigned
  normalization_error       right source text, wrong normalized value / span
  truth_annotation_issue    the truth value itself has an open validation issue
  unsupported_document      document rejected by the classifier
Evidence used: alignment report (where the truth value sits in the OCR), open
truth issues, and the prediction's evidence boxes. Categories are
*assignments with evidence*, meant to be spot-checked by a human.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from ..annotation.layout import norm

FIELD_TO_ALIGN = {"student_name": "STUDENT_NAME", "father_name": "FATHER_NAME", "mother_name": "MOTHER_NAME",
                  "hall_ticket_number": "HALL_TICKET", "year_of_pass": "EXAM_SESSION", "month_of_pass": "EXAM_SESSION",
                  "result": "RESULT", "medium": "MEDIUM", "total_marks": "TOTAL_MARKS", "cgpa": "CGPA"}


def _overlap(a, b) -> bool:
    return a and b and not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def categorize_field(field: str, row: dict, full_pred: dict, align: dict | None, flagged: bool) -> tuple[str, str]:
    if not full_pred.get("supported", True):
        return "unsupported_document", "document was not classified as an Intermediate certificate"
    if flagged:
        return "truth_annotation_issue", "truth value has an open validation issue"
    pred = row["pred"]
    status = (align or {}).get("status")
    if field in ("maximum_marks", "percentage"):
        return "truth_annotation_issue" if field == "maximum_marks" else "normalization_error", \
            "derived field; see truth semantics notes"
    if field in ("state", "board", "document_type"):
        return "field_association_error", "header classification"
    if pred in (None, ""):
        if status == "accepted" or (field == "total_marks" and status == "accepted"):
            return "layout_error", "truth value is present in OCR but was not extracted"
        return "ocr_error", f"truth value not cleanly present in OCR (alignment status: {status})"
    fv = full_pred.get("fields", {}).get(field, {})
    ev = [e["bbox"] for e in fv.get("evidence", [])]
    abox = (align or {}).get("bbox")
    same_place = any(_overlap(b, abox) for b in ev) if abox else False
    sim = fuzz.ratio(norm(str(pred)), norm(str(row["truth"])))
    if sim == 100 and str(pred) != str(row["truth"]):
        return "ocr_error", ("differs only in spacing/punctuation (OCR word segmentation, e.g. merged words); "
                             "not repaired because the correct split cannot be known")
    label_value = (align or {}).get("label_value")
    if status in ("review", "not_found") and label_value and             fuzz.ratio(norm(str(pred)), norm(label_value)) >= 90 and sim < 90:
        return "truth_annotation_issue", (f"probable: prediction equals the OCR value printed at the field label "
                                          f"({label_value!r}) and the truth value is not found there "
                                          f"(alignment: {status})")
    if status == "accepted":
        if same_place:
            return "normalization_error", f"prediction taken from the truth's position; similarity {sim:.0f}"
        return "field_association_error", "prediction taken from a different position than the truth value"
    if sim >= 80:
        return "ocr_error", f"prediction is a close OCR variant of truth (similarity {sim:.0f})"
    if status in ("review", "not_found") and (align or {}).get("suggestion") and \
            fuzz.ratio(norm(str(pred)), norm(str(align["suggestion"]))) >= 90:
        return "truth_annotation_issue", ("prediction equals the OCR value at the printed label, which disagrees "
                                          "with the truth (alignment: " + str(status) + ")")
    return "field_association_error", f"prediction differs from truth (similarity {sim:.0f})"


def categorize_subject_row(r: dict, doc_flagged: bool, align_row: dict | None) -> tuple[str, str]:
    if doc_flagged:
        return "truth_annotation_issue", "document's subject truth has an open validation issue"
    if align_row is not None and align_row.get("status") != "accepted":
        return "ocr_error", "truth row could not be located in OCR (value misread or row not printed as annotated)"
    if r["pred"] is None:
        return "table_extraction_error", "truth row present in OCR but no predicted row matched"
    if r["name"] and not (r["marks"] and r["maximum"]):
        return "table_extraction_error", "row found but marks/maximum assigned from wrong cell"
    if not r["paper"]:
        return "table_extraction_error", "paper (year) assignment wrong"
    return "normalization_error", "subject name spelling differs"
