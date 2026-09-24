"""Pydantic output schema.

Two views of one result:
  * `ExtractionResult` - the full, traceable form: every field carries its value,
    confidence, the confidence source, the OCR evidence it came from, and
    review status.
  * `ExtractionResult.to_flat()` - the plain JSON schema requested in the spec
    (values only), for downstream consumers that do not need provenance.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ReviewLevel = Literal["auto", "review_recommended", "manual_required"]

FLAT_FIELDS = [
    "document_type", "state", "board", "student_name", "father_name", "mother_name",
    "hall_ticket_number", "year_of_pass", "month_of_pass", "course", "group", "medium",
    "total_marks", "maximum_marks", "percentage", "result",
]
# Proposed additions (see README "Schema proposals"); present in the full view only.
EXTRA_FIELDS = ["cgpa", "document_subtype"]


class Evidence(BaseModel):
    page: int
    text: str
    bbox: list[int] = Field(description="[x0, y0, x1, y1] in page-image pixels")
    ocr_confidence: float | None = None


class FieldValue(BaseModel):
    value: Any = None
    raw: str | None = Field(None, description="text as read from the document, before normalization")
    confidence: float | None = Field(None, description="see confidence_source; None when not computable")
    confidence_source: str = ""
    normalization: str = ""
    derived: bool = Field(False, description="computed from other fields rather than read from the page")
    evidence: list[Evidence] = Field(default_factory=list)
    review: ReviewLevel = "manual_required"


class SubjectRow(BaseModel):
    subject_name: FieldValue
    paper: FieldValue
    marks: FieldValue
    maximum_marks: FieldValue
    row_confidence: float | None = None
    review: ReviewLevel = "manual_required"


class ValidationIssue(BaseModel):
    field: str
    check: str
    status: Literal["warning", "error", "info"]
    reason: str


class ReviewSummary(BaseModel):
    status: ReviewLevel
    fields_requiring_review: list[str] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    file: str
    document_type: str
    document_type_score: float | None = None
    document_type_score_kind: str = ""
    supported: bool = True
    pages: int = 1
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    subjects: list[SubjectRow] = Field(default_factory=list)
    validation: list[ValidationIssue] = Field(default_factory=list)
    review: ReviewSummary
    notes: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)

    def to_flat(self) -> dict[str, Any]:
        out: dict[str, Any] = {"file": self.file, "document_type": self.document_type}
        for f in FLAT_FIELDS:
            if f == "document_type":
                continue
            fv = self.fields.get(f)
            out[f] = fv.value if fv is not None else (None if f in {"year_of_pass", "total_marks",
                                                                     "maximum_marks", "percentage"} else "")
        out["subjects"] = [{"subject_name": s.subject_name.value, "paper": s.paper.value,
                            "marks": s.marks.value, "maximum_marks": s.maximum_marks.value}
                           for s in self.subjects]
        order = ["file", "document_type"] + [f for f in FLAT_FIELDS if f != "document_type"]
        flat = {k: out.get(k) for k in order}
        flat["subjects"] = out["subjects"]
        # keep spec key order: subjects sits between medium and total_marks
        keys = order[:order.index("medium") + 1] + ["subjects"] + order[order.index("medium") + 1:]
        flat = {k: flat[k] for k in keys}
        # cgpa is not in the spec schema but IS a column of the truth file (AP grade
        # memos print CGPA instead of a total); emitted so it can be evaluated.
        cg = self.fields.get("cgpa")
        flat["cgpa"] = cg.value if cg is not None else None
        flat["notes"] = self.notes
        return flat
