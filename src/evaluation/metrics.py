"""Field-level and subject-table metrics against the truth file.

Conventions (see README "Evaluation protocol"):
  * A blank truth value means UNANNOTATED: the document is excluded from that
    field's metric (reported as coverage), never scored as "should be empty".
  * Every field is reported twice: over all annotated documents, and over the
    "clean" subset excluding documents whose truth value for that field has an
    open warning/error in annotations/truth_issues.jsonl.
  * Small-n: each accuracy carries a Wilson 95% interval.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

from rapidfuzz.distance import Levenshtein

from ..preprocessing.classify import board_id
from ..validation.normalize import collapse_ws, normalize_medium, normalize_result


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _name(v) -> str:
    return collapse_ws(str(v or ""))


def _digits(v) -> str:
    return re.sub(r"\D", "", str(v or ""))


def _canon_result(v) -> str:
    return normalize_result(str(v or ""))[0]


def _canon_medium(v) -> str:
    return normalize_medium(str(v or ""))[0]


def _doc_type(v) -> str:
    s = str(v or "").lower()
    return "intermediate" if "intermediate" in s else ("other" if s else "")


def _pct(v) -> float | None:
    return None if v is None else round(float(v), 1)


# field -> (comparison key fn, is_text)
FIELD_SPECS: dict[str, tuple[Callable[[Any], Any], bool]] = {
    "document_type": (_doc_type, False),
    "state": (lambda v: str(v or "").strip(), False),
    "board": (lambda v: board_id(str(v or "")), False),
    "student_name": (_name, True),
    "father_name": (_name, True),
    "mother_name": (_name, True),
    "hall_ticket_number": (_digits, True),
    "year_of_pass": (lambda v: None if v is None else int(v), False),
    "month_of_pass": (lambda v: str(v or "").strip().upper(), False),
    "course": (_name, True),
    "group": (lambda v: str(v or "").strip().upper(), False),
    "medium": (_canon_medium, False),
    "total_marks": (lambda v: None if v is None else int(v), False),
    "maximum_marks": (lambda v: None if v is None else int(v), False),
    "percentage": (_pct, False),
    "result": (_canon_result, False),
    "cgpa": (lambda v: None if v is None else round(float(v), 2), False),
}


def field_metrics(pairs: list[tuple[dict, dict]], flagged: set[tuple[str, str]]) -> dict[str, dict]:
    """pairs: [(truth_record, predicted_flat)]; flagged: {(file, field)} with open truth issues."""
    out = {}
    for field, (key, is_text) in FIELD_SPECS.items():
        rows = []
        for t, p in pairs:
            tv = t.get(field)
            if _blank(tv):
                continue
            pv = p.get(field)
            tk, pk = key(tv), (key(pv) if not _blank(pv) else None)
            rec = {"file": t["file"], "truth": tv, "pred": pv, "correct": pk is not None and pk == tk,
                   "flagged_truth": (t["file"], field) in flagged}
            if is_text:
                ts, ps = str(tk), str(pk or "")
                rec["cer"] = Levenshtein.distance(ps, ts) / max(1, len(ts))
                rec["similarity"] = Levenshtein.normalized_similarity(ps, ts)
                # secondary: ignoring spaces/punctuation (OCR word-segmentation differences)
                alnum = lambda x: re.sub(r"[^A-Z0-9]", "", x.upper())  # noqa: E731
                rec["correct_ignoring_spacing"] = pk is not None and alnum(ps) == alnum(ts)
            if field == "total_marks" and pk is not None:
                rec["abs_error"] = abs(pk - tk)
            rows.append(rec)
        out[field] = {"all": _summ(rows, is_text, field), "clean_truth": _summ([r for r in rows if not r["flagged_truth"]],
                                                                               is_text, field),
                      "documents_in_partition": len(pairs), "rows": rows}
    return out


def _summ(rows: list[dict], is_text: bool, field: str) -> dict:
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    s = {"n_annotated": n, "n_predicted": sum(r["pred"] not in (None, "") for r in rows), "exact_match": k,
         "accuracy": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}
    if is_text and n:
        k2 = sum(r["correct_ignoring_spacing"] for r in rows)
        s["accuracy_ignoring_spacing"] = round(k2 / n, 4)
        s["mean_cer"] = round(sum(r["cer"] for r in rows) / n, 4)
        s["mean_similarity"] = round(sum(r["similarity"] for r in rows) / n, 4)
    if field == "total_marks":
        errs = [r["abs_error"] for r in rows if "abs_error" in r]
        s["mae_when_predicted"] = round(sum(errs) / len(errs), 2) if errs else None
    return s


# ------------------------------------------------------------------------------ subjects
def _sname(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper()).replace("PRACTICALS", "PRACTICAL")


def match_subject_rows(truth_rows: list[dict], pred_rows: list[dict]) -> list[tuple[int, int | None]]:
    """Greedy one-to-one assignment truth->pred maximizing name similarity,
    then paper agreement, then marks agreement."""
    cands = []
    for i, t in enumerate(truth_rows):
        for j, p in enumerate(pred_rows):
            sim = Levenshtein.normalized_similarity(_sname(t["subject_name"]), _sname(p.get("subject_name")))
            if sim < 0.6:
                continue
            score = sim * 100 + (10 if t["paper"] == p.get("paper") else 0) + (5 if t["marks"] == p.get("marks") else 0)
            cands.append((score, i, j))
    cands.sort(reverse=True)
    used_t, used_p, pairs = set(), set(), {}
    for _, i, j in cands:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        pairs[i] = j
    return [(i, pairs.get(i)) for i in range(len(truth_rows))]


def subject_metrics(pairs: list[tuple[dict, dict]], flagged_docs: set[str]) -> dict:
    agg = {"all": _new_subj(), "clean_truth": _new_subj()}
    per_doc = []
    for t, p in pairs:
        tr = t.get("subjects") or []
        if not tr:
            continue
        pr = p.get("subjects") or []
        m = match_subject_rows(tr, pr)
        d = _new_subj()
        rows = []
        for i, j in m:
            trow, prow = tr[i], (pr[j] if j is not None else None)
            name_ok = prow is not None and _sname(prow["subject_name"]) == _sname(trow["subject_name"])
            paper_ok = prow is not None and prow.get("paper") == trow["paper"]
            marks_ok = prow is not None and prow.get("marks") == trow["marks"]
            max_ok = prow is not None and prow.get("maximum_marks") == trow["maximum_marks"]
            full = name_ok and paper_ok and marks_ok and max_ok
            for k, v in (("name", name_ok), ("paper", paper_ok), ("marks", marks_ok), ("maximum", max_ok),
                         ("complete_row", full)):
                d[k] += int(v)
            d["truth_rows"] += 1
            rows.append({"truth": trow, "pred": prow, "name": name_ok, "paper": paper_ok, "marks": marks_ok,
                         "maximum": max_ok, "complete_row": full})
        d["pred_rows"] += len(pr)
        d["matched_pred_rows"] += sum(1 for _, j in m if j is not None)
        d["docs"] = 1
        d["docs_all_rows_exact"] = int(all(r["complete_row"] for r in rows) and len(pr) == len(tr))
        for bucket in (["all"] + ([] if t["file"] in flagged_docs else ["clean_truth"])):
            for k in d:
                agg[bucket][k] += d[k]
        per_doc.append({"file": t["file"], "flagged_truth": t["file"] in flagged_docs, **{k: d[k] for k in d},
                        "rows": rows})
    return {b: _subj_rates(v) for b, v in agg.items()} | {"per_document": per_doc}


def _new_subj() -> dict:
    return {"truth_rows": 0, "pred_rows": 0, "matched_pred_rows": 0, "name": 0, "paper": 0, "marks": 0,
            "maximum": 0, "complete_row": 0, "docs": 0, "docs_all_rows_exact": 0}


def _subj_rates(d: dict) -> dict:
    n = d["truth_rows"]
    out = dict(d)
    for k in ("name", "paper", "marks", "maximum", "complete_row"):
        out[f"{k}_accuracy"] = round(d[k] / n, 4) if n else None
        out[f"{k}_wilson95"] = wilson(d[k], n)
    out["row_precision"] = round(d["matched_pred_rows"] / d["pred_rows"], 4) if d["pred_rows"] else None
    return out
