"""Evaluate saved predictions for one partition and write metrics + error report."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ..common.io import doc_id, read_json, read_jsonl, write_json, write_jsonl
from .error_analysis import FIELD_TO_ALIGN, categorize_field, categorize_subject_row
from .metrics import field_metrics, subject_metrics

REPORT_FIELDS = ["student_name", "father_name", "mother_name", "hall_ticket_number", "year_of_pass",
                 "month_of_pass", "course", "group", "medium", "total_marks", "maximum_marks", "percentage",
                 "result", "state", "board", "document_type", "cgpa"]


def evaluate_partition(pred_dir: Path, partition: str, truth_file: str, out_dir: Path) -> dict:
    split = read_json("data/split.json")
    files = split["partitions"][partition]
    truth = {r["file"]: r for r in read_jsonl(truth_file)}
    issues = read_jsonl("annotations/truth_issues.jsonl")
    flagged = {(i["file"], i["field"]) for i in issues if i["severity"] in ("warning", "error")}
    flagged_subject_docs = {i["file"] for i in issues if i["severity"] in ("warning", "error")
                            and (i["field"].startswith("subjects") or i["check"] == "subject_sum_mismatch")}
    align = defaultdict(dict)
    for r in read_jsonl("annotations/alignment_report.jsonl"):
        align[r["file"]][r["field"]] = r

    pairs, fulls = [], {}
    for f in files:
        full = read_json(pred_dir / f"{doc_id(f)}.json")
        fulls[f] = full
        pairs.append((truth[f], full["flat"]))
    fm = field_metrics(pairs, flagged)
    sm = subject_metrics(pairs, flagged_subject_docs)

    errors = []
    for field in REPORT_FIELDS:
        for row in fm[field]["rows"]:
            if row["correct"]:
                continue
            a = align[row["file"]].get(FIELD_TO_ALIGN.get(field, ""), None)
            cat, why = categorize_field(field, row, fulls[row["file"]]["full"], a, row["flagged_truth"])
            errors.append({"file": row["file"], "field": field, "truth": row["truth"], "prediction": row["pred"],
                           "category": cat, "evidence": why,
                           "alignment_status": (a or {}).get("status")})
    subj_align = _subject_alignment(partition)
    for d in sm["per_document"]:
        for k, r in enumerate(d["rows"]):
            if r["complete_row"]:
                continue
            ar = subj_align.get((d["file"], k))
            cat, why = categorize_subject_row(r, d["flagged_truth"], ar)
            errors.append({"file": d["file"], "field": f"subjects[{k}]", "truth": r["truth"], "prediction": r["pred"],
                           "category": cat, "evidence": why, "alignment_status": (ar or {}).get("status")})

    summary = {
        "partition": partition, "documents": len(files),
        "fields": {f: {"all": fm[f]["all"], "clean_truth": fm[f]["clean_truth"]} for f in REPORT_FIELDS},
        "subjects": {"all": sm["all"], "clean_truth": sm["clean_truth"]},
        "error_categories": dict(Counter(e["category"] for e in errors).most_common()),
        "review_routing": dict(Counter(fulls[f]["full"]["review"]["status"] for f in files)),
    }
    write_json(out_dir / f"metrics_{partition}.json", summary)
    write_json(out_dir / f"metrics_{partition}_detail.json",
               {"fields": {f: fm[f]["rows"] for f in REPORT_FIELDS}, "subjects": sm["per_document"]})
    write_jsonl(out_dir / f"error_report_{partition}.jsonl", errors)
    (out_dir / f"error_report_{partition}.md").write_text(_markdown(summary, errors), encoding="utf-8")
    return summary


def _subject_alignment(partition: str) -> dict:
    """Subject-row alignment status from the review queue (rows not accepted)."""
    out = {}
    for q in read_jsonl("annotations/annotation_review_queue.jsonl"):
        if q["kind"] == "subject_row":
            k = int(q["field"][len("subjects["):-1])
            out[(q["file"], k)] = {"status": q["status"]}
    return out


def _markdown(summary: dict, errors: list[dict]) -> str:
    lines = [f"# Error report - {summary['partition']} ({summary['documents']} documents)", "",
             "## Error categories", ""]
    for k, v in summary["error_categories"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Errors", ""]
    for e in sorted(errors, key=lambda e: (e["file"], e["field"])):
        t = e["truth"] if not isinstance(e["truth"], dict) else \
            f"{e['truth']['subject_name']} {e['truth']['paper']} {e['truth']['marks']}/{e['truth']['maximum_marks']}"
        p = e["prediction"]
        if isinstance(p, dict):
            p = f"{p['subject_name']} {p['paper']} {p['marks']}/{p['maximum_marks']}"
        lines += [f"{e['file'].replace('.pdf', '')}", f"field: {e['field']}", f"truth: {t}",
                  f"prediction: {p}", f"error: {e['category']} - {e['evidence']}", ""]
    return "\n".join(lines)
