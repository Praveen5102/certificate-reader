"""Phase 3: align truth to OCR words; write BIO proposals and review queues.

Outputs
    annotations/bio/<doc>.json                 word-level BIO proposals (train + validation only)
    annotations/alignment_report.jsonl         one row per (doc, field) with status + evidence
    annotations/annotation_review_queue.jsonl  items a human must look at
    annotations/truth_suggestions.jsonl        OCR-located values for BLANK truth fields
                                               (proposals only - never merged automatically;
                                               not produced for the test partition)
    annotations/alignment_summary.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation.align import IGNORE, DocAligner, align_document, bio_label_list  # noqa: E402
from src.annotation.anchors import TemplateIndex  # noqa: E402
from src.annotation.layout import load_pages  # noqa: E402
from src.common.io import doc_id, load_yaml, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402

PRIORITY = {"error": "high", "warning": "medium", "info": "low"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--fields-config", default="configs/fields.yaml")
    args = ap.parse_args()
    dcfg = load_yaml(args.dataset_config)
    fcfg = load_yaml(args.fields_config)
    truth = {r["file"]: r for r in read_jsonl(dcfg["truth_file"])}
    manifest = {d["file"]: d for d in read_json("data/dataset_manifest.json")["documents"]}
    split = read_json("data/split.json")
    part_of = {f: k for k, fs in split["partitions"].items() for f in fs}
    issues = read_jsonl("annotations/truth_issues.jsonl")
    issues_by = defaultdict(list)
    for i in issues:
        issues_by[(i["file"], i["field"])].append(i)
    aligner = DocAligner(fcfg, issues_by)

    report, queue, suggestions = [], [], []
    status_counts: dict[str, Counter] = defaultdict(Counter)
    subj_counts: Counter = Counter()
    label_counts: Counter = Counter()
    docs_written = 0
    for file in sorted(part_of, key=lambda f: int("".join(c for c in f if c.isdigit()))):
        part = part_of[file]
        rec = truth[file]
        m = manifest[file]
        ocr = [read_json(p) for p in m["ocr_files"]]
        roles = [r["role"] for r in m["classification"]["page_roles"]]
        pages = load_pages(ocr, roles)
        tmpls = {p.page: TemplateIndex(p, fcfg["template_phrases"]) for p in pages}
        res = align_document(aligner, rec, pages, tmpls)

        for fr in res["fields"]:
            status_counts[fr["field"]][fr["status"]] += 1
            row = {"file": file, "partition": part, **fr}
            if part == "test":
                row["suggestion"] = None   # blind: no OCR/rule suggestions for test truth
            report.append(row)
            if fr["status"] in ("review", "not_found"):
                sev = "high" if fr["status"] == "not_found" else "medium"
                queue.append({"file": file, "partition": part, "kind": "field_alignment", "field": fr["field"],
                              "status": fr["status"], "priority": sev, "truth_value": fr["truth"],
                              # blind review for test: no OCR/rule readings shown
                              "ocr_value": None if part == "test" else (fr["ocr_text"] or None),
                              "suggestion": None if part == "test" else fr["suggestion"],
                              "page": fr["page"], "bbox": fr["bbox"], "reason": fr["reason"],
                              "action": "verify against the page image; if the truth is wrong, record the "
                                        "correction in a new truth version (do not edit the canonical file)",
                              "review_status": "open"})
            elif fr["status"] == "unannotated" and fr["suggestion"] and part != "test":
                suggestions.append({"file": file, "partition": part, "field": fr["field"],
                                    "proposed_value": fr["suggestion"], "page": fr["page"], "bbox": fr["bbox"],
                                    "source": "OCR text right of the printed field label (rule-based)",
                                    "status": "proposed - requires human confirmation before use"})
        for r in res["subjects"]:
            subj_counts[r["status"]] += 1
            if r["status"] != "accepted":
                queue.append({"file": file, "partition": part, "kind": "subject_row", "field": f"subjects[{r['row']}]",
                              "status": r["status"], "priority": "medium", "truth_value": r["truth"],
                              "ocr_value": None if part == "test" else r.get("ocr_text"), "suggestion": None,
                              "page": r.get("page"),
                              "bbox": None, "reason": r["reason"], "review_status": "open",
                              "action": "verify row against the page image"})
        # truth-validation issues (error/warning) are review items too
        for i in issues:
            if i["file"] == file and i["severity"] in ("error", "warning"):
                queue.append({"file": file, "partition": part, "kind": "truth_issue", "field": i["field"],
                              "status": i["check"], "priority": PRIORITY[i["severity"]],
                              "truth_value": i["truth_value"], "ocr_value": None, "suggestion": None,
                              "page": None, "bbox": None, "reason": i["detail"], "review_status": "open",
                              "action": "confirm or correct in a new truth version"})

        if part in ("train", "validation"):
            labels = res["word_labels"]
            for v in labels.values():
                label_counts[v] += 1
            out_pages = []
            for p in pages:
                out_pages.append({"page": p.page, "width": p.width, "height": p.height, "role": p.role,
                                  "words": [{"id": w.idx, "text": w.text, "bbox": list(w.bbox),
                                             "ocr_confidence": w.conf, "label": labels[w.uid]}
                                            for w in p.words]})
            write_json(f"annotations/bio/{doc_id(file)}.json",
                       {"file": file, "partition": part, "label_scheme": "BIO (word level); IGNORE = no loss",
                        "table_masked": res["table_masked"], "subjects_complete": res["subjects_complete"],
                        "pages": out_pages})
            docs_written += 1

    # a field can appear twice (truth_issue + field_alignment): both are kept, they need different checks
    for n, q in enumerate(queue):
        q["id"] = f"R{n + 1:04d}"
    write_jsonl("annotations/alignment_report.jsonl", report)
    write_jsonl("annotations/annotation_review_queue.jsonl", queue)
    write_jsonl("annotations/truth_suggestions.jsonl", suggestions)
    summary = {
        "documents_aligned": len(part_of),
        "bio_documents_written": docs_written,
        "label_set": bio_label_list(),
        "field_status_counts": {k: dict(v) for k, v in status_counts.items()},
        "subject_row_status_counts": dict(subj_counts),
        "word_label_counts_train_val": dict(label_counts.most_common()),
        "review_queue": {"total": len(queue),
                         "by_kind": dict(Counter(q["kind"] for q in queue)),
                         "by_priority": dict(Counter(q["priority"] for q in queue)),
                         "by_partition": dict(Counter(q["partition"] for q in queue))},
        "truth_suggestions": len(suggestions),
        "ignore_label": IGNORE,
    }
    write_json("annotations/alignment_summary.json", summary)
    import json
    print(json.dumps({k: summary[k] for k in ("field_status_counts", "subject_row_status_counts", "review_queue",
                                              "truth_suggestions")}, indent=1))


if __name__ == "__main__":
    main()
