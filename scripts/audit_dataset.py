"""Phase 1: dataset verification.

Writes (never touches the truth file or the raw PDFs):
    data/dataset_audit.json       - findings, counts, duplicate groups
    data/dataset_manifest.json    - one entry per PDF
    annotations/truth_issues.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402

from src.annotation.truth_validation import validate_all  # noqa: E402
from src.common.io import (load_yaml, read_json, read_jsonl, resolve, sha256_file,  # noqa: E402
                           write_json, write_jsonl)
from src.preprocessing.audit import (group_students, phash, registration_numbers,  # noqa: E402
                                     same_student_pairs, summarize_groups)
from src.preprocessing.classify import board_id, classify_document  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--ocr-config", default="configs/ocr.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ocr_dir = resolve(load_yaml(args.ocr_config)["output_dir"])
    raw = resolve(cfg["raw_pdf_dir"])
    truth_path = resolve(cfg["truth_file"])
    truth = read_jsonl(truth_path)
    truth_by_file = {r["file"]: r for r in truth}
    page_index = read_json(resolve(cfg["rendering"]["output_dir"]) / "page_index.json")

    pdfs = sorted(raw.glob("*.pdf"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    manifest, sha_by, reg_by, phash_by = [], {}, {}, {}
    for pdf in pdfs:
        entry = {"file": pdf.name, "path": pdf.relative_to(resolve(".")).as_posix(),
                 "sha256": sha256_file(pdf), "bytes": pdf.stat().st_size}
        with fitz.open(pdf) as doc:
            entry["pages"] = doc.page_count
            meta = doc.metadata or {}
            entry["producer"] = " | ".join(v for v in (meta.get("creator"), meta.get("producer")) if v)
        pages = page_index.get(pdf.name, [])
        entry["page_images"] = [p["path"] for p in pages]
        entry["embedded_text_layer"] = any(p["has_text_layer"] for p in pages)
        ocr_pages = []
        for p in pages:
            op = ocr_dir / f"{pdf.stem}_p{p['page']}.json"
            if op.exists():
                ocr_pages.append(read_json(op))
        entry["ocr_files"] = [(ocr_dir / f"{pdf.stem}_p{p['page']}.json").relative_to(resolve(".")).as_posix()
                              for p in pages]
        entry["ocr_complete"] = len(ocr_pages) == len(pages) and len(pages) > 0
        if ocr_pages:
            confs = [w["confidence"] for p in ocr_pages for w in p["words"]]
            entry["ocr_stats"] = {
                "words": len(confs),
                "mean_word_confidence": round(sum(confs) / len(confs), 4) if confs else None,
                "low_conf_words_lt_0.8": sum(c < 0.8 for c in confs),
                "rotations": [p["rotation"] for p in ocr_pages],
            }
            cls = classify_document([[l["text"] for l in p["lines"]] for p in ocr_pages])
            entry["classification"] = cls
            reg_by[pdf.name] = registration_numbers(
                [p for p, r in zip(ocr_pages, cls["page_roles"]) if r["role"] != "notes"])
            entry["ocr_registration_numbers"] = reg_by[pdf.name]
        if pages:
            phash_by[pdf.name] = phash(resolve(pages[0]["path"]))
        sha_by[pdf.name] = entry["sha256"]
        t = truth_by_file.get(pdf.name)
        entry["in_truth"] = t is not None
        entry["truth_document_type"] = t["document_type"] if t else None
        manifest.append(entry)

    # ---- truth checks ------------------------------------------------------------------
    issues = validate_all(truth)
    pdf_names = {p.name for p in pdfs}
    for r in truth:
        if r["file"] not in pdf_names:
            issues.append({"file": r["file"], "field": "file", "check": "truth_without_pdf",
                           "severity": "error", "truth_value": r["file"], "detail": "no PDF with this name"})
    cls_map = {e["file"]: e.get("classification", {}) for e in manifest}
    for r in truth:
        c = cls_map.get(r["file"]) or {}
        if not c:
            continue
        # Truth vs OCR-evidence disagreements: flagged, never "fixed".
        t_is_inter = r["document_type"] == "Intermediate certificate"
        if t_is_inter != (c["document_type"] == "intermediate_certificate"):
            issues.append({"file": r["file"], "field": "document_type", "check": "doc_type_disagrees_with_ocr",
                           "severity": "warning", "truth_value": r["document_type"],
                           "detail": f"OCR evidence classifies as {c['document_type']} ({c['evidence']})"})
        if r["state"] and c["state"] and r["state"] != c["state"]:
            issues.append({"file": r["file"], "field": "state", "check": "state_disagrees_with_ocr",
                           "severity": "warning", "truth_value": r["state"],
                           "detail": f"OCR evidence says {c['state']} ({c['state_evidence']})"})
        if r["board"] and c["board"] and board_id(r["board"]) != board_id(c["board"]):
            issues.append({"file": r["file"], "field": "board", "check": "board_disagrees_with_ocr",
                           "severity": "warning", "truth_value": r["board"],
                           "detail": f"OCR header reads {c['board']!r} (board id {board_id(c['board'])} "
                                     f"vs truth {board_id(r['board'])})"})
    for i in issues:
        i["status"] = "open"
    write_jsonl("annotations/truth_issues.jsonl", issues)

    # ---- duplicates / leakage groups ------------------------------------------------
    dcfg = cfg["duplicates"]
    pairs = same_student_pairs(truth_by_file, reg_by, phash_by, sha_by,
                               dcfg["phash_near_dup_threshold"], dcfg["name_similarity_threshold"])
    files = [e["file"] for e in manifest]
    group_of = group_students(files, pairs)
    for e in manifest:
        e["student_group"] = group_of[e["file"]]
    household_pairs = [p for p in pairs if p["relation"] == "same_parents_different_student"]

    # Same student in two truth records: non-empty values that disagree are
    # annotation conflicts (one of them is wrong, or the documents differ).
    conflict_fields = ["student_name", "father_name", "mother_name", "year_of_pass", "month_of_pass",
                       "total_marks", "result", "medium"]
    for p in pairs:
        if p["relation"] != "same_student" or p["a"] not in truth_by_file or p["b"] not in truth_by_file:
            continue
        ta, tb = truth_by_file[p["a"]], truth_by_file[p["b"]]
        identical = any("identical" in r for r in p["reasons"])
        for f in conflict_fields:
            va, vb = ta.get(f), tb.get(f)
            if va not in (None, "") and vb not in (None, "") and va != vb:
                issues.append({"file": p["a"], "field": f, "check": "duplicate_truth_conflict",
                               "severity": "error" if identical else "warning", "truth_value": va,
                               "detail": f"{p['b']} (same student; {'identical image' if identical else 'same reg. no.'})"
                                         f" has {f}={vb!r}", "status": "open"})
    write_jsonl("annotations/truth_issues.jsonl", issues)

    # ---- summary ---------------------------------------------------------------------
    truth_files = set(truth_by_file)
    sev = Counter(i["severity"] for i in issues)
    by_check = Counter(f"{i['severity']}:{i['check']}" for i in issues)
    fields_missing = Counter(i["field"] for i in issues if i["check"] == "missing_value")
    audit = {
        "truth_file": cfg["truth_file"],
        "truth_file_sha256": sha256_file(truth_path),
        "dataset_version": cfg["dataset_version"],
        "counts": {
            "pdf_files": len(pdfs),
            "truth_records": len(truth),
            "pdfs_with_truth": len(pdf_names & truth_files),
            "pdfs_without_truth": sorted(pdf_names - truth_files),
            "truth_without_pdf": sorted(truth_files - pdf_names),
            "total_pages": sum(e["pages"] for e in manifest),
            "page_count_distribution": dict(Counter(e["pages"] for e in manifest)),
            "multi_page_documents": [e["file"] for e in manifest if e["pages"] > 1],
            "pdfs_with_embedded_text_layer": [e["file"] for e in manifest if e["embedded_text_layer"]],
            "ocr_complete": sum(e["ocr_complete"] for e in manifest),
        },
        "document_types": {
            "truth": dict(Counter(r["document_type"] for r in truth)),
            "ocr_classifier": dict(Counter((e.get("classification") or {}).get("document_type", "not_ocred")
                                           for e in manifest)),
            "ocr_subtypes": dict(Counter((e.get("classification") or {}).get("document_subtype", "")
                                         for e in manifest)),
            "ocr_state": dict(Counter((e.get("classification") or {}).get("state", "") or "undetermined"
                                      for e in manifest)),
            "ocr_board": dict(Counter((e.get("classification") or {}).get("board", "") or "undetermined"
                                      for e in manifest)),
            "non_target_documents": [e["file"] for e in manifest
                                     if (e.get("truth_document_type") not in (None, "Intermediate certificate"))
                                     or (e.get("classification") or {}).get("document_type")
                                     not in (None, "intermediate_certificate")],
        },
        "duplicates": {
            "exact_file_duplicates": [p for p in pairs if any("identical file bytes" in r for r in p["reasons"])],
            "same_student_pairs": [p for p in pairs if p["relation"] == "same_student"],
            "same_student_groups": summarize_groups(group_of),
            "same_household_pairs": household_pairs,
        },
        "truth_quality": {
            "issues_by_severity": dict(sev),
            "issues_by_check": dict(sorted(by_check.items())),
            "missing_value_counts_by_field": dict(fields_missing.most_common()),
            "records_with_subjects": sum(1 for r in truth if r["subjects"]),
            "records_without_subjects": sum(1 for r in truth if not r["subjects"]),
            "hall_ticket_annotated": sum(1 for r in truth if r["hall_ticket_number"]),
            "systematic_findings": [
                "maximum_marks holds a per-subject maximum (100 or 60) in every record where it is set; it is "
                "never >= total_marks, so it does not represent the grand maximum.",
                "percentage is never printed on TS/AP certificates; where set it equals 100*total/sum(subject "
                "maxima) (a derived value) except in the records flagged percentage_inconsistent.",
                "Blank truth values co-occur with documents whose fields are clearly printed (e.g. fully "
                "blank records for readable certificates), so blank must mean 'unannotated', not 'absent'.",
                "Several person-name and result values contain OCR residue from the certificate template "
                "('THIS IS TO CERTIFY THAT', 'COPA' for 'CGPA').",
                "course and group are blank in 97/98 records; hall_ticket_number is set in 1/98.",
            ],
        },
    }
    write_json("data/dataset_audit.json", audit)
    write_json("data/dataset_manifest.json", {"dataset_version": cfg["dataset_version"], "documents": manifest})
    print(f"PDFs={len(pdfs)} truth={len(truth)} pages={audit['counts']['total_pages']} "
          f"issues={dict(sev)} groups={audit['duplicates']['same_student_groups']}")


if __name__ == "__main__":
    main()
