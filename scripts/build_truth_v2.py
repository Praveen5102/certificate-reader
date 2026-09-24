"""Build truth v2 from the canonical truth + per-certificate review decisions.

Inputs (read-only):
    ground_truth_98_clean.jsonl                 canonical v1 truth (never modified)
    annotations/truth_v2/_drafts.json           per-field draft: v1 value where the OCR confirms it,
                                                otherwise the value read at the printed label
    annotations/truth_v2/decisions.jsonl        one reviewed decision per certificate (read from the
                                                page image): field overrides, subject rows, notes
Outputs:
    annotations/truth_v2/ground_truth_v2.jsonl
    annotations/truth_v2/changelog.jsonl        every field that differs from v1, with the reason
    annotations/truth_v2/verification.json      internal consistency checks on v2

Conventions (agreed with the project owner on 2026-09-23):
    result         = the value printed after "passed in" / "passed with" (grade, division,
                     COMPARTMENTALLY, ...); AP CGPA memos -> "PASSED" plus the cgpa field
    maximum_marks  = the grand maximum = sum of subject maxima (None when a subject maximum
                     is not printed, e.g. AP Open School marks without a max column)
    percentage     = 100 * total / maximum_marks, rounded to 1 decimal (derived)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.inference.postprocess import GROUPS, _subject_key  # noqa: E402

V2 = Path("annotations/truth_v2")
FIELDS = ["file", "document_type", "state", "board", "student_name", "father_name", "mother_name",
          "hall_ticket_number", "year_of_pass", "month_of_pass", "course", "group", "medium", "subjects",
          "total_marks", "maximum_marks", "percentage", "result", "cgpa"]


def main() -> None:
    v1 = {r["file"]: r for r in read_jsonl("ground_truth_98_clean.jsonl")}
    drafts = read_json(V2 / "_drafts.json")
    decisions = {}
    for d in read_jsonl(V2 / "decisions.jsonl"):
        f = f"certificate_{d['n']}.pdf"
        if f in decisions:
            raise SystemExit(f"duplicate decision for {f}")
        decisions[f] = d
    missing = sorted(set(v1) - set(decisions))
    if missing:
        raise SystemExit(f"no review decision for: {missing}")

    out: dict[str, dict] = {}
    # two passes so "subjects_from" can reference an already-built record
    order = sorted(v1, key=lambda f: ("subjects_from" in decisions[f], f))
    for f in order:
        d, base = decisions[f], drafts[f]["draft"]
        rec = {k: base.get(k, v1[f].get(k)) for k in FIELDS if k not in ("file", "subjects")}
        rec["file"] = f
        rec["document_type"] = v1[f]["document_type"]
        subjects = [dict(s) for s in base["subjects"]]
        if "subjects" in d:
            subjects = [{"subject_name": n, "paper": p, "marks": m, "maximum_marks": mx}
                        for n, p, m, mx in d["subjects"]]
        if "subjects_from" in d:
            src = f"certificate_{d['subjects_from']}.pdf"
            subjects = [dict(s) for s in out[src]["subjects"]]
        for i, name in (d.get("rename") or {}).items():
            subjects[int(i)]["subject_name"] = name
        for i, upd in (d.get("subj") or {}).items():
            subjects[int(i)].update(upd)
        for n, p, m, mx in d.get("add_subjects") or []:
            subjects.append({"subject_name": n, "paper": p, "marks": m, "maximum_marks": mx})
        for k, v in (d.get("subj_all") or {}).items():
            for s in subjects:
                s[k] = v
        rec["subjects"] = subjects
        rec.update(d.get("set") or {})
        if rec.get("result"):
            rec["result"] = rec["result"].strip().rstrip(".").strip()
        # derived fields (convention above)
        maxes = [s["maximum_marks"] for s in subjects]
        rec["maximum_marks"] = sum(maxes) if subjects and all(isinstance(x, int) for x in maxes) else None
        if f in ("certificate_88.pdf",):
            rec["maximum_marks"] = 500          # printed grand maximum (open-school memo)
        # group is not printed on TS/AP certificates: derived from the subject set
        # (exact match only), same rule as src/inference/postprocess.py
        keys = {_subject_key(x["subject_name"]) for x in subjects}
        rec["group"] = next((g for g, need in GROUPS.items() if need <= keys), "")
        t, gm = rec.get("total_marks"), rec["maximum_marks"]
        rec["percentage"] = round(100.0 * t / gm, 1) if isinstance(t, int) and gm else None
        rec["review_note"] = d.get("note", "")
        out[f] = {k: rec.get(k) for k in FIELDS} | {"review_note": rec["review_note"]}

    records = [out[f] for f in sorted(out, key=lambda x: int("".join(c for c in x if c.isdigit())))]
    # the truth file keeps exactly the v1 schema; review notes live in their own file
    write_jsonl(V2 / "ground_truth_v2.jsonl", [{k: r[k] for k in FIELDS} for r in records])
    write_jsonl(V2 / "review_notes.jsonl", [{"file": r["file"], "note": r["review_note"]} for r in records])

    # ---- changelog vs v1 ---------------------------------------------------------------
    changes = []
    for r in records:
        o = v1[r["file"]]
        for k in FIELDS:
            if k == "file":
                continue
            a, b = o.get(k), r.get(k)
            if a != b:
                kind = "filled" if a in (None, "", []) else ("cleared" if b in (None, "", []) else "corrected")
                if k in ("maximum_marks", "percentage"):
                    kind = "derived (convention change)" if a not in (None, "") else "derived"
                changes.append({"file": r["file"], "field": k, "kind": kind,
                                "v1": a if k != "subjects" else f"{len(a)} rows",
                                "v2": b if k != "subjects" else f"{len(b)} rows",
                                "note": r["review_note"]})
    write_jsonl(V2 / "changelog.jsonl", changes)

    # ---- verification ------------------------------------------------------------------
    problems, sums_ok, sums_checked = [], 0, 0
    for r in records:
        if r["document_type"] != "Intermediate certificate":
            continue
        subj = r["subjects"]
        if subj and isinstance(r["total_marks"], int):
            sums_checked += 1
            s = sum(x["marks"] for x in subj)
            if s == r["total_marks"]:
                sums_ok += 1
            else:
                problems.append({"file": r["file"], "check": "subject_sum", "detail": f"{s} != {r['total_marks']}"})
        for x in subj:
            if isinstance(x["maximum_marks"], int) and x["marks"] > x["maximum_marks"]:
                problems.append({"file": r["file"], "check": "marks_gt_max", "detail": str(x)})
        for k in ("student_name", "total_marks", "year_of_pass", "result"):
            if r.get(k) in (None, ""):
                problems.append({"file": r["file"], "check": "missing", "detail": k})
    coverage = {k: sum(1 for r in records if r.get(k) not in (None, "", [])) for k in FIELDS if k != "file"}
    coverage_v1 = {k: sum(1 for r in v1.values() if r.get(k) not in (None, "", [])) for k in FIELDS if k != "file"}
    kinds = {}
    for c in changes:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    ver = {"records": len(records), "subject_sum_checked": sums_checked, "subject_sum_matches_total": sums_ok,
           "problems": problems, "coverage_v2": coverage, "coverage_v1": coverage_v1,
           "changes": len(changes), "changes_by_kind": kinds}
    write_json(V2 / "verification.json", ver)
    print(json.dumps({k: ver[k] for k in ("records", "subject_sum_checked", "subject_sum_matches_total",
                                          "changes", "changes_by_kind")}, indent=1))
    print("problems:", json.dumps(problems, indent=1))
    print("coverage v1 -> v2:", {k: f"{coverage_v1[k]}->{coverage[k]}" for k in coverage})


if __name__ == "__main__":
    main()
