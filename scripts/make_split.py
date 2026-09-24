"""Create the deterministic train/validation/test manifests.

No pre-existing split file was supplied with the dataset, so the split is
generated here with the seed in configs/dataset.yaml. Re-running with the same
inputs reproduces it exactly; the audit hash of the truth file is recorded so a
changed dataset is detectable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import load_yaml, read_json, read_jsonl, resolve, write_json, write_jsonl  # noqa: E402
from src.preprocessing.split import make_split  # noqa: E402

# Documents whose page images were opened by the developer while designing the
# pipeline, before the split existed. Recorded for transparency: if any land in
# test, that is reported (the rule baseline was not tuned on them, but they
# were seen).
INSPECTED_BEFORE_SPLIT = ["certificate_01.pdf", "certificate_04.pdf", "certificate_16.pdf",
                          "certificate_26.pdf", "certificate_35.pdf", "certificate_55.pdf",
                          "certificate_67.pdf", "certificate_70.pdf", "certificate_73.pdf",
                          "certificate_100.pdf"]
# Documents whose OCR *header lines only* (board/title area) were printed while
# debugging the board/state classifier, before the split existed.
OCR_HEADER_INSPECTED_BEFORE_SPLIT = ["certificate_10.pdf", "certificate_11.pdf", "certificate_17.pdf",
                                     "certificate_21.pdf", "certificate_36.pdf", "certificate_52.pdf",
                                     "certificate_53.pdf", "certificate_67.pdf", "certificate_75.pdf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    scfg = cfg["split"]
    if scfg.get("frozen") and not args.force:
        sys.exit("split is frozen in configs/dataset.yaml; pass --force to regenerate (document why)")
    manifest = read_json("data/dataset_manifest.json")["documents"]
    audit = read_json("data/dataset_audit.json")
    truth = {r["file"]: r for r in read_jsonl(cfg["truth_file"])}
    targets = set(cfg["target_document_types"])

    # Household pairs (same parents, different student) are also kept together:
    # identical parent names across partitions would let the model memorize them.
    group_of = {d["file"]: d["student_group"] for d in manifest}
    for p in audit["duplicates"]["same_household_pairs"]:
        ga, gb = group_of[p["a"]], group_of[p["b"]]
        keep, drop = min(ga, gb), max(ga, gb)
        for f, g in group_of.items():
            if g == drop:
                group_of[f] = keep

    eligible, excluded = [], []
    for d in manifest:
        t = truth.get(d["file"])
        if t is None:
            excluded.append({"file": d["file"], "reason": "no truth record"})
            continue
        if t["document_type"] not in targets:
            excluded.append({"file": d["file"], "reason": f"non-target document type {t['document_type']!r}"})
            continue
        state = t["state"] or (d.get("classification") or {}).get("state") or "unknown"
        stratum = f"{state}|{'subjects' if t['subjects'] else 'no_subjects'}"
        eligible.append({"file": d["file"], "group": group_of[d["file"]], "stratum": stratum})

    parts = make_split(eligible, scfg["sizes"], scfg["seed"])
    part_of = {f: k for k, fs in parts.items() for f in fs}

    # Leakage check: every multi-document group must be within one partition,
    # including documents excluded from the split (e.g. an untruthed duplicate).
    leaks = []
    for g in {group_of[f] for f in group_of}:
        members = [f for f in group_of if group_of[f] == g]
        ps = {part_of[f] for f in members if f in part_of}
        if len(ps) > 1:
            leaks.append({"group": members, "partitions": sorted(ps)})
    excluded_linked = [{"file": e["file"], "linked_to": [f for f in group_of if group_of[f] == group_of[e["file"]]
                                                         and f != e["file"]],
                        "partition_of_link": sorted({part_of[f] for f in group_of
                                                     if group_of[f] == group_of[e["file"]] and f in part_of})}
                       for e in excluded]
    excluded_linked = [x for x in excluded_linked if x["linked_to"]]

    strata = {e["file"]: e["stratum"] for e in eligible}
    for k, files in parts.items():
        rows = [{"file": f, "partition": k, "stratum": strata[f], "student_group": group_of[f]} for f in files]
        write_jsonl(resolve(scfg["manifest_dir"]) / k / "manifest.jsonl", rows)
    summary = {
        "seed": scfg["seed"], "sizes": {k: len(v) for k, v in parts.items()},
        "dataset_version": cfg["dataset_version"], "truth_file_sha256": audit["truth_file_sha256"],
        "partitions": parts, "excluded": excluded, "excluded_but_linked_to_split_docs": excluded_linked,
        "strata_counts": {k: _count([strata[f] for f in v]) for k, v in parts.items()},
        "leakage_violations": leaks,
        "inspected_before_split": INSPECTED_BEFORE_SPLIT,
        "inspected_docs_in_test": sorted(set(INSPECTED_BEFORE_SPLIT) & set(parts["test"])),
        "ocr_header_inspected_before_split": OCR_HEADER_INSPECTED_BEFORE_SPLIT,
        "ocr_header_inspected_in_test": sorted(set(OCR_HEADER_INSPECTED_BEFORE_SPLIT) & set(parts["test"])),
        "within_partition_duplicate_groups": _within(parts, group_of),
        "rules": "Test partition must not be used for training, tuning, threshold selection or rule design.",
    }
    write_json(resolve(scfg["manifest_dir"]) / "split.json", summary)
    print({k: len(v) for k, v in parts.items()}, "leaks:", leaks,
          "inspected-in-test:", summary["inspected_docs_in_test"])


def _within(parts, group_of):
    """Same-student groups that sit together inside one partition. Not leakage,
    but they are correlated samples and inflate that partition's effective weight."""
    out = {}
    for k, files in parts.items():
        groups = {}
        for f in files:
            groups.setdefault(group_of[f], []).append(f)
        out[k] = [v for v in groups.values() if len(v) > 1]
    return out


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    main()
