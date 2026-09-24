import copy
import json

import pytest

from src.annotation.truth_validation import validate_record
from src.common.io import PROJECT_ROOT, read_json, sha256_file
from src.preprocessing.split import make_split

BASE = {"file": "x.pdf", "document_type": "Intermediate certificate", "state": "Telangana", "board": "B",
        "student_name": "A B", "father_name": "C D", "mother_name": "E F", "hall_ticket_number": "",
        "year_of_pass": 2020, "month_of_pass": "MARCH", "course": "", "group": "", "medium": "",
        "subjects": [{"subject_name": "ENGLISH", "paper": "I", "marks": 50, "maximum_marks": 100},
                     {"subject_name": "ENGLISH", "paper": "II", "marks": 60, "maximum_marks": 100}],
        "total_marks": 111, "maximum_marks": 100, "percentage": None, "result": "PASSED", "cgpa": None}


def test_validation_flags_and_never_mutates():
    rec = copy.deepcopy(BASE)
    rec["student_name"] = "THIS TO CERTY TAT SOME NAME"
    before = json.dumps(rec, sort_keys=True)
    issues = validate_record(rec)
    assert json.dumps(rec, sort_keys=True) == before      # truth is never changed
    checks = {i["check"] for i in issues}
    assert "subject_sum_mismatch" in checks               # 110 != 111
    assert "maximum_below_total" in checks
    assert "name_ocr_residue" in checks
    assert "missing_value" in checks                      # blank hall ticket = unannotated


needs_data = pytest.mark.skipif(not (PROJECT_ROOT / "data" / "split.json").exists(),
                                reason="private dataset not present (it is never committed)")


@needs_data
def test_truth_file_unchanged_since_audit():
    audit = read_json(PROJECT_ROOT / "data" / "dataset_audit.json")
    assert sha256_file(PROJECT_ROOT / audit["truth_file"]) == audit["truth_file_sha256"]


def test_split_keeps_groups_together_and_is_deterministic():
    docs = [{"file": f"d{i}.pdf", "group": f"g{i // 2}" if i < 6 else f"g{i}", "stratum": "s" + str(i % 2)}
            for i in range(20)]
    a = make_split(docs, {"train": 14, "validation": 3, "test": 3}, seed=1)
    b = make_split(docs, {"train": 14, "validation": 3, "test": 3}, seed=1)
    assert a == b
    part_of = {f: k for k, fs in a.items() for f in fs}
    for g in {d["group"] for d in docs}:
        assert len({part_of[d["file"]] for d in docs if d["group"] == g}) == 1
    assert {k: len(v) for k, v in a.items()} == {"train": 14, "validation": 3, "test": 3}


@needs_data
def test_saved_split_has_no_leakage():
    s = read_json(PROJECT_ROOT / "data" / "split.json")
    assert s["leakage_violations"] == []
    assert sum(s["sizes"].values()) == 97
    assert not set(s["partitions"]["test"]) & set(s["partitions"]["train"])


V1_SHA256 = "83e50b0e64e2858787de5ba71df77f2222bdd0de496a782d45fc85eef6d8a0ea"


@needs_data
def test_canonical_v1_truth_never_modified():
    assert sha256_file(PROJECT_ROOT / "ground_truth_98_clean.jsonl") == V1_SHA256


@needs_data
def test_truth_v2_is_consistent():
    from src.common.io import read_jsonl
    v1 = read_jsonl(PROJECT_ROOT / "ground_truth_98_clean.jsonl")
    v2 = read_jsonl(PROJECT_ROOT / "annotations" / "truth_v2" / "ground_truth_v2.jsonl")
    assert [r["file"] for r in v2] == [r["file"] for r in v1]
    assert [list(r) for r in v2] == [list(r) for r in v1]          # same schema / key order
    for r in v2:
        if r["document_type"] == "Intermediate certificate" and r["subjects"]:
            assert sum(s["marks"] for s in r["subjects"]) == r["total_marks"], r["file"]
