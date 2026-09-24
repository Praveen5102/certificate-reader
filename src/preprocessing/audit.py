"""Dataset audit: PDFs vs truth, duplicates, same-student groups, doc types."""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image
from rapidfuzz import fuzz

# TS/AP Intermediate registration (hall-ticket) numbers are 10 digits and, for
# the years present here, start with 0-2 (year prefix). Phone numbers (start
# 6-9) and 12-digit Aadhaar numbers are excluded by construction.
REG_NO = re.compile(r"(?<!\d)([0-2]\d{9})(?!\d)")


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def registration_numbers(pages_ocr: list[dict]) -> list[str]:
    found = []
    for p in pages_ocr:
        for line in p["lines"]:
            compact = re.sub(r"(?<=\d)[\s.](?=\d)", "", line["text"])
            found.extend(REG_NO.findall(compact))
    return sorted(set(found))


def phash(image_path: Path) -> imagehash.ImageHash:
    with Image.open(image_path) as im:
        return imagehash.phash(im.convert("L"), hash_size=16)


def _name_key(s: str) -> str:
    return re.sub(r"[^A-Z ]", "", (s or "").upper()).strip()


def same_student_pairs(truth_by_file: dict[str, dict], reg_by_file: dict[str, list[str]],
                       phash_by_file: dict[str, imagehash.ImageHash], sha_by_file: dict[str, str],
                       phash_thr: int, name_thr: float) -> list[dict]:
    pairs = []
    files = sorted(set(reg_by_file) | set(truth_by_file) | set(phash_by_file))
    for a, b in combinations(files, 2):
        reasons = []
        if sha_by_file.get(a) and sha_by_file.get(a) == sha_by_file.get(b):
            reasons.append("identical file bytes")
        if a in phash_by_file and b in phash_by_file:
            d = phash_by_file[a] - phash_by_file[b]
            if d <= phash_thr:
                reasons.append(f"near-identical page-1 image (phash distance {d})")
        shared = set(reg_by_file.get(a, [])) & set(reg_by_file.get(b, []))
        if shared:
            reasons.append(f"shared registration number(s) in OCR: {sorted(shared)}")
        ta, tb = truth_by_file.get(a), truth_by_file.get(b)
        if ta and tb:
            sa, sb = _name_key(ta["student_name"]), _name_key(tb["student_name"])
            fa, fb = _name_key(ta["father_name"]), _name_key(tb["father_name"])
            ma, mb = _name_key(ta["mother_name"]), _name_key(tb["mother_name"])
            if sa and sb:
                s_sim = fuzz.token_set_ratio(sa, sb) / 100
                f_sim = fuzz.token_set_ratio(fa, fb) / 100 if fa and fb else 0.0
                if s_sim >= name_thr and (f_sim >= name_thr or not (fa and fb)):
                    reasons.append(f"truth student name similarity {s_sim:.2f}, father {f_sim:.2f}")
        if reasons:
            pairs.append({"a": a, "b": b, "relation": "same_student", "reasons": reasons})
        elif ta and tb:
            fa, fb = _name_key(ta["father_name"]), _name_key(tb["father_name"])
            ma, mb = _name_key(ta["mother_name"]), _name_key(tb["mother_name"])
            if fa and fb and ma and mb and fuzz.ratio(fa, fb) >= 95 and fuzz.ratio(ma, mb) >= 95:
                pairs.append({"a": a, "b": b, "relation": "same_parents_different_student",
                              "reasons": [f"father {fa!r} and mother {ma!r} match; student names differ "
                                          f"({ta['student_name']!r} vs {tb['student_name']!r})"]})
    return pairs


def group_students(files: list[str], pairs: list[dict]) -> dict[str, str]:
    uf = UnionFind(files)
    for p in pairs:
        if p["relation"] == "same_student":
            uf.union(p["a"], p["b"])
    return {f: uf.find(f) for f in files}


def summarize_groups(group_of: dict[str, str]) -> list[list[str]]:
    groups = defaultdict(list)
    for f, g in group_of.items():
        groups[g].append(f)
    return sorted([sorted(v) for v in groups.values() if len(v) > 1])
