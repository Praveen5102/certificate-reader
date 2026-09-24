"""Deterministic, leakage-aware train/validation/test split.

Units of assignment are *groups* (documents of the same student or same
household, from the audit), so a student never straddles partitions. Groups
are stratified by state and by whether the truth has subject rows, then
assigned greedily to hit the configured sizes exactly.
"""
from __future__ import annotations

import random
from collections import defaultdict


def make_split(docs: list[dict], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    """docs: [{"file", "group", "stratum"}]. Returns {partition: [files]}."""
    total = sum(sizes.values())
    if total != len(docs):
        raise ValueError(f"split sizes sum to {total} but there are {len(docs)} eligible documents")
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        groups[d["group"]].append(d)
    rng = random.Random(seed)
    # Deterministic order: shuffle within stratum, larger groups first so they
    # are placed while every partition still has room.
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for g, members in sorted(groups.items()):
        by_stratum[members[0]["stratum"]].append(g)
    ordered: list[str] = []
    strata = sorted(by_stratum)
    for s in strata:
        rng.shuffle(by_stratum[s])
    # Interleave strata so each partition gets a proportional mix.
    while any(by_stratum[s] for s in strata):
        for s in strata:
            if by_stratum[s]:
                ordered.append(by_stratum[s].pop())
    ordered.sort(key=lambda g: -len(groups[g]))  # stable: keeps interleaving among equal sizes

    parts = {k: [] for k in sizes}
    # Fill the small partitions first (test, then validation), cycling so strata spread out.
    fill_order = sorted(sizes, key=lambda k: sizes[k])
    for g in ordered:
        members = [m["file"] for m in groups[g]]
        placed = False
        for k in fill_order:
            if len(parts[k]) + len(members) <= sizes[k] and _wants(parts, sizes, k, fill_order):
                parts[k].extend(members)
                placed = True
                break
        if not placed:
            for k in reversed(fill_order):  # largest partition absorbs what's left
                if len(parts[k]) + len(members) <= sizes[k]:
                    parts[k].extend(members)
                    placed = True
                    break
        if not placed:
            raise RuntimeError(f"cannot place group {g} of size {len(members)}")
    return {k: sorted(v, key=_file_num) for k, v in parts.items()}


def _wants(parts, sizes, k, fill_order) -> bool:
    """Round-robin: a partition takes the next group only if it is not ahead of
    the others in relative fill."""
    frac = {p: len(parts[p]) / sizes[p] for p in sizes}
    return frac[k] <= min(frac.values()) + 1e-9 or k == fill_order[-1]


def _file_num(f: str) -> int:
    digits = "".join(c for c in f if c.isdigit())
    return int(digits) if digits else 0
