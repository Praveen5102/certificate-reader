"""Geometry helpers over OCR words (layout-aware, no fixed coordinates).

Everything here is relative: "same row" means vertical overlap between boxes,
"right of" is measured from the anchor's right edge, distances are normalized
by the page's median text height. No absolute coordinates for any template.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from rapidfuzz import fuzz


@dataclass
class Word:
    uid: str            # "<page>:<id>"
    page: int
    idx: int            # word index within page
    line_id: int
    text: str
    bbox: tuple[int, int, int, int]
    conf: float

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def h(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class PageWords:
    page: int
    width: int
    height: int
    words: list[Word]
    role: str = "certificate"
    text_h: float = field(init=False)

    def __post_init__(self):
        hs = [w.h for w in self.words if w.h > 0]
        self.text_h = statistics.median(hs) if hs else 20.0


def load_pages(ocr_pages: list[dict], roles: list[str] | None = None) -> list[PageWords]:
    out = []
    for i, p in enumerate(ocr_pages):
        words = [Word(uid=f"{p['page']}:{w['id']}", page=p["page"], idx=w["id"], line_id=w["line_id"],
                      text=w["text"], bbox=tuple(w["bbox"]), conf=w["confidence"]) for w in p["words"]]
        out.append(PageWords(p["page"], p["width"], p["height"], words,
                             role=(roles[i] if roles else "certificate")))
    return out


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def v_overlap(a: tuple, b: tuple) -> float:
    """Vertical overlap as a fraction of the smaller box height."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, inter) / max(1, min(a[3] - a[1], b[3] - b[1]))


def h_overlap(a: tuple, b: tuple) -> float:
    inter = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, inter) / max(1, min(a[2] - a[0], b[2] - b[0]))


def union_bbox(words: list[Word]) -> tuple[int, int, int, int]:
    return (min(w.bbox[0] for w in words), min(w.bbox[1] for w in words),
            max(w.bbox[2] for w in words), max(w.bbox[3] for w in words))


def visual_rows(words: list[Word], min_overlap: float = 0.5) -> list[list[Word]]:
    """Group words into visual rows by vertical overlap (robust to OCR line
    splits across table columns). Rows are sorted top-to-bottom, words left-to-right."""
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        placed = False
        for row in rows:
            ref = union_bbox(row)
            if v_overlap(ref, w.bbox) >= min_overlap and abs(ref[1] + ref[3] - w.bbox[1] - w.bbox[3]) / 2 < \
                    max(ref[3] - ref[1], w.h):
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    for r in rows:
        r.sort(key=lambda w: w.bbox[0])
    rows.sort(key=lambda r: sum(w.cy for w in r) / len(r))
    return rows


def find_phrase(page: PageWords, phrase: str, cutoff: int = 85, max_words: int = 6) -> list[tuple[list[Word], float]]:
    """Find word spans (within one visual row) whose concatenation fuzzily
    matches `phrase`. Returns [(words, score)] best first."""
    target = norm(phrase)
    if not target:
        return []
    hits = []
    for row in visual_rows(page.words):
        for i in range(len(row)):
            acc = ""
            for j in range(i, min(i + max_words, len(row))):
                acc += norm(row[j].text)
                if len(acc) > len(target) * 1.6 + 4:
                    break
                s = fuzz.ratio(acc, target)
                if s >= cutoff:
                    hits.append((row[i:j + 1], s))
            # also allow the phrase to be a prefix of a single long OCR word
            w = norm(row[i].text)
            if len(w) > len(target) and w.startswith(target[: max(3, len(target) - 1)]):
                hits.append(([row[i]], 90.0))
    # keep best non-overlapping
    hits.sort(key=lambda h: (-h[1], len(h[0])))
    seen, out = set(), []
    for ws, s in hits:
        key = tuple(w.uid for w in ws)
        if any(w.uid in seen for w in ws):
            continue
        seen.update(key)
        out.append((ws, s))
    return out


def words_right_of(page: PageWords, anchor: list[Word], max_gap_h: float = 25.0,
                   stop_words: set[str] | None = None) -> list[Word]:
    """Words on the same visual row to the right of the anchor, stopping at a
    large horizontal gap (in units of text height) or a stop word."""
    ab = union_bbox(anchor)
    cands = [w for w in page.words if w.bbox[0] >= ab[2] - page.text_h * 0.3
             and v_overlap(ab, w.bbox) >= 0.45 and w.uid not in {a.uid for a in anchor}]
    cands.sort(key=lambda w: w.bbox[0])
    out, last_x = [], ab[2]
    for w in cands:
        if (w.bbox[0] - last_x) > max_gap_h * page.text_h:
            break
        if stop_words and norm(w.text) in stop_words:
            break
        out.append(w)
        last_x = w.bbox[2]
    return out
