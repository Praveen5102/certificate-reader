"""Anchor-based value location ("the value printed to the right of a label").

Used by
  * annotation alignment - to mask/suggest values for fields the truth leaves blank;
  * the rule baseline     - as its extraction method.
No coordinates are hard-coded: everything is relative to the detected label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..validation.normalize import MEDIA, MONTHS, parse_number_words, to_float, to_int
from .layout import PageWords, Word, find_phrase, norm, union_bbox, v_overlap, visual_rows, words_right_of

# Generic words that are only a field label when they start their visual row
# ("in JUNE 2024", "with ENGLISH", "NAME : ..."); elsewhere they are prose.
ROW_START_ANCHORS = {"IN", "WITH", "NAME"}


@dataclass
class Candidate:
    field: str
    words: list[Word]
    anchor: list[Word]
    anchor_score: float
    text: str = ""
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.text:
            self.text = " ".join(w.text for w in self.words)

    @property
    def ocr_conf(self) -> float:
        return min((w.conf for w in self.words), default=0.0)


class TemplateIndex:
    """Word uids that belong to printed template phrases on a page."""

    def __init__(self, page: PageWords, phrases: list[str]):
        self.uids: set[str] = set()
        for ph in phrases:
            n = len(ph.split())
            for ws, _ in find_phrase(page, ph, cutoff=88, max_words=n + 2):
                # Never swallow a glued value: only mark words whose text is
                # itself (almost) entirely template.
                for w in ws:
                    if len(norm(w.text)) <= len(norm(ph)) + 2:
                        self.uids.add(w.uid)
        for w in page.words:
            if norm(w.text) in {"BEARING", "NAME"} or w.text.strip() in {":", "-", "."}:
                self.uids.add(w.uid)

    def __contains__(self, w: Word) -> bool:
        return w.uid in self.uids


def find_anchors(page: PageWords, phrases: list[str]) -> list[tuple[list[Word], float]]:
    rows = visual_rows(page.words)
    row_first = {r[0].uid for r in rows if r}
    out = []
    for ph in phrases:
        n = len(ph.split())
        generic = norm(ph) in ROW_START_ANCHORS
        exact = generic or len(norm(ph)) <= 4
        for ws, score in find_phrase(page, ph, cutoff=100 if exact else 86, max_words=n + 1):
            if exact and norm("".join(w.text for w in ws)) != norm(ph):
                continue
            if generic and ws[0].uid not in row_first:
                continue
            out.append((ws, score))
    # longest/most specific anchors first
    out.sort(key=lambda a: (-len(norm("".join(w.text for w in a[0]))), -a[1]))
    dedup, seen = [], set()
    for ws, s in out:
        if any(w.uid in seen for w in ws):
            continue
        seen.update(w.uid for w in ws)
        dedup.append((ws, s))
    return dedup


_NAME_TOKEN = re.compile(r"^[A-Z][A-Z.'\-]*$")


def _take_name(words: list[Word], tmpl: TemplateIndex) -> list[Word]:
    out = []
    for w in words:
        t = w.text.strip(" :;,")
        if w in tmpl or not t:
            if out:
                break
            continue
        if not _NAME_TOKEN.match(t) or any(c.isdigit() for c in t):
            break
        out.append(w)
    return out


def _take_regex(words: list[Word], pattern: str, tmpl: TemplateIndex, max_words: int = 1) -> list[Word]:
    rx = re.compile(pattern)
    for i, w in enumerate(words):
        for j in range(i, min(i + max_words, len(words))):
            span = words[i:j + 1]
            joined = "".join(x.text for x in span).replace(" ", "")
            if rx.search(joined):
                return span
    return []


def _take_session(words: list[Word], tmpl: TemplateIndex) -> list[Word]:
    month_rx = "|".join(m[:3] for m in MONTHS)
    for i, w in enumerate(words[:4]):
        span = words[i:i + 3]
        for j in range(1, len(span) + 1):
            joined = "".join(x.text for x in span[:j]).upper()
            if re.search(r"(19[89]\d|20[0-3]\d)", joined) and re.search(month_rx, joined):
                return span[:j]
    return []


def _take_categorical(words: list[Word], tmpl: TemplateIndex, vocab: list[str] | None = None) -> list[Word]:
    # Values are upper case on certificates; the online memo prints medium in
    # title case ("English"), so the medium vocabulary match is case-insensitive.
    token_rx = re.compile(r"[A-Z][A-Z\-]*" if vocab is None else r"[A-Za-z][A-Za-z\-]*")
    out = []
    for w in words:
        if w.text.strip().endswith(":") and out:   # next label ("DATE:") ends the value
            break
        t = w.text.strip(" :;,.")
        if w in tmpl or not t or not token_rx.fullmatch(t):
            if out:
                break
            continue
        out.append(w)
        if len(out) >= 3:
            break
    if vocab and out:
        # keep only the first token for single-word vocabularies (medium)
        keep = [w for w in out if norm(w.text) in {norm(v) for v in vocab}]
        return keep[:1]
    return out


def _take_int(words: list[Word], lo: int = 10, hi: int = 1200) -> list[Word]:
    for w in words:
        v, _ = to_int(w.text)
        if v is not None and lo <= v <= hi:
            return [w]
    return []


def _take_number_words(words: list[Word]) -> list[Word]:
    """Longest leading span (<= 4 words) that parses as digit words."""
    for i in range(min(3, len(words))):
        for j in range(min(i + 4, len(words)), i, -1):
            if parse_number_words(" ".join(w.text for w in words[i:j])) is not None:
                return words[i:j]
    return []


def _take_float(words: list[Word]) -> list[Word]:
    for w in words:
        v, _ = to_float(w.text)
        if v is not None and 0 < v <= 10 and "." in w.text:
            return [w]
    return []


def value_words(page: PageWords, anchor: list[Word], max_gap_h: float = 18.0) -> list[Word]:
    """Words of the value that follows an anchor.

    Uses OCR *lines* rather than raw x-order: on skewed photos a label row can
    interleave with a neighbouring line (e.g. the title), but the detector's
    text lines follow the baseline. First the remainder of the anchor's own
    OCR line; otherwise the nearest OCR line starting to the right on the same
    visual row."""
    ab = union_bbox(anchor)
    last = anchor[-1]
    same = sorted([w for w in page.words if w.line_id == last.line_id and w.bbox[0] >= last.bbox[2] - 2
                   and w.uid not in {a.uid for a in anchor}], key=lambda w: w.bbox[0])
    if any(any(c.isalnum() for c in w.text) for w in same):
        return same
    right = words_right_of(page, anchor, max_gap_h=max_gap_h)
    lines: dict[int, list[Word]] = {}
    for w in right:
        if w.line_id != last.line_id:
            lines.setdefault(w.line_id, []).append(w)
    if not lines:
        return same
    first_line = min(lines.values(), key=lambda ws: min(w.bbox[0] for w in ws))
    first_line_all = sorted([w for w in page.words if w.line_id == first_line[0].line_id
                             and w.bbox[0] >= ab[2] - page.text_h], key=lambda w: w.bbox[0])
    return same + [w for w in first_line_all if w not in same]


def _vocab_ok(words: list[Word], vocab: list[str]) -> bool:
    from ..validation.normalize import normalize_categorical
    text = " ".join(w.text for w in words)
    _, note = normalize_categorical(text, vocab)
    return "not in vocabulary" not in note


def locate(page: PageWords, field: str, anchors: list[str], tmpl: TemplateIndex) -> list[Candidate]:
    from ..validation.normalize import RESULTS
    cands = []
    for anchor, ascore in find_anchors(page, anchors):
        right = value_words(page, anchor)
        if field in {"STUDENT_NAME", "FATHER_NAME", "MOTHER_NAME"}:
            ws = _take_name(right, tmpl)
        elif field == "HALL_TICKET":
            ws = _take_regex(right, r"^\d{8,12}$", tmpl, max_words=2)
        elif field == "EXAM_SESSION":
            ws = _take_session(right, tmpl)
        elif field == "RESULT":
            ws = _take_categorical(right, tmpl)
            # shrink to the longest prefix that is a known result value
            while ws and not _vocab_ok(ws, RESULTS):
                ws = ws[:-1]
        elif field == "MEDIUM":
            ws = _take_categorical(right, tmpl, vocab=MEDIA)
            if not ws:  # "ENGLISH as the medium of instruction": value is LEFT of the anchor
                left = [w for w in page.words if w.bbox[2] <= union_bbox(anchor)[0]
                        and v_overlap(union_bbox(anchor), w.bbox) >= 0.45 and norm(w.text) in {norm(m) for m in MEDIA}]
                ws = sorted(left, key=lambda w: -w.bbox[2])[:1]
        elif field == "TOTAL_MARKS":
            ws = _take_int(right, 100, 1200)
            if not ws:  # value printed below/right in a separate cell row ("Total Marks / In Figures | 576")
                ab = union_bbox(anchor)
                below = [w for w in page.words if ab[1] - page.text_h <= w.cy <= ab[3] + 2.5 * page.text_h
                         and w.bbox[0] > ab[0]]
                ws = _take_int(sorted(below, key=lambda w: (w.bbox[0])), 100, 1200)
        elif field == "CGPA":
            ws = _take_float(right)
        elif field == "TOTAL_IN_WORDS":
            ws = _take_number_words(right)
        else:
            ws = []
        if ws:
            cands.append(Candidate(field, ws, anchor, ascore))
    return cands


def best_candidate(pages: list[PageWords], field: str, anchors: list[str],
                   tmpls: dict[int, TemplateIndex]) -> Candidate | None:
    allc = []
    for p in pages:
        if p.role == "notes":
            continue
        allc.extend(locate(p, field, anchors, tmpls[p.page]))
    if not allc:
        return None
    # Prefer the most specific anchor (longest phrase), then the highest-confidence OCR.
    allc.sort(key=lambda c: (-len(norm("".join(w.text for w in c.anchor))), -c.anchor_score, -c.ocr_conf))
    return allc[0]


def table_band(page: PageWords, header_phrases: list[str], end_phrases: list[str]) -> tuple[float, float] | None:
    """Vertical extent of the marks table: from the header row ("Subject",
    "Maximum Marks", ...) to the totals row ("Total Marks", "GRAND TOTAL", ...)."""
    tops = [union_bbox(ws)[1] for ph in header_phrases for ws, _ in find_phrase(page, ph, cutoff=88)]
    bots = [union_bbox(ws)[1] for ph in end_phrases for ws, _ in find_phrase(page, ph, cutoff=88)]
    if not tops:
        return None
    top = min(tops) - page.text_h * 2.5
    below = [b for b in bots if b > top + 3 * page.text_h]
    bot = min(below) - page.text_h * 0.3 if below else page.height
    return top, bot
