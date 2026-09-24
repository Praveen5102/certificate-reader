"""Truth -> OCR-word alignment: automatic BIO label *proposals* + review items.

Policy (spec §3, §25 Phase 3: "do not automatically accept questionable alignments"):
  accepted     - truth value found in OCR with high similarity, at a position
                 consistent with the field's printed label (or uniquely), and the
                 truth value has no open validation issue.   -> BIO label
  review       - found, but similarity is imperfect, the position is ambiguous,
                 or the truth value itself is flagged.        -> IGNORE + queue
  not_found    - truth value not located in OCR.            -> IGNORE + queue
  unannotated  - truth blank. The anchor-located value (if any) is masked with
                 IGNORE so the model is not taught that a printed name is "O",
                 and offered as a *suggestion* for a human to confirm.
IGNORE words get label id -100 at training time (no loss).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from rapidfuzz import fuzz

from ..validation.normalize import MONTHS, normalize_result, parse_number_words, to_float, to_int
from .anchors import TemplateIndex, best_candidate, find_anchors, table_band
from .layout import PageWords, Word, find_phrase, norm, union_bbox, v_overlap, visual_rows

IGNORE = "IGNORE"
SCALAR_FIELDS = ["STUDENT_NAME", "FATHER_NAME", "MOTHER_NAME", "HALL_TICKET", "EXAM_SESSION",
                 "RESULT", "MEDIUM", "TOTAL_MARKS", "TOTAL_IN_WORDS", "CGPA"]
TABLE_LABELS = ["SUBJECT", "MAX_MARKS", "MARKS"]
ALL_ENTITY_LABELS = SCALAR_FIELDS + TABLE_LABELS
ACCEPT_SIM = 95.0
REVIEW_SIM = 80.0


def bio_label_list() -> list[str]:
    labels = ["O"]
    for e in ALL_ENTITY_LABELS:
        labels += [f"B-{e}", f"I-{e}"]
    return labels


@dataclass
class FieldResult:
    field: str
    status: str
    truth: Any
    ocr_text: str = ""
    similarity: float | None = None
    anchored: bool | None = None
    word_uids: list[str] = field(default_factory=list)
    page: int | None = None
    bbox: list[int] | None = None
    reason: str = ""
    suggestion: str | None = None
    label_value: str | None = None   # OCR text found at the field's printed label (evaluation evidence)


class DocAligner:
    def __init__(self, fields_cfg: dict, issues_by_file_field: dict[tuple[str, str], list[dict]]):
        self.cfg = fields_cfg
        self.anchors = fields_cfg["anchors"]
        self.issues = issues_by_file_field

    # ------------------------------------------------------------------ helpers
    def _flagged(self, file: str, truth_field: str) -> list[dict]:
        return [i for i in self.issues.get((file, truth_field), []) if i["severity"] in ("warning", "error")]

    def _is_anchored(self, page: PageWords, field: str, words: list[Word]) -> bool:
        wb = union_bbox(words)
        for anchor, _ in find_anchors(page, self.anchors[field]):
            ab = union_bbox(anchor)
            if v_overlap(ab, wb) >= 0.35 and -40 * page.text_h < (wb[0] - ab[2]) < 30 * page.text_h:
                return True
            # value printed in the cell below/right of the label ("Total Marks / In Figures")
            if field in ("TOTAL_MARKS", "TOTAL_IN_WORDS") and ab[1] - page.text_h <= (wb[1] + wb[3]) / 2 <= ab[3] + 2.5 * page.text_h \
                    and wb[0] > ab[0]:
                return True
        return False

    @staticmethod
    def _windows(page: PageWords, max_len: int, tmpl: TemplateIndex):
        """Contiguous word spans from visual rows AND from OCR lines: on skewed
        photos a visual row can interleave two printed lines, while the OCR
        line follows the baseline."""
        by_line: dict[int, list[Word]] = {}
        for w in page.words:
            by_line.setdefault(w.line_id, []).append(w)
        groups = visual_rows(page.words) + [sorted(ws, key=lambda w: w.bbox[0]) for ws in by_line.values()]
        for row in groups:
            for i in range(len(row)):
                for j in range(i, min(i + max_len, len(row))):
                    span = row[i:j + 1]
                    if any(w in tmpl for w in span):
                        break
                    yield span

    def _text_candidates(self, pages, target: str, tmpls, extra_words: int = 2) -> list[tuple[list[Word], float, PageWords]]:
        tnorm = norm(target)
        n = max(1, len(target.split()))
        out = []
        for p in pages:
            if p.role == "notes":
                continue
            for span in self._windows(p, n + extra_words, tmpls[p.page]):
                if not _alnum_edges(span):
                    continue
                s = fuzz.ratio(norm("".join(w.text for w in span)), tnorm)
                if s >= REVIEW_SIM:
                    out.append((span, s, p))
        out.sort(key=lambda c: (-c[1], len(c[0])))
        # drop spans overlapping a better one
        seen, dedup = set(), []
        for span, s, p in out:
            if any(w.uid in seen for w in span):
                continue
            seen.update(w.uid for w in span)
            dedup.append((span, s, p))
        return dedup

    # ------------------------------------------------------------------ scalar fields
    def _truth_value(self, field: str, rec: dict) -> Any:
        if field == "EXAM_SESSION":
            return (rec.get("month_of_pass") or "", rec.get("year_of_pass"))
        tf = self.cfg["entity_labels"][field]["truth_field"]
        return rec.get(tf)

    def _truth_field_names(self, field: str) -> list[str]:
        tf = self.cfg["entity_labels"][field]["truth_field"]
        return tf if isinstance(tf, list) else [tf]

    def align_scalar(self, field: str, rec: dict, pages: list[PageWords], tmpls) -> FieldResult:
        file = rec["file"]
        truth = self._truth_value(field, rec)
        flags = [i for tf in self._truth_field_names(field) for i in self._flagged(file, tf)]
        blank = (truth in (None, "") if field != "EXAM_SESSION" else (not truth[0] and truth[1] is None))
        anchored_cand = best_candidate(pages, field, self.anchors[field], tmpls)
        label_value = anchored_cand.text if anchored_cand else None
        r = self._align_scalar(field, truth, blank, flags, anchored_cand, pages, tmpls)
        r.label_value = label_value
        return r

    def _align_scalar(self, field, truth, blank, flags, anchored_cand, pages, tmpls) -> FieldResult:
        if blank:
            r = FieldResult(field, "unannotated", truth, reason="truth blank")
            if anchored_cand:
                r.word_uids = [w.uid for w in anchored_cand.words]
                r.suggestion = anchored_cand.text
                r.page, r.bbox = anchored_cand.words[0].page, list(union_bbox(anchored_cand.words))
            return r

        cands = self._scalar_candidates(field, truth, pages, tmpls)
        if not cands:
            r = FieldResult(field, "not_found", truth, reason="truth value not located in OCR")
            if anchored_cand:
                r.word_uids = [w.uid for w in anchored_cand.words]
                r.ocr_text = anchored_cand.text
                r.suggestion = anchored_cand.text
                r.page, r.bbox = anchored_cand.words[0].page, list(union_bbox(anchored_cand.words))
                r.reason += f"; OCR value at the printed label reads {anchored_cand.text!r}"
            return r

        scored = [(span, s, p, self._is_anchored(p, field, span)) for span, s, p in cands]
        anchored = [c for c in scored if c[3]]
        pool = anchored or scored
        best = max(pool, key=lambda c: c[1])
        span, sim, page, is_anch = best
        r = FieldResult(field, "accepted", truth, ocr_text=" ".join(w.text for w in span), similarity=round(sim, 1),
                        anchored=is_anch, word_uids=[w.uid for w in span], page=page.page,
                        bbox=list(union_bbox(span)))
        reasons = []
        if sim < ACCEPT_SIM:
            reasons.append(f"OCR/truth similarity {sim:.0f} < {ACCEPT_SIM:.0f}")
        if not is_anch:
            strong = [c for c in scored if c[1] >= ACCEPT_SIM]
            if len(strong) != 1:
                reasons.append(f"{len(strong)} matching positions and none next to the printed label")
            elif anchored_cand is not None and not set(w.uid for w in anchored_cand.words) & set(r.word_uids):
                reasons.append(f"truth found away from the printed label; OCR value at the label reads "
                               f"{anchored_cand.text!r}")
        if anchored_cand is not None and field in {"STUDENT_NAME", "FATHER_NAME", "MOTHER_NAME"}:
            full = anchored_cand.text
            if set(w.uid for w in anchored_cand.words) & set(r.word_uids) and \
                    fuzz.ratio(norm(full), norm(str(truth))) < 90 and len(norm(full)) > len(norm(str(truth))):
                reasons.append(f"printed value {full!r} is longer than truth {truth!r} (truth may be truncated)")
        if flags:
            reasons.append("truth value flagged: " + "; ".join(sorted({f['check'] for f in flags})))
        if reasons:
            r.status = "review"
            r.reason = " | ".join(reasons)
            if anchored_cand is not None:
                r.suggestion = anchored_cand.text
        return r

    def _scalar_candidates(self, field, truth, pages, tmpls):
        if field == "EXAM_SESSION":
            month, year = truth
            out = []
            for p in pages:
                if p.role == "notes":
                    continue
                for row in visual_rows(p.words):
                    for i in range(len(row)):
                        for j in range(i, min(i + 3, len(row))):
                            span = row[i:j + 1]
                            txt = norm("".join(w.text for w in span))
                            has_year = year is not None and str(year) in txt
                            has_month = bool(month) and month[:3] in txt
                            other_month = any(m[:3] in txt for m in MONTHS)
                            if year is not None and month:
                                ok = has_year and has_month
                            elif year is not None:
                                ok = has_year and other_month
                            else:
                                ok = has_month and bool(re.search(r"(19|20)\d\d", txt))
                            if ok:
                                # a month+year span; penalize partial truth (only one of the two known)
                                out.append((span, 100.0 if (year is not None and month) else 85.0, p))
                                break
            return _dedup(out)
        if field == "TOTAL_MARKS":
            return _dedup([([w], 100.0, p) for p in pages if p.role != "notes" for w in p.words
                           if to_int(w.text)[0] == truth or norm(w.text) == str(truth)])
        if field == "TOTAL_IN_WORDS":
            out = []
            for p in pages:
                if p.role == "notes":
                    continue
                for row in visual_rows(p.words):
                    for i in range(len(row)):
                        for j in range(i, min(i + 4, len(row))):
                            if parse_number_words(" ".join(w.text for w in row[i:j + 1])) == truth:
                                out.append((row[i:j + 1], 100.0, p))
            # prefer the longest span that parses (the whole "*SEVEN**FIVE***TWO*")
            out.sort(key=lambda c: -len(c[0]))
            return _dedup(out)
        if field == "CGPA":
            return _dedup([([w], 100.0, p) for p in pages if p.role != "notes" for w in p.words
                           if (v := to_float(w.text)[0]) is not None and abs(v - float(truth)) < 1e-6
                           and "." in w.text])
        if field == "HALL_TICKET":
            return _dedup([([w], 100.0, p) for p in pages if p.role != "notes" for w in p.words
                           if re.sub(r"\D", "", w.text) == str(truth)])
        if field == "RESULT":
            canon, note = normalize_result(str(truth))
            target = canon if "not in vocabulary" not in note else str(truth)
            return self._text_candidates(pages, target, tmpls)
        return self._text_candidates(pages, str(truth), tmpls)

    # ------------------------------------------------------------------ table
    def table_band(self, page: PageWords) -> tuple[float, float] | None:
        return table_band(page, self.cfg["table"]["header_phrases"], self.cfg["table"]["end_phrases"])

    def align_subjects(self, rec: dict, pages: list[PageWords], taken: set[str]) -> list[dict]:
        """Locate each truth subject row: name span + adjacent (max, marks) integers.
        Subject-name words are shared by the paper I and II rows; mark words are
        consumed so each printed number supports at most one truth row."""
        rows_out = []
        boundaries: dict[int, tuple[str, float] | None] = {}
        consumed: set[str] = set(taken)
        cert_pages = [p for p in pages if p.role != "notes"]
        for idx, s in enumerate(rec.get("subjects") or []):
            res = {"row": idx, "truth": s, "status": "not_found", "reason": "", "word_uids": {}}
            best = None
            for p in cert_pages:
                boundary = boundaries.setdefault(p.page, self.year_boundary(p))
                for span, sim, _ in self._subject_name_spans(p, s["subject_name"], consumed):
                    pair = self._find_pair(p, span, s, consumed, boundary)
                    if pair is not None:
                        key = (sim, 1)
                        if best is None or key > best[0]:
                            best = (key, p, span, pair)
            if best is None:
                res["reason"] = "subject row (name + adjacent max/marks pair) not located"
                rows_out.append(res)
                continue
            (sim, _), p, span, (wmax, wmarks) = best
            res.update(status="accepted" if sim >= 90 else "review", page=p.page,
                       word_uids={"SUBJECT": [w.uid for w in span], "MAX_MARKS": [wmax.uid], "MARKS": [wmarks.uid]},
                       similarity=round(sim, 1), ocr_text=" ".join(w.text for w in span))
            if sim < 90:
                res["reason"] = f"subject name similarity {sim:.0f}"
            consumed.update([wmarks.uid, wmax.uid])
            rows_out.append(res)
        return rows_out

    def _subject_name_spans(self, page: PageWords, name: str, consumed: set[str]):
        tnorm = norm(name)
        n = len(name.split())
        out = []
        for row in visual_rows(page.words):
            for i in range(len(row)):
                for j in range(i, min(i + n + 2, len(row))):
                    span = row[i:j + 1]
                    if any(w.uid in consumed for w in span):   # e.g. ENGLISH already labeled MEDIUM
                        break
                    if not _alnum_edges(span):                  # never label a leading ':' / trailing '-'
                        continue
                    sim = fuzz.ratio(norm("".join(w.text for w in span)), tnorm)
                    if sim >= 85:
                        out.append((span, sim, page))
        out.sort(key=lambda c: -c[1])
        return out

    def year_boundary(self, page: PageWords) -> tuple[str, float] | None:
        """Where the I-year and II-year parts of the table meet.

        Returns ("x", boundary) when the year headers sit side by side (columns),
        ("y", boundary) when they are stacked (row blocks / vocational sections),
        or None when the headers cannot both be found."""
        return year_boundary(page, self.cfg["table"]["year_headers"])

    @staticmethod
    def _find_pair(page: PageWords, span: list[Word], s: dict, consumed: set[str],
                   boundary: tuple[str, float] | None = None):
        """Adjacent (max, marks) integers to the right of the subject name on its
        visual row, in reading order; paper I -> leftmost unconsumed pair,
        paper II -> rightmost. Rows that print no maximum (open-school
        'Theory/Practical/Total') are not located -> sent to review."""
        sb = union_bbox(span)
        nums = [w for w in page.words if w.bbox[0] > sb[2] and v_overlap(sb, w.bbox) >= 0.4
                and to_int(w.text)[0] is not None and w.uid not in consumed]
        nums.sort(key=lambda w: w.bbox[0])
        vals = [to_int(w.text)[0] for w in nums]
        pairs = [(nums[k], nums[k + 1]) for k in range(len(nums) - 1)
                 if vals[k] == s["maximum_marks"] and vals[k + 1] == s["marks"]]
        if boundary is not None and s.get("paper") in ("I", "II"):
            axis, b = boundary
            coord = (lambda w: w.cx) if axis == "x" else (lambda w: w.cy)
            want_first = s["paper"] == "I"
            pairs = [pr for pr in pairs if (coord(pr[1]) < b) == want_first]
        if pairs:
            return pairs[0] if s["paper"] == "I" else pairs[-1]
        return None


def _alnum_edges(span: list[Word]) -> bool:
    """A labeled span must start and end with a word containing a letter or digit."""
    return any(c.isalnum() for c in span[0].text) and any(c.isalnum() for c in span[-1].text)


def _year_header_spans(page: PageWords, phrases: list[str]) -> list[list[Word]]:
    """Exact (normalized) matches only: fuzzy matching cannot tell 'I Year'
    from 'II Year'."""
    targets = {norm(p) for p in phrases}
    out = []
    for row in visual_rows(page.words):
        for i in range(len(row)):
            for j in range(i, min(i + 2, len(row))):
                if norm("".join(w.text for w in row[i:j + 1])) in targets:
                    out.append(row[i:j + 1])
    return out


def year_boundary(page: PageWords, year_headers: dict[str, list[str]]) -> tuple[str, float] | None:
    first = _year_header_spans(page, year_headers["I"])
    second = _year_header_spans(page, year_headers["II"])
    if not first or not second:
        return None
    a, b = union_bbox(first[0]), union_bbox(second[0])
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if abs(bx - ax) >= abs(by - ay):
        # side-by-side column groups: boundary midway between the two headers
        return ("x", (a[2] + b[0]) / 2 if b[0] > a[2] else (ax + bx) / 2)
    return ("y", (a[3] + b[1]) / 2 if b[1] > a[3] else (ay + by) / 2)


def _dedup(cands):
    seen, out = set(), []
    for span, s, p in sorted(cands, key=lambda c: -c[1]):
        key = tuple(w.uid for w in span)
        if any(w.uid in seen for w in span):
            continue
        seen.update(key)
        out.append((span, s, p))
    return out


def align_document(aligner: DocAligner, rec: dict | None, pages: list[PageWords],
                   tmpls: dict[int, TemplateIndex]) -> dict:
    """Produce word labels + per-field results for one document."""
    word_label: dict[str, str] = {w.uid: "O" for p in pages for w in p.words}
    field_results: list[FieldResult] = []
    taken: set[str] = set()
    for field in SCALAR_FIELDS:
        fr = aligner.align_scalar(field, rec, pages, tmpls)
        field_results.append(fr)
        if fr.status == "accepted":
            if any(u in taken for u in fr.word_uids):
                fr.status, fr.reason = "review", "words already assigned to another field"
            else:
                for k, u in enumerate(fr.word_uids):
                    word_label[u] = ("B-" if k == 0 else "I-") + field
                taken.update(fr.word_uids)
                continue
        for u in fr.word_uids:
            if u not in taken:
                word_label[u] = IGNORE

    subj_rows = aligner.align_subjects(rec, pages, taken)
    subjects_complete = bool(rec.get("subjects")) and all(r["status"] == "accepted" for r in subj_rows)
    subject_flags = aligner._flagged(rec["file"], "total_marks") + [
        i for (f, fld), iss in aligner.issues.items() if f == rec["file"] and fld.startswith("subjects")
        for i in iss if i["severity"] in ("warning", "error")]
    for r in subj_rows:
        for lab, uids in r["word_uids"].items():
            for k, u in enumerate(uids):
                if r["status"] == "accepted" and u not in taken:
                    word_label[u] = ("B-" if k == 0 else "I-") + lab
                    taken.add(u)
                elif u not in taken:
                    word_label[u] = IGNORE

    # Table region words that are not positively labeled: "O" only when the truth
    # table is complete and unflagged; otherwise we cannot be sure they are not
    # unannotated subject cells -> IGNORE.
    table_uncertain = (not rec.get("subjects")) or (not subjects_complete) or bool(subject_flags)
    band_found = []
    for p in pages:
        if p.role == "notes":
            continue
        band = aligner.table_band(p)
        band_found.append(band is not None)
        if band is None:
            if table_uncertain:
                for row in visual_rows(p.words):
                    if sum(to_int(w.text)[0] is not None for w in row) >= 2:
                        for w in row:
                            if word_label[w.uid] == "O":
                                word_label[w.uid] = IGNORE
            continue
        top, bot = band
        if table_uncertain:
            for w in p.words:
                if top <= w.cy <= bot and word_label[w.uid] == "O":
                    word_label[w.uid] = IGNORE

    return {
        "fields": [asdict(f) for f in field_results],
        "subjects": subj_rows,
        "subjects_complete": subjects_complete,
        "table_masked": table_uncertain,
        "table_band_found": band_found,
        "word_labels": word_label,
    }
