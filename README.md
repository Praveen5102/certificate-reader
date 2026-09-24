# Intermediate Certificate ML Extraction System

Structured JSON extraction from Telangana / Andhra Pradesh Intermediate (12th)
pass certificates and marks memos:

```
PDF / image -> document validation & classification -> page rendering -> OCR (words + boxes)
  -> field / table tagging (rules today; fine-tuned LiLT next) -> table reconstruction
  -> normalization -> validation -> confidence -> review routing -> JSON
```

**Status (2026-09-24):**
- Truth v2 built (`annotations/truth_v2/`).
- LiLT model trained (`experiments/exp_004_lilt`).
- Pipeline v2 is the default (`experiments/exp_006_pipeline_v2`). On the 10 held-out test
  certificates it gets 134/138 subject rows exactly right and 9/10 certificates fully right.
- Every test run is listed in `experiments/test_evaluations.log`.

## Layout

```
DataSet/                      raw PDFs (read-only, never modified)
ground_truth_98_clean.jsonl   canonical truth (read-only, never modified)
configs/                      dataset / ocr / fields / model / training / inference YAML
data/
  dataset_audit.json          Phase 1 findings (counts, duplicates, truth quality)
  dataset_manifest.json       one entry per PDF: hashes, pages, OCR stats, classification, student group
  split.json                  seed, partitions, leakage checks, disclosure of pre-split inspection
  {train,validation,test}/manifest.jsonl
  processed/pages/            rendered page images + page_index.json
  processed/ocr/              OCR per page: words, lines, boxes, confidences
annotations/
  truth_issues.jsonl          every truth-validation finding (flag only)
  alignment_report.jsonl      per (document, field): where the truth value sits in OCR
  bio/                        word-level BIO proposals (train + validation only)
  annotation_review_queue.jsonl
  truth_suggestions.jsonl     OCR-located values for BLANK truth fields (proposals, not truth)
src/
  preprocessing/  render, classify, audit, split, augment
  ocr/            RapidOCR wrapper (engine-agnostic output format)
  annotation/     truth validation, layout geometry, anchors, truth->OCR alignment
  models/         rule baseline, feature encoding, layout-model tagger
  training/       fine-tuning loop, experiment tracking
  evaluation/     metrics, error analysis
  inference/      schema (pydantic), table reconstruction, post-processing, pipeline
  validation/     normalization, post-extraction checks
scripts/          render_pages, run_ocr, audit_dataset, make_split, build_annotations,
                  run_baseline, train, evaluate, infer, augment_pages
experiments/      exp_NNN_<name>/ (never overwritten), index.jsonl, test_evaluations.log
reports/          first_deliverable.md
tests/
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

## Reproduce

```bash
python scripts/render_pages.py        # PDF -> page images
python scripts/run_ocr.py             # OCR (resumable; ~4 s/page on CPU)
python scripts/audit_dataset.py       # data/dataset_audit.json, manifest, truth_issues
python scripts/make_split.py          # deterministic leakage-aware split
python scripts/build_annotations.py   # BIO proposals + review queue + suggestions
python scripts/run_baseline.py        # rule baseline on train/validation (add --include-test for the logged test run)
python -m pytest
```

Training (after approval):

```bash
python scripts/train.py --smoke --out <scratch-dir>   # code-path check only
python scripts/train.py --name lilt                   # new experiments/exp_NNN_lilt/
python scripts/evaluate.py --exp experiments/exp_NNN_lilt --partitions validation
```

Inference:

```bash
python scripts/infer.py certificate.pdf                              # flat JSON (rule baseline)
python scripts/infer.py a.pdf b.jpg --out results/ --full            # traceable JSON per file
python scripts/infer.py a.pdf --tagger model --model-dir experiments/exp_NNN_lilt/model
```

## Key design decisions

**Truth is never modified.** Problems are flagged in `annotations/truth_issues.jsonl`
and the review queue. Corrections should go into a new versioned truth file,
with `dataset_version` bumped in `configs/dataset.yaml` and the split regenerated.

**Blank truth means "unannotated", not "absent".** Many blank values belong to
fields that are clearly printed on the certificate. Blank fields are excluded
from metrics (reported as coverage). In training data the printed value is
masked (`IGNORE`, no loss) so the model is not taught that a printed name is "O".

**OCR engine: RapidOCR (PP-OCR via ONNX Runtime).** Pretrained, pip-only (Tesseract
would need a system install), gives word boxes, word confidences and line
grouping, and runs at about 4 s/page on CPU. OCR output is stored per page so
extraction errors can be traced.

**Model: LiLT-RoBERTa-base (MIT) for token classification over OCR words + boxes.**
LayoutLMv3 weights are CC BY-NC-SA 4.0 (non-commercial), which blocks a
commercial deployment. LiLT uses text and 2-D layout without an image backbone,
matches what the OCR layer produces, and trains on CPU. Table *structure*
(subject -> paper -> marks) is rebuilt geometrically from tagged cells and the
detected "I Year / II Year" headers, never from reading order alone.

**Confidence is evidence, not invention.**
- Rules: min OCR word confidence x anchor match quality.
- Model: min token probability x min OCR confidence.
- Independent readings that agree (total in figures and in words) are combined by noisy-OR.
- Scores are uncalibrated. Review thresholds in `configs/inference.yaml` are provisional and must be fitted on the validation split.

**Validation flags, never overwrites.** Examples:
- sum(subject marks) != total
- total in figures != total in words
- marks > maximum
- implausible year
- hall-ticket format

**Test discipline.**
- `src/training/train.py` refuses to load the test partition.
- Every test evaluation is logged in `experiments/test_evaluations.log`.
- `data/split.json` records which documents were looked at before the split existed.

## Deviations from the spec (and why)

| Spec | Here | Reason |
|---|---|---|
| Hugging Face `datasets` | plain PyTorch dataset | pandas/pyarrow native DLLs are blocked by this machine's Windows Application Control policy |
| MLflow / W&B | file-based tracking (`run.json`, `index.jsonl`) | MLflow imports pandas (blocked); W&B needs an external account |
| Tesseract | RapidOCR | not installed; RapidOCR is pip-only and gives word boxes + confidences |
| "existing deterministic split" | generated split (seed 20240917) | no split file was supplied |
| "existing OCR/rule baseline" | re-implemented rule baseline | the original baseline code was not supplied |
| LayoutLMv3 candidate | LiLT recommended | license (non-commercial weights) and CPU-only hardware |

## Schema proposals (not applied to the canonical schema)

- `cgpa`: already a truth column; AP grade memos print CGPA instead of a total.
  Emitted in the flat output so it can be evaluated.
- `document_subtype`: pass certificate cum memorandum / memorandum of marks /
  online results memo / open-school certificate / vocational.
- Total marks in words, used as a second independent reading of `total_marks`.
- Clarify `maximum_marks` (grand maximum vs. per-subject) and `result` (printed
  grade/division vs. PASSED/QUALIFIED). See the report.

## Website (local)

Double-click `start_website.bat` (or run `python scripts/serve.py`), then open
http://127.0.0.1:8000. Upload a PDF or photo; the extracted details, subject
table, and any items needing a human check appear below. Everything runs on
this computer — nothing is uploaded anywhere.
