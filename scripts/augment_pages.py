"""Offline augmentation of TRAIN pages: augment image -> re-OCR -> re-align -> BIO.

    python scripts/augment_pages.py --copies 2

Writes data/processed/augmented/<doc>_aug<k>_p<N>.{png,json} and
annotations/bio_augmented/<doc>_aug<k>.json. An augmented copy is kept only if
at least as many truth fields align ("accepted") as on the original page, so a
transform that damaged the text is discarded rather than trained on.
Only the train partition is ever augmented.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.annotation.align import DocAligner, align_document  # noqa: E402
from src.annotation.anchors import TemplateIndex  # noqa: E402
from src.annotation.layout import load_pages  # noqa: E402
from src.common.io import doc_id, load_yaml, read_json, read_jsonl, resolve, write_json  # noqa: E402
from src.ocr.engine import OCREngine  # noqa: E402
from src.preprocessing.augment import augment  # noqa: E402


def accepted(res) -> int:
    return sum(f["status"] == "accepted" for f in res["fields"]) + sum(r["status"] == "accepted" for r in res["subjects"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dcfg, fcfg = load_yaml("configs/dataset.yaml"), load_yaml("configs/fields.yaml")
    truth = {r["file"]: r for r in read_jsonl(dcfg["truth_file"])}
    manifest = {d["file"]: d for d in read_json("data/dataset_manifest.json")["documents"]}
    issues_by = defaultdict(list)
    for i in read_jsonl("annotations/truth_issues.jsonl"):
        issues_by[(i["file"], i["field"])].append(i)
    aligner = DocAligner(fcfg, issues_by)
    ocr = OCREngine()
    rng = random.Random(args.seed)
    out_img = resolve("data/processed/augmented")
    out_img.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for f in read_json("data/split.json")["partitions"]["train"]:
        m = manifest[f]
        roles = [r["role"] for r in m["classification"]["page_roles"]]
        orig_pages = load_pages([read_json(p) for p in m["ocr_files"]], roles)
        orig = align_document(aligner, truth[f], orig_pages,
                              {p.page: TemplateIndex(p, fcfg["template_phrases"]) for p in orig_pages})
        for k in range(args.copies):
            ocr_pages, params = [], []
            for pi, img_path in enumerate(m["page_images"]):
                img, prm = augment(Image.open(resolve(img_path)), rng)
                stem = f"{doc_id(f)}_aug{k}_p{pi + 1}"
                img.save(out_img / f"{stem}.png")
                o = ocr.ocr_page(img, page=pi + 1)
                write_json(out_img / f"{stem}.json", o)
                ocr_pages.append(o)
                params.append(prm)
            pages = load_pages(ocr_pages, roles)
            res = align_document(aligner, truth[f], pages,
                                 {p.page: TemplateIndex(p, fcfg["template_phrases"]) for p in pages})
            if accepted(res) < accepted(orig):
                dropped += 1
                continue
            kept += 1
            labels = res["word_labels"]
            write_json(f"annotations/bio_augmented/{doc_id(f)}_aug{k}.json", {
                "file": f, "partition": "train", "augmentation": params, "source": "offline augmentation",
                "pages": [{"page": p.page, "width": p.width, "height": p.height, "role": p.role,
                           "words": [{"id": w.idx, "text": w.text, "bbox": list(w.bbox), "ocr_confidence": w.conf,
                                      "label": labels[w.uid]} for w in p.words]} for p in pages]})
            print(f"{f} aug{k}: kept ({accepted(res)} vs {accepted(orig)} accepted alignments)", flush=True)
    print(f"kept={kept} dropped={dropped}")


if __name__ == "__main__":
    main()
