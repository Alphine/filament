"""
Solar Filament Segmentation 2026 -- INTER-ANNOTATOR AGREEMENT (human ceiling),
CPU-only, annotation file only (no images, no model, no training).

QUESTION. Our model reaches official PQ 0.4325 with a mean TP IoU of 0.676, and
54% of its errors are localisation near-misses (IoU 0-0.5). If human
annotators disagree with each other at a similar level, those near-misses are
mostly ground-truth ambiguity and our realistic ceiling is close. If their
agreement is high (IoU>0.85) there is real room to improve the boundaries.

METHOD. The official metric scores a prediction against EVERY annotator pass
separately (different image_id for the same file = different pass). Here one
pass acts as the "prediction" and another as the "ground truth", with EXACTLY
the same rule as official_pq_holdout (IoU>0.5 is a hit; FP = prediction
without a hit; FN = ground truth without a hit; PQ = sumIoU / (TP + 0.5FP +
0.5FN)), for all ordered pairs (A,B), A != B, on images with >=2 passes.
Reported for (a) all multi-pass training images and (b) our 50 hold-out images
(seed 1234). Besides PQ: the mean IoU of hit pairs and the share of near-misses
among the errors, so they can be compared directly with our model's numbers.
"""

import json
import os
import time
from collections import Counter

import numpy as np
import pycocotools.mask as mask_util

OUT = "/kaggle/working"
N_VAL = 50


def discover():
    ann = None
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".json") and "nnotation" in f:
                ann = os.path.join(root, f)
    if not ann:
        raise SystemExit("annotation file not found")
    return ann


def load(ann_path):
    coco = json.load(open(ann_path))
    images = {im["id"]: im for im in coco["images"]}
    by_file = {}
    for a in coco["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            continue
        fn = os.path.basename(im["file_name"])
        by_file.setdefault(fn, {}).setdefault(a["image_id"], []).append(
            (a, im["height"], im["width"]))
    return by_file


def rles_of(items):
    out = []
    for a, h, w in items:
        seg = a.get("segmentation")
        if not seg:
            continue
        r = mask_util.frPyObjects(seg, h, w)
        r = mask_util.merge(r) if isinstance(r, list) else r
        out.append(r)
    return out


def compare(pred, gt):
    """Official rule: hit = IoU>0.5. Returns (sumIoU, tp, fp, fn, iou_hits,
    pred_max_iou, gt_max_iou)."""
    n_p, n_g = len(pred), len(gt)
    if n_p == 0 or n_g == 0:
        return 0.0, 0, n_p, n_g, [], [0.0] * n_p, [0.0] * n_g
    iou = np.asarray(mask_util.iou(pred, gt, [0] * n_g), dtype=np.float64)
    hit = iou > 0.5
    ious = iou[hit].tolist()
    fp = int((hit.sum(1) == 0).sum())
    fn = int((hit.sum(0) == 0).sum())
    return float(sum(ious)), len(ious), fp, fn, ious, iou.max(1).tolist(), iou.max(0).tolist()


def pooled(stats):
    S = sum(s[0] for s in stats)
    tp = sum(s[1] for s in stats)
    fp = sum(s[2] for s in stats)
    fn = sum(s[3] for s in stats)
    d = tp + 0.5 * fp + 0.5 * fn
    return (S / d if d else 0.0), tp, fp, fn


def summarize(name, per_image):
    """per_image: dict fn -> list of compare() tuples (all ordered pairs)."""
    flat = [t for v in per_image.values() for t in v]
    if not flat:
        return {"name": name, "n_images": 0}
    pq, tp, fp, fn = pooled(flat)
    ious = [x for t in flat for x in t[4]]
    cat = Counter()
    for t in flat:
        for m in t[5]:
            if m > 0.5:
                continue
            cat["FP_localization" if m > 0 else "FP_pure"] += 1
        for m in t[6]:
            if m > 0.5:
                continue
            cat["FN_localization" if m > 0 else "FN_pure"] += 1
    n_err = sum(cat.values())
    rng = np.random.default_rng(0)
    keys = list(per_image)
    bs = []
    for _ in range(2000):
        pick = rng.integers(0, len(keys), len(keys))
        bs.append(pooled([t for k in pick for t in per_image[keys[k]]])[0])
    per_img_pq = [pooled(v)[0] for v in per_image.values() if v]
    res = {
        "name": name, "n_images": len(per_image), "n_pairs": len(flat),
        "pq": round(pq, 4), "pq_se_bootstrap": round(float(np.std(bs)), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "mean_iou_tp": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou_tp": round(float(np.median(ious)), 4) if ious else None,
        "frac_iou_ge_0.85": round(float(np.mean(np.array(ious) >= 0.85)), 4) if ious else None,
        "frac_iou_lt_0.65": round(float(np.mean(np.array(ious) < 0.65)), 4) if ious else None,
        "errors": dict(cat),
        "near_miss_share_of_errors": round((cat["FP_localization"] + cat["FN_localization"]) / n_err, 4) if n_err else None,
        "per_image_pq_quantiles": {q: round(float(np.percentile(per_img_pq, q)), 4)
                                   for q in (10, 25, 50, 75, 90)},
    }
    return res


def main():
    t0 = time.time()
    ann = discover()
    by_file = load(ann)
    n_pass = Counter(len(v) for v in by_file.values())
    print("passes per image:", dict(sorted(n_pass.items())), flush=True)

    names = sorted(by_file)
    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(names))
    holdout = set(names[i] for i in idx[:N_VAL])

    per_all, per_hold = {}, {}
    for fn, passes in by_file.items():
        ids = sorted(passes, key=str)
        if len(ids) < 2:
            continue
        rl = {pid: rles_of(passes[pid]) for pid in ids}
        pairs = []
        for a in ids:
            for b in ids:
                if a != b:
                    pairs.append(compare(rl[a], rl[b]))
        per_all[fn] = pairs
        if fn in holdout:
            per_hold[fn] = pairs

    results = {
        "n_files_total": len(by_file),
        "n_files_multipass": len(per_all),
        "passes_per_file": {str(k): v for k, v in sorted(n_pass.items())},
        "all_multipass": summarize("all multi-pass images", per_all),
        "holdout50": summarize("hold-out 50 (seed 1234), multi-pass only", per_hold),
        "our_model_reference": {"official_pq": 0.4325, "mean_iou_tp": 0.676,
                                 "near_miss_share_of_errors": 0.538},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(results, indent=2), flush=True)
    json.dump(results, open(os.path.join(OUT, "gpu_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
