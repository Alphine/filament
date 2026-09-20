"""Panoptic Quality exactly as defined on the competition page, per annotator pass.

    PQ = sum(IoU of TP pairs) / (|TP| + 0.5 |FP| + 0.5 |FN|),   TP: IoU > 0.5

Every prediction set is scored against each annotator pass of an image separately
(passes = annotations sharing a file name but not an `image_id`); counts and IoU sums
are pooled over all passes and images before dividing.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util


def load_annotations(annotation_json: str, image_dir: str | None = None):
    """-> (per_file, dims): per_file[name] = {pass_id: [annotation, ...]}, dims[name] = (h, w)."""
    with open(annotation_json) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    per_file: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    dims = {}
    for a in coco["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            continue
        name = os.path.basename(im["file_name"])
        if image_dir is not None and not os.path.exists(os.path.join(image_dir, name)):
            continue
        per_file[name][a["image_id"]].append(a)
        dims[name] = (im["height"], im["width"])
    return {k: dict(v) for k, v in per_file.items()}, dims


def _gt_rles(anns: list[dict], h: int, w: int) -> list[dict]:
    out = []
    for a in anns:
        seg = a.get("segmentation")
        if not seg:
            continue
        r = mask_util.frPyObjects(seg, h, w)
        out.append(mask_util.merge(r) if isinstance(r, list) else r)
    return out


def pq_counts(pred_counts: list[str], passes: dict, h: int, w: int) -> tuple[float, int, int, int]:
    """(sum_iou, tp, fp, fn) of one image, pooled over its annotator passes."""
    pred = [{"size": [h, w], "counts": c.encode("ascii")} for c in pred_counts]
    s = tp = fp = fn = 0
    for anns in passes.values():
        gt = _gt_rles(anns, h, w)
        if not gt:
            fp += len(pred)
            continue
        if not pred:
            fn += len(gt)
            continue
        iou = np.asarray(mask_util.iou(pred, gt, [0] * len(gt)), dtype=np.float64)
        hit = iou > 0.5
        s += float(iou[hit].sum())
        tp += int(hit.sum())
        fp += int((hit.sum(1) == 0).sum())
        fn += int((hit.sum(0) == 0).sum())
    return s, tp, fp, fn


def pq(sum_iou: float, tp: int, fp: int, fn: int) -> float:
    den = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / den if den else 0.0


def evaluate(predictions: dict[str, list[str]], annotation_json: str, names: list[str] | None = None,
             image_dir: str | None = None) -> dict:
    """predictions[name] = list of RLE counts strings. Returns pooled PQ and per-image counts."""
    per_file, dims = load_annotations(annotation_json, image_dir)
    names = names if names is not None else sorted(per_file)
    per_image, tot = {}, np.zeros(4)
    for n in names:
        h, w = dims[n]
        r = pq_counts(predictions.get(n, []), per_file[n], h, w)
        per_image[n] = {"sum_iou": r[0], "tp": r[1], "fp": r[2], "fn": r[3]}
        tot += r
    return {"pq": pq(*tot), "sum_iou": tot[0], "tp": int(tot[1]), "fp": int(tot[2]),
            "fn": int(tot[3]), "per_image": per_image}


def holdout_names(all_names: list[str], n: int = 50, seed: int = 1234) -> list[str]:
    """The fixed hold-out split: never used for training or checkpoint selection."""
    names = sorted(all_names)
    idx = np.random.default_rng(seed).permutation(len(names))
    return [names[i] for i in idx[:n]]
