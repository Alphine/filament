"""Breakdown: where the score is lost, using the organizers' OFFICIAL METRIC.

PQ = sum(IoU TP) / (|TP| + 0.5|FP| + 0.5|FN|)

There are three ways to lose score, and each has a different remedy:

  FP  predictions that hit no GT at all. Remedy: precision -- do not emit
      instances we are not confident about. This is what the two-stage
      approach (high-confidence detection first) targets.
  FN  GT instances that no prediction touches. Remedy: recall -- and the
      earlier diagnosis showed that the missed ones are small and dim.
  IoU shape quality on pairs that are already matched. Remedy: finer
      segmentation, e.g. segmenting inside a full-resolution crop.

This script computes the CEILING of each: the PQ obtained if one loss source
is removed perfectly while the other two are left as they are. That gives an
honest upper bound for each improvement direction, so we know the size of the
prize BEFORE burning GPU time, not after.

Important caveat: the stored probability maps come from a SINGLE U-Net model
(Kaggle score 0.24), not from the best ensemble+TTA (0.34). The absolute
numbers are therefore outdated. What matters here is the PROPORTION of loss
across sources, and that proportion is far more stable across models than
the absolute values.
"""
import io
import json
import os
import sys

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from official_metric_check import (get_overlap_matrices, poly_to_rle,
                                   rles_to_layers)
from tune_postproc import instances

PROB_DIR = os.path.join(HERE, "val_probs")
ANN = os.path.join(HERE, "data", "train_annotations.json")
IOU_THR = 0.5


def collect(gt_by_annimg, pred_by_img):
    """Collect per-annotator-image statistics using the official rules."""
    tp_iou, fp, fn = [], 0, 0
    for annimg, gt_rles in gt_by_annimg.items():
        image_id = annimg.split("-", maxsplit=1)[1]
        pred_rles = pred_by_img.get(image_id, [])
        if not gt_rles:
            fp += len(pred_rles)
            continue
        if not pred_rles:
            fn += len(gt_rles)
            continue
        iou = get_overlap_matrices(torch.from_numpy(rles_to_layers(gt_rles)),
                                   torch.from_numpy(rles_to_layers(pred_rles)))
        hit = iou > IOU_THR
        tp_iou.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    return tp_iou, fp, fn


def pq(sum_iou, tp, fp, fn):
    den = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / den if den > 0 else 0.0


def main():
    val_names = open(os.path.join(PROB_DIR, "val_names.txt")).read().split("\n")
    stems = {fn.rsplit(".", 1)[0] for fn in val_names}

    d = json.load(io.open(ANN, encoding="utf-8"))
    imgs = {i["id"]: i for i in d["images"]}
    gt_by_annimg = {}
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        if im["file_name"].rsplit(".", 1)[0] not in stems or not a.get("segmentation"):
            continue
        gt_by_annimg.setdefault(a["image_id"], []).append(
            poly_to_rle(a["segmentation"], im["height"], im["width"]))

    params = dict(thr=0.4, min_area=600)
    pred_by_img = {}
    for fn in val_names:
        stem = fn.rsplit(".", 1)[0]
        lab = instances(cv2.imread(os.path.join(PROB_DIR, stem + ".png"),
                                   cv2.IMREAD_GRAYSCALE), **params)
        rles = []
        for i in range(1, int(lab.max()) + 1):
            m = np.asfortranarray((lab == i).astype(np.uint8))
            if m.sum():
                rles.append(mask_util.encode(m)["counts"].decode("utf-8"))
        pred_by_img[stem] = rles

    tp_iou, fp, fn = collect(gt_by_annimg, pred_by_img)
    tp = len(tp_iou)
    S = sum(tp_iou)
    base = pq(S, tp, fp, fn)

    bar = "=" * 66
    print(f"\n{bar}\nCURRENT STATE (official metric)\n{bar}")
    print(f"  TP={tp}  FP={fp}  FN={fn}   sum(IoU TP)={S:.1f}")
    print(f"  Mean IoU over TP pairs = {S/tp:.3f}")
    print(f"  PQ = {base:.4f}")

    print(f"\n{bar}\nCEILING FOR EACH IMPROVEMENT DIRECTION\n{bar}")
    rows = [
        ("remove ALL false positives", pq(S, tp, 0, fn),
         "perfect precision: emit only correct instances"),
        ("remove ALL false negatives", pq(S, tp, fp, 0),
         "perfect recall: no GT instance is missed"),
        ("TP-pair IoU set to 1.0", pq(float(tp), tp, fp, fn),
         "perfect shape on the pairs that already match"),
        ("remove FP AND FN", pq(S, tp, 0, 0),
         "perfect detection, shape left as is"),
    ]
    for name, v, why in rows:
        print(f"  {name:<28} PQ={v:.4f}  ({100*(v-base)/base:+5.1f}%)")
        print(f"  {'':<28} {why}")

    print(f"\n{bar}\nINTERPRETATION\n{bar}")
    best = max(rows[:3], key=lambda r: r[1])
    print(f"  Direction with the highest ceiling: {best[0]} -> PQ {best[1]:.4f}")
    print(f"  For reference: our best Kaggle score is 0.34; the public cluster is 0.55")
    print(f"  FP:TP ratio = {fp/tp:.2f}  (for every correct prediction, "
          f"{fp/tp:.2f} wrong predictions)")

    json.dump(dict(tp=tp, fp=fp, fn=fn, sum_iou=float(S),
                   mean_iou_tp=float(S/tp), pq=float(base),
                   ceiling_no_fp=float(pq(S, tp, 0, fn)),
                   ceiling_no_fn=float(pq(S, tp, fp, 0)),
                   ceiling_perfect_iou=float(pq(float(tp), tp, fp, fn)),
                   ceiling_no_fp_fn=float(pq(S, tp, 0, 0)),
                   params=params, note="prob maps from a single U-Net (score 0.24)"),
              open(os.path.join(HERE, "where_score_lost.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
