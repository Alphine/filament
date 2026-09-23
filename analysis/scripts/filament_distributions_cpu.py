"""
Solar Filament Segmentation 2026 -- Dice/IoU score distributions and
one-to-many / many-to-one fragmentation analysis, CPU-ONLY, NO TRAINING.

Motivation: the competition's official evaluation rubric weighs quantitative
comparison (70%) on more than the pooled PQ number -- it explicitly asks for
the *distribution* of Dice/IoU scores and the distribution of one-to-many /
many-to-one relations (fragmentation and over-merging). Existing analysis
files (analysis/results/fpdiag/*.json) only store the single best IoU per
prediction/per ground-truth instance, which is enough for TP/FP/FN counts but
not for detecting fragmentation (one ground-truth instance overlapped by
several predictions, or one prediction spanning several ground-truth
instances) -- that needs the full prediction x ground-truth overlap matrix
per image/pass, so this re-runs inference on the 50-image hold-out.

EXACT final pipeline (stage-A 5-fold + TTA, candidates thr=0.4/area
400-40000/min_mean_prob=0.7, crop refiner) -- checkpoints are loaded via
kernel_sources (the original training kernels' outputs), NOT retrained. Same
sacred hold-out (seed 1234, 50 images).

Integrity reference: official PQ = 0.4325 (TP=400, FP=239, FN=211).
Run as a Kaggle CPU kernel with:
  kernel_sources = ["abukayyisahbasyid/filament-unet-ensemble-5fold",
                     "abukayyisahbasyid/filament-crop-refiner"]
  competition_sources = ["filament-segmentation-2026"]
Writes analysis/results/distributions_summary.json (this exact file, copied
from /kaggle/working after the run).
"""

import glob
import json
import os
import subprocess
import sys
import time

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                     "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

import numpy as np
import pycocotools.mask as mu

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

INPUT = "/kaggle/input"
folds = sorted(glob.glob(os.path.join(INPUT, "**", "unet_fold*.pt"), recursive=True))
refiner_path = next(p for p in glob.glob(os.path.join(INPUT, "**", "refiner_best.pt"), recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
device = fp.default_device()
pipe = fp.Pipeline(fp.load_stage_a(folds, device), fp.load_refiner(refiner_path, device), device)
per_file, dims = fm.load_annotations(ann_json, train_dir)
names = fm.holdout_names(list(per_file))
print("n_holdout:", len(names), flush=True)

FRAG_THR = 0.1  # "meaningful overlap" threshold for fragmentation counting, separate from the official TP threshold 0.5

dice_tp, iou_tp = [], []
# per ground-truth instance: how many predictions overlap it meaningfully (IoU > FRAG_THR)
#   '0' = missed entirely, '1_hit' = clean 1:1 TP, '1_miss' = one overlapping pred but IoU<=0.5 (near-miss),
#   '>=2' = split across multiple predictions (one-to-many / over-segmentation)
frag_gt_counts = {"0": 0, "1_hit": 0, "1_miss": 0, ">=2": 0}
# symmetric, per prediction: '>=2' = one prediction spanning multiple GT (many-to-one / under-segmentation/merging)
frag_pred_counts = {"0": 0, "1_hit": 0, "1_miss": 0, ">=2": 0}

official_s = official_tp = official_fp = official_fn = 0
t0 = time.time()
for k, name in enumerate(names, 1):
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    lab = pipe.segment(g)
    pred_counts = fp.label_map_to_rles(lab)
    pred = [{"size": [h, w], "counts": c.encode("ascii")} for c in pred_counts]

    # official-metric integrity check (must reproduce the known hold-out PQ)
    s, tp, fpn, fnn = fm.pq_counts(pred_counts, per_file[name], h, w)
    official_s += s; official_tp += tp; official_fp += fpn; official_fn += fnn

    for anns in per_file[name].values():
        gt = fm._gt_rles(anns, h, w)
        if not pred or not gt:
            continue
        iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)), dtype=np.float64)  # [n_pred, n_gt]

        tp_mask = iou > 0.5
        for v in iou[tp_mask]:
            iou_tp.append(float(v))
            dice_tp.append(float(2 * v / (1 + v)))

        frag = iou > FRAG_THR
        n_preds_per_gt = frag.sum(axis=0)
        n_gts_per_pred = frag.sum(axis=1)
        best_iou_per_gt = iou.max(axis=0) if iou.size else np.zeros(len(gt))
        best_iou_per_pred = iou.max(axis=1) if iou.size else np.zeros(len(pred))

        for j, n in enumerate(n_preds_per_gt):
            if n == 0:
                frag_gt_counts["0"] += 1
            elif n == 1:
                frag_gt_counts["1_hit" if best_iou_per_gt[j] > 0.5 else "1_miss"] += 1
            else:
                frag_gt_counts[">=2"] += 1
        for i, n in enumerate(n_gts_per_pred):
            if n == 0:
                frag_pred_counts["0"] += 1
            elif n == 1:
                frag_pred_counts["1_hit" if best_iou_per_pred[i] > 0.5 else "1_miss"] += 1
            else:
                frag_pred_counts[">=2"] += 1

    if k % 10 == 0 or k == len(names):
        print(f"{k}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

official_pq = fm.pq(official_s, official_tp, official_fp, official_fn)
print("INTEGRITY official_pq =", official_pq, "(expected ~0.4325)", flush=True)


def pct(a, qs=(5, 25, 50, 75, 95)):
    return {str(q): float(np.percentile(a, q)) for q in qs} if len(a) else {}


out = {
    "n_holdout_images": len(names),
    "n_tp_pairs": len(iou_tp),
    "integrity_official_pq": official_pq,
    "integrity_official_tp_fp_fn": [official_tp, official_fp, official_fn],
    "iou_tp_distribution": {"values": iou_tp, "mean": float(np.mean(iou_tp)) if iou_tp else None,
                             "percentiles": pct(np.array(iou_tp))},
    "dice_tp_distribution": {"values": dice_tp, "mean": float(np.mean(dice_tp)) if dice_tp else None,
                              "percentiles": pct(np.array(dice_tp))},
    "fragmentation_threshold_iou": FRAG_THR,
    "gt_fragmentation": frag_gt_counts,
    "pred_fragmentation": frag_pred_counts,
}
json.dump(out, open("/kaggle/working/distributions_summary.json", "w"), indent=1)
print("done", flush=True)
