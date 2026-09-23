"""
Solar Filament Segmentation 2026 -- out-of-fold (OOF) predictions over the 657 pool
images, CPU-ONLY, NO TRAINING. One kernel per fold (FOLD = 1..5).

Motivation: every decision so far was measured on the 50-image hold-out (bootstrap SE
of PQ 0.017), which cannot resolve effects of about +-0.01; six post-processing tests
ended "not provable". Scoring each pool image with the stage-A model that did NOT train
on it gives 657 scored images (about 13x the hold-out; expected SE about 0.004-0.005)
for comparing future stage-A variants (for example a consensus/STAPLE training target).

Protocol
  * Splits reproduced exactly from training/filament_unet_ensemble.py: names sorted, hold-out
    = first 50 of default_rng(1234).permutation, pool = the remaining 657, folds =
    np.array_split(default_rng(99).permutation(len(pool)), 5). Checked: the hold-out equals
    filament_metric.holdout_names and folds are disjoint and cover the pool.
  * Fold F's model (unet_fold{F}.pt, single model, 4-flip TTA) predicts only fold F's
    validation images; candidates use the production thresholds (0.4 / area 400-40000 /
    mean p >= 0.7); the crop refiner redraws them.
  * Stored per image: official counts WITH refiner and WITHOUT refiner (raw stage-A
    candidates), number of passes, and (npz) the uint8 probability map, candidate and final
    label maps so post-processing can later be re-tested without inference.

Known biases (same for any stage-A variant scored this way, so comparisons stay fair):
  * the refiner was trained on all 657 pool images -> the WITH-refiner numbers are
    optimistic; the WITHOUT-refiner numbers are free of this;
  * each fold's validation images chose that fold's checkpoint (mild optimism);
  * a single model is used per image, not the 5-model ensemble, so absolute PQ is lower
    than the released pipeline's.
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

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

FOLD = int(os.environ.get("FOLD", "1"))  # 1..5
INPUT = "/kaggle/input"
fold_path = next(p for p in glob.glob(os.path.join(INPUT, "**", f"unet_fold{FOLD}.pt"), recursive=True))
refiner_path = next(p for p in glob.glob(os.path.join(INPUT, "**", "refiner_best.pt"), recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
device = fp.default_device()
model = fp.load_stage_a([fold_path], device)
refiner = fp.load_refiner(refiner_path, device)
per_file, dims = fm.load_annotations(ann_json, train_dir)

# ---- reproduce the training splits exactly
names = sorted(per_file)
assert len(names) == 707, len(names)
idx = np.random.default_rng(1234).permutation(len(names))
holdout = [names[i] for i in idx[:50]]
pool = [names[i] for i in idx[50:]]
assert holdout == fm.holdout_names(list(per_file)), "hold-out mismatch"
folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), 5)
all_idx = np.concatenate(folds)
assert len(set(all_idx.tolist())) == len(pool) == 657
val = [pool[i] for i in folds[FOLD - 1]]
assert not set(val) & set(holdout)
print(f"fold {FOLD}: {len(val)} validation images; model {fold_path}", flush=True)

results, store = [], {}
t0 = time.time()
for k, name in enumerate(val, 1):
    t_img = time.time()
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    passes = per_file[name]
    prob = fp.predict_ensemble(model, g, device)
    cand = fp.extract_candidates(prob)
    lab = fp.refine_candidates(g, cand, refiner, device)
    with_ref = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
    no_ref = fm.pq_counts(fp.label_map_to_rles(cand), passes, h, w)
    results.append({"name": name, "fold": FOLD, "n_pass": len(passes),
                    "with_refiner": [float(with_ref[0]), *map(int, with_ref[1:])],
                    "without_refiner": [float(no_ref[0]), *map(int, no_ref[1:])],
                    "seconds": round(time.time() - t_img)})
    key = name.rsplit(".", 1)[0]
    store[f"{key}__prob"] = (prob * 255).round().astype(np.uint8)
    store[f"{key}__cand"] = cand.astype(np.int16)
    store[f"{key}__lab"] = lab.astype(np.int16)
    if k % 5 == 0 or k == len(val):
        print(f"{k}/{len(val)} {time.time() - t0:.0f}s last={results[-1]['with_refiner']}", flush=True)
        json.dump({"fold": FOLD, "status": "running", "results": results},
                  open(f"/kaggle/working/oof_fold{FOLD}.json", "w"))
    if k % 40 == 0:
        np.savez_compressed(f"/kaggle/working/oof_preds_fold{FOLD}.npz", **store)

json.dump({"fold": FOLD, "status": "done", "elapsed_s": round(time.time() - t0), "n_images": len(val),
           "results": results}, open(f"/kaggle/working/oof_fold{FOLD}.json", "w"))
np.savez_compressed(f"/kaggle/working/oof_preds_fold{FOLD}.npz", **store)
print("done", flush=True)
