"""
Solar Filament Segmentation 2026 -- near-limb instance dropping: extended threshold grid and the
test submissions, CPU-ONLY, no network.

The first limb study (filament_limb_mask_cpu.py) found drop@0.90 (remove instances whose centroid
lies beyond 0.90 R of the disk centre) actionable: +0.0059 out of fold (CI95 +0.004..+0.008),
+0.0052 on the hold-out ensemble; the gain was still rising at the edge of the grid. This kernel
  1. scores drop@t for T_GRID on the stored out-of-fold and hold-out label maps (same data and
     disk estimate as before; t is then chosen by split-half CV locally), and
  2. writes submission_drop@t.csv for every t: the released pipeline's test predictions
     (submission_base.csv of the leaderboard-probe run: the released weights and code re-run on a
     T4; it reproduced the hold-out counts 400/239/211 exactly) with near-limb instances removed,
     validated with validate_submission_rows.
"""

import csv
import glob
import json
import os
import time

import cv2
import numpy as np
import pycocotools.mask as mu

import filament_metric as fm
import filament_pipeline as fp

INPUT, OUT = "/kaggle/input", "/kaggle/working"
T_GRID = (0.75, 0.80, 0.85, 0.88, 0.90)
t0 = time.time()
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)


def disk(gray):
    _, b = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == i).astype(np.uint8)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    m = m | (1 - ff)
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(m.sum() / np.pi))


# ---- 1. extended grid on the stored label maps
stores = {}
for p in glob.glob(f"{INPUT}/**/*.npz", recursive=True):
    tag = "holdout" if os.path.basename(p).startswith("preds_chunk") else "oof"
    for k in np.load(p).files:
        if k.endswith("__lab"):
            stores[(tag, k[:-5])] = p
results = []
for k, (tag, stem) in enumerate(sorted(stores), 1):
    name = stem + ".jpeg"
    lab = np.load(stores[(tag, stem)])[stem + "__lab"].astype(np.int32)
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    cx, cy, R = disk(g)
    ids = [i for i in range(1, int(lab.max()) + 1) if (lab == i).any()]
    dist = {}
    for i in ids:
        ys, xs = np.nonzero(lab == i)
        dist[i] = np.sqrt((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) / R
    rec = {"name": name, "set": tag, "rules": {}}
    for t in T_GRID:
        keep = lab.copy()
        for i in ids:
            if dist[i] > t:
                keep[keep == i] = 0
        s, tp, fpn, fnn = fm.pq_counts(fp.label_map_to_rles(keep), per_file[name], h, w)
        rec["rules"][f"drop@{t}"] = [float(s), int(tp), int(fpn), int(fnn)]
    results.append(rec)
    if k % 100 == 0:
        print(f"grid {k}/{len(stores)} {time.time() - t0:.0f}s", flush=True)
json.dump({"t_grid": T_GRID, "results": results}, open(f"{OUT}/limb_drop_ext.json", "w"))

# ---- 2. test submissions
base_csv = next(p for p in glob.glob(f"{INPUT}/**/submission_base.csv", recursive=True))
rows_by_stem = {}
with open(base_csv) as f:
    for r in csv.DictReader(f):
        rows_by_stem.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append(r)
test_stems = {f.rsplit(".", 1)[0] for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg"))}
test_files = {f.rsplit(".", 1)[0]: f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg"))}
kept = {t: [] for t in T_GRID}
n_base = sum(len(v) for v in rows_by_stem.values())
for stem, rows in rows_by_stem.items():
    cx, cy, R = disk(fp.read_gray(os.path.join(test_dir, test_files[stem])))
    for r in rows:
        m = mu.decode({"size": [2048, 2048], "counts": r["segmentation_rle"].encode("ascii")})
        ys, xs = np.nonzero(m)
        d = np.sqrt((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) / R
        for t in T_GRID:
            if d <= t:
                kept[t].append(r)
meta = {"base_rows": n_base, "test": {}}
for t in T_GRID:
    rows = kept[t]
    by = {}
    for r in rows:                                   # renumber ids per image: 1..n
        by.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append(r["segmentation_rle"])
    out = [{"filament_id": f"{s}_{i}", "segmentation_rle": rle} for s, rl in by.items() for i, rle in enumerate(rl, 1)]
    meta["test"][f"drop@{t}"] = fp.validate_submission_rows(out, test_stems)
    with open(f"{OUT}/submission_drop@{t}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        w.writeheader()
        w.writerows(out)
meta["seconds"] = round(time.time() - t0)
json.dump(meta, open(f"{OUT}/limb_drop_meta.json", "w"), indent=1)
print("done", meta, flush=True)
