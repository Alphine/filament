"""
Solar Filament Segmentation 2026 -- off-limb / near-limb masking of predicted instances,
CPU-ONLY, no network: reuses stored label maps.

Motivation: the audit of the existing hold-out diagnostics found more "phantom" false positives
(touching no ground truth) near the solar limb, and a public strategy write-up masks the
off-disk region. Not tested so far.

Data: final label maps (stage-A + refiner) stored by earlier runs --
  out-of-fold: oof_preds_fold{1..5}.npz (single model per image, 657 pool images)
  hold-out:    preds_chunk{0..3}.npz   (released 5-model ensemble, 50 images)
Disk per image: Otsu threshold of the image, largest component, holes filled; centre = its
centroid, radius R = sqrt(area / pi).
Rules, for t in T_GRID:
  drop@t  remove instances whose centroid lies farther than t*R from the disk centre
  clip@t  remove the pixels of every instance beyond t*R; drop instances that become empty
Stored per image: official counts (pooled over passes) for base and every rule, the disk, and
per-instance centroid distance / R. Integrity: base counts must equal the stored OOF counts.
"""

import glob
import json
import os
import time

import cv2
import numpy as np

import filament_metric as fm
import filament_pipeline as fp

INPUT, OUT = "/kaggle/input", "/kaggle/working"
T_GRID = (0.90, 0.93, 0.95, 0.97, 1.00)
t0 = time.time()

ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)
stores = {}
for p in glob.glob(f"{INPUT}/**/*.npz", recursive=True):
    tag = "holdout" if os.path.basename(p).startswith("preds_chunk") else "oof"
    z = np.load(p)
    for k in z.files:
        if k.endswith("__lab"):
            stores[(tag, k[:-5])] = p
ref = {}
for p in glob.glob(f"{INPUT}/**/oof_fold*.json", recursive=True):
    for r in json.load(open(p))["results"]:
        ref[r["name"]] = r["with_refiner"]
print(f"label maps: {sum(t == 'oof' for t, _ in stores)} OOF, {sum(t == 'holdout' for t, _ in stores)} hold-out; "
      f"OOF reference counts {len(ref)}", flush=True)


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


def counts(lab, name, h, w):
    s, tp, fpn, fnn = fm.pq_counts(fp.label_map_to_rles(lab), per_file[name], h, w)
    return [float(s), int(tp), int(fpn), int(fnn)]


results = []
items = sorted(stores)
for k, (tag, stem) in enumerate(items, 1):
    name = stem + ".jpeg"
    if name not in per_file:
        continue
    lab = np.load(stores[(tag, stem)])[stem + "__lab"].astype(np.int32)
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    cx, cy, R = disk(g)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    ids = [i for i in range(1, int(lab.max()) + 1) if (lab == i).any()]
    dist = {}
    for i in ids:
        ys, xs = np.nonzero(lab == i)
        dist[i] = float(np.sqrt((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) / R)
    rec = {"name": name, "set": tag, "n_pass": len(per_file[name]), "disk": [cx, cy, R],
           "inst_dist": [round(dist[i], 4) for i in ids], "base": counts(lab, name, h, w), "rules": {}}
    if tag == "oof":
        rec["base_matches_stored"] = rec["base"][1:] == ref[name][1:]
    for t in T_GRID:
        drop = lab.copy()
        for i in ids:
            if dist[i] > t:
                drop[drop == i] = 0
        rec["rules"][f"drop@{t}"] = counts(drop, name, h, w)
        clip = lab.copy()
        clip[rr > t] = 0
        rec["rules"][f"clip@{t}"] = counts(clip, name, h, w)
    results.append(rec)
    if k % 50 == 0:
        print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        json.dump({"status": "running", "t_grid": T_GRID, "results": results}, open(f"{OUT}/limb_mask.json", "w"))
ok = sum(r.get("base_matches_stored", True) for r in results)
print(f"integrity: base counts equal stored OOF counts on {ok}/{len(results)} images", flush=True)
json.dump({"status": "done", "t_grid": T_GRID, "seconds": round(time.time() - t0), "results": results},
          open(f"{OUT}/limb_mask.json", "w"))
print("done", flush=True)
