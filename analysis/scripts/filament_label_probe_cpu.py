"""
Solar Filament Segmentation 2026 -- label-convention probes: mask boundary offset and small-instance
dropping, CPU-ONLY, no inference, no network.

Every variant we submitted reads 0.36 on the leaderboard although the hold-out says 0.438; the gap is
far outside sampling noise, and the leading explanation is a different labelling convention in the
test set (wider/narrower masks, or small filaments left unlabelled). A leaderboard probe can only
separate these if its effect under the NULL (test labelled like training) is known exactly. This kernel
  1. scores each probe on the stored out-of-fold (657) and hold-out (50) label maps of the released
     pipeline with the near-limb filter (drop@0.88), against the training annotations, and
  2. writes the same probes as test submissions, built from submission_drop@0.88.csv (the current
     best submission), validated with validate_submission_rows. Nothing is submitted here.

Probes (all applied to the final instance label map):
  dil1 / dil2   every instance grown by 1 / 2 px per side (3x3), never overwriting another instance
  ero1 / ero2   every instance shrunk by 1 / 2 px per side (3x3); instances that vanish are dropped
  small@A       instances with area < A px removed
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
SMALL_GRID = (500, 700, 900, 1200, 1600)
K3 = np.ones((3, 3), np.uint8)
t0 = time.time()


def dilate(lab, n):
    out = lab.astype(np.uint16)
    for _ in range(n):
        out = np.where(out > 0, out, cv2.dilate(out, K3))
    return out.astype(np.int32)


def erode(lab, n):
    out = lab.astype(np.uint16)
    for _ in range(n):
        same = (cv2.erode(out, K3) == out) & (cv2.dilate(out, K3) == out)   # all 8 neighbours carry the same id
        out = np.where(same, out, 0).astype(np.uint16)
    return out.astype(np.int32)


def variants(lab):
    """name -> label map; `lab` already has the near-limb filter applied."""
    out = {"base": lab, "dil1": dilate(lab, 1), "dil2": dilate(lab, 2), "ero1": erode(lab, 1), "ero2": erode(lab, 2)}
    areas = np.bincount(lab.ravel())
    for a in SMALL_GRID:
        small = np.flatnonzero(areas < a)
        out[f"small@{a}"] = np.where(np.isin(lab, small), 0, lab)
    return out, [int(x) for x in areas[1:] if x]


# ---- self-test of the morphology on a synthetic map (fail fast, before any data is read)
_t = np.zeros((40, 40), np.int32)
_t[10:20, 10:20] = 1
_t[10:20, 21:30] = 2
_d, _e = dilate(_t, 1), erode(_t, 1)
assert (_d == 1).sum() == 12 * 11 and (_d[10:20, 20] == 2).all() and (_d[_t > 0] == _t[_t > 0]).all(), "dilate self-test"
assert (_e == 1).sum() == 8 * 8 and (_e == 2).sum() == 8 * 7, "erode self-test"

ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)

# ---- 1. probes on the stored label maps
stores, files = {}, {}
for p in glob.glob(f"{INPUT}/**/*.npz", recursive=True):
    tag = "holdout" if os.path.basename(p).startswith("preds_chunk") else "oof"
    files[p] = np.load(p)
    for k in files[p].files:
        if k.endswith("__lab"):
            stores[(tag, k[:-5])] = p
assert sum(t == "oof" for t, _ in stores) == 657 and sum(t == "holdout" for t, _ in stores) == 50, len(stores)
results = []
for k, (tag, stem) in enumerate(sorted(stores), 1):
    name = stem + ".jpeg"
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    lab = fp.drop_near_limb(files[stores[(tag, stem)]][stem + "__lab"].astype(np.int32), g)
    var, areas = variants(lab)
    rec = {"name": name, "set": tag, "n_pass": len(per_file[name]), "areas": areas, "rules": {}}
    for v, m in var.items():
        s, tp, fpn, fnn = fm.pq_counts(fp.label_map_to_rles(m), per_file[name], h, w)
        rec["rules"][v] = [float(s), int(tp), int(fpn), int(fnn)]
    results.append(rec)
    if k % 50 == 0 or k == len(stores):
        print(f"probe {k}/{len(stores)} {time.time() - t0:.0f}s", flush=True)
        json.dump({"status": "running" if k < len(stores) else "done", "small_grid": SMALL_GRID, "results": results},
                  open(f"{OUT}/label_probe.json", "w"))

# ---- 2. the same probes as test submissions
base_csv = next(p for p in glob.glob(f"{INPUT}/**/submission_drop@0.88.csv", recursive=True))
rows_by_stem = {}
with open(base_csv) as f:
    for r in csv.DictReader(f):
        rows_by_stem.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append(r["segmentation_rle"])
test_stems = {f.rsplit(".", 1)[0] for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg"))}
out_rows = {}
for stem, rles in sorted(rows_by_stem.items()):
    lab = np.zeros((2048, 2048), np.int32)
    for i, c in enumerate(rles, 1):
        m = mu.decode({"size": [2048, 2048], "counts": c.encode("ascii")}).astype(bool)
        assert not (lab[m] > 0).any(), f"overlapping instances in {stem}"
        lab[m] = i
    var, _ = variants(lab)
    assert fp.label_map_to_rles(var["base"]) == rles, f"round trip changed {stem}"
    for v, m in var.items():
        out_rows.setdefault(v, []).extend({"filament_id": f"{stem}_{i}", "segmentation_rle": c}
                                          for i, c in enumerate(fp.label_map_to_rles(m), 1))
meta = {"base_csv": base_csv, "test": {}}
for v, rows in out_rows.items():
    meta["test"][v] = fp.validate_submission_rows(rows, test_stems)
    meta["test"][v]["mean_area"] = round(float(np.mean([mu.area({"size": [2048, 2048], "counts": r["segmentation_rle"].encode("ascii")})
                                                         for r in rows])), 1)
    if v != "base":
        with open(f"{OUT}/submission_{v}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
            wr.writeheader()
            wr.writerows(rows)
meta["seconds"] = round(time.time() - t0)
json.dump(meta, open(f"{OUT}/label_probe_meta.json", "w"), indent=1)
print("done", meta, flush=True)
