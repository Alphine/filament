"""
Solar Filament Segmentation 2026 -- leaderboard PROBES: post-processing variants of the released
pipeline, to test hypotheses about the style of the (unseen) test labels. Kaggle GPU, inference only.

Why: the recipe's expected score (~0.41) is ~0.05 above its leaderboard score (0.36), and the
test images do not explain it (analysis in report.md), so the test labels may differ in style.
The only legitimate source of information about them is leaderboard feedback.

Variants (released 5-fold stage-A ensemble + 4-flip TTA and the released crop refiner):
  base    production: candidates p > 0.4, closing 3x3, area 400-40000, mean p >= 0.7; refiner
  strict  as base but mean p >= 0.8                       (annotators mark fewer filaments?)
  loose   p > 0.3, area >= 200, mean p >= 0.6             (annotators mark more small ones?)
  shrink  base masks eroded with a 5x5 ellipse (~2 px)    (narrower boundaries?)
  grow    base masks dilated with a 5x5 ellipse (~2 px)   (wider boundaries?), no overwriting
Stage-A runs once per image; every variant is derived from the same probability map.

Compute safety:
  * FAIL FAST: the first hold-out image's base counts must equal the released pipeline's known
    counts for that image, else stop.
  * INTEGRITY GATE before the test set: the 50-image hold-out base must reproduce TP/FP/FN
    400/239/211, else stop without running the 180 test images.
Outputs: probe_holdout.json (per-image official counts of every variant), submission_<variant>.csv
(validated) and probe_meta.json.
"""

import csv
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

import cv2
import numpy as np

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

INPUT, OUT = "/kaggle/input", "/kaggle/working"
REF = {"20131013185334Ch.jpeg": [7, 5, 5]}   # released pipeline, first hold-out image (TP, FP, FN)
VARIANTS = ("base", "strict", "loose", "shrink", "grow")
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def candidates(prob, thr, min_area, min_mean, max_area=fp.CAND_MAX_AREA):
    m = (prob > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fp.CAND_CLOSE_K,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros(lab.shape, np.int32)
    nid = 1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        comp = lab == i
        if prob[comp].mean() < min_mean:
            continue
        out[comp] = nid
        nid += 1
    return out


def morph(lab, op):
    """Erode or dilate every instance of a label map; dilation never overwrites other instances."""
    out = np.zeros_like(lab)
    for i in range(1, int(lab.max()) + 1):
        m = (lab == i).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.erode(m, K5) if op == "shrink" else cv2.dilate(m, K5)
        sel = m.astype(bool) & ((out == 0) if op == "grow" else True)
        if op == "grow":
            sel &= (lab == 0) | (lab == i)
        out[sel] = i
    return out


def variants(gray, prob, refiner, device):
    base_c = candidates(prob, fp.CAND_THR, fp.CAND_MIN_AREA, fp.CAND_MIN_MEAN_PROB)
    base = fp.refine_candidates(gray, base_c, refiner, device)
    return {
        "base": base,
        "strict": fp.refine_candidates(gray, candidates(prob, fp.CAND_THR, fp.CAND_MIN_AREA, 0.8), refiner, device),
        "loose": fp.refine_candidates(gray, candidates(prob, 0.3, 200, 0.6), refiner, device),
        "shrink": morph(base, "shrink"),
        "grow": morph(base, "grow"),
    }


t0 = time.time()
weights = sorted(glob.glob(f"{INPUT}/**/unet_fold*.pt", recursive=True))
assert len(weights) == 5, weights
refiner_path = next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
device = fp.default_device()
print("device", device, flush=True)
stage_a = fp.load_stage_a(weights, device)
refiner = fp.load_refiner(refiner_path, device)
per_file, dims = fm.load_annotations(ann_json, train_dir)

# ---- hold-out (fail fast on image 1, integrity gate on all 50)
hold = {v: [] for v in VARIANTS}
for k, n in enumerate(fm.holdout_names(list(per_file)), 1):
    g = fp.read_gray(os.path.join(train_dir, n))
    h, w = g.shape
    labs = variants(g, fp.predict_ensemble(stage_a, g, device), refiner, device)
    for v, lab in labs.items():
        c = fm.pq_counts(fp.label_map_to_rles(lab), per_file[n], h, w)
        hold[v].append({"name": n, "counts": [float(c[0]), *map(int, c[1:])]})
    if k == 1:
        got = hold["base"][0]["counts"][1:]
        assert n in REF and got == REF[n], f"FAIL FAST: base counts {got} on {n} differ from released {REF.get(n)}"
        print(f"fail-fast OK on {n}: {got} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(hold, open(f"{OUT}/probe_holdout.json", "w"))
tot = {v: np.sum([r["counts"] for r in rows], 0) for v, rows in hold.items()}
pq = {v: fm.pq(*t) for v, t in tot.items()}
print("hold-out PQ:", {v: round(p, 4) for v, p in pq.items()}, flush=True)
assert tot["base"][1:].astype(int).tolist() == [400, 239, 211], f"INTEGRITY: base {tot['base'][1:]} != 400/239/211"
print(f"integrity gate OK ({time.time() - t0:.0f}s) -- running the test set", flush=True)

# ---- test set
test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg")))
rows = {v: [] for v in VARIANTS}
for k, fname in enumerate(test_files, 1):
    stem = fname.rsplit(".", 1)[0]
    g = fp.read_gray(os.path.join(test_dir, fname))
    for v, lab in variants(g, fp.predict_ensemble(stage_a, g, device), refiner, device).items():
        rows[v] += [{"filament_id": f"{stem}_{i}", "segmentation_rle": r}
                    for i, r in enumerate(fp.label_map_to_rles(lab), 1)]
    if k % 30 == 0:
        print(f"test {k}/{len(test_files)} {time.time() - t0:.0f}s", flush=True)
stems = {f.rsplit(".", 1)[0] for f in test_files}
meta = {"holdout_pq": {v: round(p, 4) for v, p in pq.items()},
        "holdout_tp_fp_fn": {v: t[1:].astype(int).tolist() for v, t in tot.items()}, "test": {}}
for v in VARIANTS:
    meta["test"][v] = fp.validate_submission_rows(rows[v], stems)
    with open(f"{OUT}/submission_{v}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        wr.writeheader()
        wr.writerows(rows[v])
meta["seconds"] = round(time.time() - t0)
json.dump(meta, open(f"{OUT}/probe_meta.json", "w"), indent=1)
print("done", meta, flush=True)
