"""
Solar Filament Segmentation 2026 -- candidate/refiner threshold grid on the hold-out,
stored per image so it can be re-tuned for the single-pass setting, CPU-ONLY.

Motivation: on the 31 single-pass hold-out images the pipeline scores PQ 0.396
(multi-pass 0.460), close to the leaderboard's 0.36, so the test set may resemble the
single-pass setting. Thresholds were tuned on all 50 images; a single annotator marks
fewer objects, so the best thresholds there may differ. An earlier all-image
threshold-tightening simulation gained nothing out of sample (-0.0005).

No stage-A inference: the stage-A probability maps (uint8) saved by
filament_merge_cpu.py are reused (kernel_sources = the four filament-merge-chunk
kernels). For every probability threshold the connected components are extracted once
and each component is redrawn by the crop refiner once; the area / mean-probability
filters and the refiner threshold are then applied without extra network calls.
The exact final pipeline assembly is kept (later ids overwrite earlier ones).

Grid: CAND_THR x MIN_MEAN_PROB x MIN_AREA x REFINE_THR. Per image and configuration it
stores the official counts (sum IoU, TP, FP, FN), pooled over the image's passes.
Integrity: the production configuration (0.4, 0.7, 400, 0.5) must reproduce the
stored candidate maps and the pooled TP/FP/FN 400/239/211 (checked locally).
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

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
THRS = (0.30, 0.35, 0.40, 0.45, 0.50)
MEANS = (0.60, 0.65, 0.70, 0.75, 0.80)
AREAS = (200, 400, 800)
RTHRS = (0.3, 0.4, 0.5, 0.6)
PROD = (0.40, 0.70, 400, 0.5)

INPUT = "/kaggle/input"
refiner_path = next(p for p in glob.glob(os.path.join(INPUT, "**", "refiner_best.pt"), recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
device = fp.default_device()
refiner = fp.load_refiner(refiner_path, device)
per_file, dims = fm.load_annotations(ann_json, train_dir)
names = fm.holdout_names(list(per_file))
names = [n for k, n in enumerate(names) if k % CHUNK_TOTAL == CHUNK_INDEX]
store = {}
for p in glob.glob(os.path.join(INPUT, "**", "preds_chunk*.npz"), recursive=True):
    z = np.load(p)
    for k in z.files:
        store[k] = p
print(f"chunk {CHUNK_INDEX}: {len(names)} images; stored arrays {len(store)}", flush=True)
clahe = cv2.createCLAHE(2.0, (8, 8))
close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fp.CAND_CLOSE_K,) * 2) if fp.CAND_CLOSE_K else None


def load(key):
    return np.load(store[key])[key]


def components(prob, thr):
    """All components of prob > thr with area >= min(AREAS) and <= max area, in pipeline order."""
    m = (prob > thr).astype(np.uint8)
    if close_k is not None:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min(AREAS) or area > fp.CAND_MAX_AREA:
            continue
        comp = lab == i
        mp = float(prob[comp].mean())
        if mp < min(MEANS):
            continue
        out.append({"mask": comp, "area": area, "mean": mp})
    return out


@torch.no_grad()
def refine_prob(gray, comp):
    """Refiner probability for one component: (x0, y0, x1, y1, full-size float prob crop)."""
    h, w = gray.shape
    y0_, y1_, x0_, x1_ = fp._bbox(comp)
    bbox = [x0_, y0_, x1_ - x0_, y1_ - y0_]
    seed = fp.seed_box(h, w, bbox)
    x0, y0, x1, y1 = fp.square_bounds(bbox, h, w)
    cs = fp.REFINE_CROP
    img = cv2.resize(gray[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR)
    sd = cv2.resize(seed[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
    x = np.stack([img, clahe.apply(img), sd * 255], 0).astype(np.float32) / 255.0
    x = torch.from_numpy((x - fp.IMAGENET_MEAN) / fp.IMAGENET_STD)[None].to(device)
    pr = torch.sigmoid(refiner(x)).float().cpu().numpy()[0, 0]
    return x0, y0, x1, y1, pr


def assemble(h, w, refined, keep, rthr):
    out = np.zeros((h, w), np.int32)
    for nid, j in enumerate(keep, 1):
        x0, y0, x1, y1, pr = refined[j]
        crop_mask = (pr > rthr).astype(np.uint8)
        full = cv2.resize(crop_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        out[y0:y1, x0:x1][full.astype(bool)] = nid
    return out


results = []
t0 = time.time()
for k, name in enumerate(names, 1):
    t_img = time.time()
    key = name.rsplit(".", 1)[0]
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    passes = per_file[name]
    prob = load(f"{key}__prob").astype(np.float32) / 255.0
    rec = {"name": name, "n_pass": len(passes), "counts": {}}
    for thr in THRS:
        comps = components(prob, thr)
        refined = [refine_prob(g, c["mask"]) for c in comps]
        if thr == PROD[0]:
            prod_keep = [j for j, c in enumerate(comps) if c["area"] >= PROD[2] and c["mean"] >= PROD[1]]
            cand_rec = np.zeros((h, w), np.int32)
            for nid, j in enumerate(prod_keep, 1):
                cand_rec[comps[j]["mask"]] = nid
            rec["cand_matches_stored"] = bool(np.array_equal(cand_rec, load(f"{key}__cand").astype(np.int32)))
        for mean_min in MEANS:
            for area_min in AREAS:
                keep = [j for j, c in enumerate(comps) if c["area"] >= area_min and c["mean"] >= mean_min]
                for rthr in RTHRS:
                    lab = assemble(h, w, refined, keep, rthr)
                    s, tp, fpn, fnn = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
                    rec["counts"][f"{thr}|{mean_min}|{area_min}|{rthr}"] = [float(s), int(tp), int(fpn), int(fnn)]
    rec["seconds"] = round(time.time() - t_img)
    results.append(rec)
    print(f"{k}/{len(names)} {name} passes={len(passes)} prod={rec['counts']['0.4|0.7|400|0.5']} "
          f"cand_ok={rec['cand_matches_stored']} {rec['seconds']}s", flush=True)
    json.dump({"chunk_index": CHUNK_INDEX, "status": "running", "results": results},
              open(f"/kaggle/working/tune_chunk{CHUNK_INDEX}.json", "w"))

json.dump({"chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "status": "done",
           "elapsed_s": round(time.time() - t0),
           "grid": {"thr": THRS, "mean": MEANS, "area": AREAS, "refine_thr": RTHRS}, "results": results},
          open(f"/kaggle/working/tune_chunk{CHUNK_INDEX}.json", "w"))
print("done", flush=True)
