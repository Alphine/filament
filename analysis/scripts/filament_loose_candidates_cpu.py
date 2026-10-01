"""
Solar Filament Segmentation 2026 -- looser candidate thresholds for the learned instance scorer,
CPU-ONLY (refiner inference only), CHUNK_INDEX of CHUNK_TOTAL.

The candidate thresholds (p > 0.4, area >= 400, mean p >= 0.7) were tuned when nothing downstream could
remove false positives: loosening them added 61 FP for 10 TP. With the instance scorer as the last
stage, extra candidates can be kept only where they pay, so recall may now be cheaper. For every
configuration in CONFIGS this kernel re-extracts candidates from the STORED stage-A probability maps
(out of fold 657, hold-out 50), redraws them with the released refiner, applies the near-limb filter
and writes per-instance scorer features and per-pass IoU, in the format of filament_oof_instances_cpu.py.
The scorer is refitted and evaluated locally per configuration (grouped CV by image).

"prod" repeats the released thresholds as an integrity check: its label maps are compared with the
stored ones (`same_as_stored`).
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

import filament_metric as fm
import filament_pipeline as fp
import filament_scorer as fs

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
CONFIGS = {"prod": (0.4, 400, 0.7), "loose": (0.3, 200, 0.6), "xloose": (0.2, 150, 0.45)}   # thr, min area, min mean p
INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()


def gt_rles(anns, h, w):
    out = []
    for a in anns:
        if a.get("segmentation"):
            r = mu.frPyObjects(a["segmentation"], h, w)
            out.append(mu.merge(r) if isinstance(r, list) else r)
    return out


def per_pass(inst, lab, passes, h, w):
    """Adds the per-pass maximum IoU to every instance; returns the official counts of the image."""
    pred = [{"size": [h, w], "counts": c.encode("ascii")} for c in fp.label_map_to_rles(lab)]
    assert len(pred) == len(inst)
    for r in inst:
        r["iou"] = []
    counts = np.zeros(4)
    for _, anns in sorted(passes.items(), key=lambda kv: str(kv[0])):
        gt = gt_rles(anns, h, w)
        if gt and pred:
            iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)), np.float64)
            hit = iou > 0.5
            counts += [iou[hit].sum(), hit.sum(), (hit.sum(1) == 0).sum(), (hit.sum(0) == 0).sum()]
            for r, row in zip(inst, iou):
                r["iou"].append(float(row.max()))
        else:
            counts += [0, 0, len(pred) if not gt else 0, len(gt) if not pred else 0]
            for r in inst:
                r["iou"].append(0.0)
    ref = fm.pq_counts([p["counts"].decode("ascii") for p in pred], passes, h, w)
    assert abs(ref[0] - counts[0]) < 1e-6 and tuple(ref[1:]) == tuple(int(c) for c in counts[1:]), (ref, counts)
    return [float(counts[0]), *map(int, counts[1:])]


ann_json = next(os.path.join(r, f) for r, _, fs_ in os.walk(INPUT) for f in fs_ if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)
device = fp.default_device()
refiner = fp.load_refiner(next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True)), device)
stores, files = {}, {}
for p in glob.glob(f"{INPUT}/**/*.npz", recursive=True):
    tag = "holdout" if os.path.basename(p).startswith("preds_chunk") else "oof"
    files[p] = np.load(p)
    for k in files[p].files:
        if k.endswith("__lab"):
            stores[(tag, k[:-5])] = p
assert sum(t == "oof" for t, _ in stores) == 657 and sum(t == "holdout" for t, _ in stores) == 50, len(stores)
mine = [key for k, key in enumerate(sorted(stores)) if k % CHUNK_TOTAL == CHUNK_INDEX]
print(f"chunk {CHUNK_INDEX}/{CHUNK_TOTAL}: {len(mine)} images on {device}", flush=True)

results = []
for k, (tag, stem) in enumerate(mine, 1):
    name, z = stem + ".jpeg", files[stores[(tag, stem)]]
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    prob = z[stem + "__prob"].astype(np.float32) / 255.0
    stored = fp.drop_near_limb(z[stem + "__lab"].astype(np.int32), g)
    rec = {"name": name, "set": tag, "n_pass": len(per_file[name]), "configs": {}}
    for cfg, (thr, min_area, min_mean) in CONFIGS.items():
        cand = fp.extract_candidates(prob, thr, min_area, min_mean)
        lab = fp.drop_near_limb(fp.refine_candidates(g, cand, refiner, device), g)
        inst, _ = fs.image_instances(g, lab, cand, prob)
        rec["configs"][cfg] = {"counts": per_pass(inst, lab, per_file[name], h, w), "n_cand": int(cand.max()), "instances": inst}
        if cfg == "prod":
            rec["same_as_stored"] = bool(((lab > 0) == (stored > 0)).all() and len(inst) == len(np.unique(stored)) - 1)
    results.append(rec)
    if k % 10 == 0 or k == len(mine):
        n = {c: sum(len(r["configs"][c]["instances"]) for r in results) for c in CONFIGS}
        print(f"{k}/{len(mine)} {time.time() - t0:.0f}s instances {n} same_as_stored {sum(r['same_as_stored'] for r in results)}/{k}", flush=True)
        json.dump({"status": "running" if k < len(mine) else "done", "chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL,
                   "configs": CONFIGS, "results": results}, open(f"{OUT}/loose_chunk{CHUNK_INDEX}.json", "w"))
try:
    import lightgbm
    print("lightgbm", lightgbm.__version__, flush=True)
except ImportError:
    print("lightgbm not installed in this image", flush=True)
print("done", round(time.time() - t0), "s", flush=True)
