"""
Solar Filament Segmentation 2026 -- refiner-confidence and mask-stability features for the instance
scorer, CPU-ONLY (refiner inference only), CHUNK_INDEX of CHUNK_TOTAL.

The released scorer (AUC 0.78) sees only stage-A probabilities, shape and context. This kernel adds
what the refiner knows about each instance of the stored out-of-fold (657) and hold-out (50)
predictions: the refiner probability inside the mask, the width of its uncertain band, and how much the
mask changes under a mirrored crop or a smaller / larger seed box (filament_scorer.refiner_features).
Output: refiner_feats_chunk{i}.json, {image name: {instance id: features}}, joined locally with
oof_instances.json by (name, id). `reproduced` counts the instances whose re-run refiner mask equals
the stored one (integrity check).
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
from scipy import ndimage

import filament_pipeline as fp
import filament_scorer as fs

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
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

res, n_inst, n_same = {}, 0, 0
for k, (tag, stem) in enumerate(mine, 1):
    name, z = stem + ".jpeg", files[stores[(tag, stem)]]
    g = fp.read_gray(os.path.join(train_dir, name))
    cand, stored = z[stem + "__cand"].astype(np.int32), z[stem + "__lab"].astype(np.int32)
    lab = fp.drop_near_limb(stored, g)
    ids = [i for i, sl in enumerate(ndimage.find_objects(lab), 1) if sl is not None]
    res[name] = {str(i): f for i, f in fs.refiner_features(g, cand, ids, refiner, device).items()}
    # integrity: the refiner re-run here must redraw the stored masks
    objs = ndimage.find_objects(cand)
    for i in ids:
        sl = objs[i - 1]
        p, (x0, y0, x1, y1) = fp.refiner_prob(g, [sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start], refiner, device)
        full = cv2.resize((p > fp.REFINE_THR).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        same = (full == (stored[y0:y1, x0:x1] == i)) | ((stored[y0:y1, x0:x1] != i) & (stored[y0:y1, x0:x1] > 0))   # later ids may overwrite
        n_same += bool(same.all())
        n_inst += 1
    if k % 10 == 0 or k == len(mine):
        print(f"{k}/{len(mine)} {time.time() - t0:.0f}s instances {n_inst} reproduced {n_same}", flush=True)
        json.dump({"status": "running" if k < len(mine) else "done", "chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL,
                   "n_instances": n_inst, "reproduced": n_same, "features": res}, open(f"{OUT}/refiner_feats_chunk{CHUNK_INDEX}.json", "w"))
print("done", round(time.time() - t0), "s", flush=True)
