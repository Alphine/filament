"""
Solar Filament Segmentation 2026 -- per-instance mask variants and their IoU with every annotator pass,
CPU-ONLY (refiner inference only), CHUNK_INDEX of CHUNK_TOTAL.

About half of the remaining error is near-misses: a prediction overlaps a ground-truth filament but with
IoU <= 0.5 (38 hold-out false positives lie at IoU 0.4-0.5 alone). A per-image refiner threshold gains only
+0.005 even as an oracle, so this study asks the per-INSTANCE question: for every instance of the stored
out-of-fold (657) and hold-out (50) predictions it redraws the mask in ten ways and stores, per variant and
per annotator pass, the maximum IoU with the ground truth. The analysis then measures the oracle (best
variant per instance) and a learned selector (analyze_mask_variants.py).

Variants (all from the released refiner on the stage-A candidate's crop):
  prod   threshold 0.5, seed box padded 40 %   (the released mask; integrity-checked against the stored one)
  t30 t40 t60 t70   thresholds 0.3 / 0.4 / 0.6 / 0.7 of the same probability map
  flip   refiner on the mirrored crop          p20 / p60   seed box padded 20 % / 60 %
  avg4   mean probability of prod, flip, p20, p60 at 0.5
  cand   the stage-A candidate itself (no refiner)
Instances are those that survive the near-limb filter (the ids of oof_instances.json).
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
import pycocotools.mask as mu
from scipy import ndimage

import filament_metric as fm
import filament_pipeline as fp

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
INPUT, OUT = "/kaggle/input", "/kaggle/working"
VARIANTS = ("prod", "t30", "t40", "t60", "t70", "flip", "p20", "p60", "avg4", "cand")
t0 = time.time()


def gt_rles(anns, h, w):
    out = []
    for a in anns:
        if a.get("segmentation"):
            r = mu.frPyObjects(a["segmentation"], h, w)
            out.append(mu.merge(r) if isinstance(r, list) else r)
    return out


def variant_masks(g, box, cand_crop_fn, refiner, device):
    """name -> (bool mask on the crop, crop bounds) for the ten variants of one candidate box."""
    p, (x0, y0, x1, y1) = fp.refiner_prob(g, box, refiner, device)
    pf, _ = fp.refiner_prob(g, box, refiner, device, flip=True)
    ps, _ = fp.refiner_prob(g, box, refiner, device, seed_pad=0.2)
    pl, _ = fp.refiner_prob(g, box, refiner, device, seed_pad=0.6)
    m = {"prod": p > 0.5, "t30": p > 0.3, "t40": p > 0.4, "t60": p > 0.6, "t70": p > 0.7,
         "flip": pf > 0.5, "p20": ps > 0.5, "p60": pl > 0.5, "avg4": (p + pf + ps + pl) / 4 > 0.5}
    out = {k: cv2.resize(v.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool) for k, v in m.items()}
    out["cand"] = cand_crop_fn(x0, y0, x1, y1)
    return out, (x0, y0, x1, y1)


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

images, n_inst, n_same = [], 0, 0
for k, (tag, stem) in enumerate(mine, 1):
    name, z = stem + ".jpeg", files[stores[(tag, stem)]]
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    cand, stored = z[stem + "__cand"].astype(np.int32), z[stem + "__lab"].astype(np.int32)
    lab = fp.drop_near_limb(stored, g)
    ids = [i for i, sl in enumerate(ndimage.find_objects(lab), 1) if sl is not None]
    objs = ndimage.find_objects(cand)
    recs, rles, owner = [], [], []
    for i in ids:
        sl = objs[i - 1]
        box = [sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start]
        masks, (x0, y0, x1, y1) = variant_masks(g, box, lambda a, b, c, d, i=i: cand[b:d, a:c] == i, refiner, device)
        prod = masks["prod"]
        st = stored[y0:y1, x0:x1]
        n_same += bool(((prod == (st == i)) | ((st != i) & (st > 0))).all())     # later ids may overwrite pixels
        n_inst += 1
        rec = {"id": i, "area": {}, "iou_prod": {}, "iou": {}}
        for v in VARIANTS:
            m = masks[v]
            rec["area"][v] = int(m.sum())
            rec["iou_prod"][v] = float((m & prod).sum() / max(int((m | prod).sum()), 1))
            if m.any():
                full = np.zeros((h, w), np.uint8)
                full[y0:y1, x0:x1] = m
                rles.append(mu.encode(np.asfortranarray(full)))
                owner.append((len(recs), v))
        recs.append(rec)
    n_gt, per_pass = [], []
    for _, anns in sorted(per_file[name].items(), key=lambda kv: str(kv[0])):
        gt = gt_rles(anns, h, w)
        n_gt.append(len(gt))
        best = np.asarray(mu.iou(rles, gt, [0] * len(gt)), np.float64).max(1) if gt and rles else np.zeros(len(rles))
        per_pass.append(best)
    for j, (ri, v) in enumerate(owner):
        recs[ri]["iou"][v] = [float(best[j]) for best in per_pass]
    for r in recs:
        for v in VARIANTS:
            r["iou"].setdefault(v, [0.0] * len(n_gt))      # empty mask: dropped, nothing submitted
    ref = fm.pq_counts(fp.label_map_to_rles(lab), per_file[name], h, w)
    images.append({"name": name, "set": tag, "n_gt": n_gt, "official": [float(ref[0]), *map(int, ref[1:])], "instances": recs})
    if k % 10 == 0 or k == len(mine):
        print(f"{k}/{len(mine)} {time.time() - t0:.0f}s instances {n_inst} prod==stored {n_same}", flush=True)
        json.dump({"status": "running" if k < len(mine) else "done", "chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL,
                   "variants": VARIANTS, "n_instances": n_inst, "prod_same_as_stored": n_same, "images": images},
                  open(f"{OUT}/variants_chunk{CHUNK_INDEX}.json", "w"))
print("done", round(time.time() - t0), "s", flush=True)
