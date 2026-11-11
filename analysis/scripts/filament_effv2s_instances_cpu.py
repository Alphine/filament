"""
Instances of OUR EfficientNetV2-S stage-A maps (training/filament_effv2s_stagea.py), for the adoption gate. CPU-ONLY.

Input: the training kernels attached as kernel sources; every effv2s_maps_fold<F>_{oof,holdout}.npz found under
/kaggle/input is used (the gate needs all five folds; a subset gives a partial, descriptive file with status "partial").
  oof      each pool image's map comes from the one model that never saw it (657 images with five folds)
  holdout  the 50 hold-out images, uint8 maps averaged over the folds found (the ensemble)
Instances: exactly as in filament_pubw_oof_cpu.py, so the two studies are comparable -- 8-connected regions of
p > 0.35 under the LOOSE rule (peak > 0.5, area >= 100, mean p >= 0.5), flag `strict` for the public card's keep rule
(peak > 0.99, area >= 200, mean p >= 0.85). There is no refiner: cand = lab. Per instance the 36 scorer features and its
IoU per annotator pass; per image the official counts of the loose set (asserted against filament_metric.pq_counts).
The near-limb filter and the scorer refit are applied offline (analyze_effv2s.py).
"""

import glob
import json
import os
import re
import time

import numpy as np
import pycocotools.mask as mu
from scipy import ndimage as ndi

import filament_metric as fm
import filament_pipeline as fp
import filament_scorer as feats

INPUT, OUT = "/kaggle/input", "/kaggle/working"
T0 = time.time()
BUDGET_S = 40 * 60          # the whole job takes about 10 minutes; anything far beyond that is a fault -> stop
H = W = 2048
T_LO = 0.35
LOOSE = dict(t_hi=0.5, min_area=100, min_meanp=0.5)
STRICT = dict(t_hi=0.99, min_area=200, min_meanp=0.85)


def u8(t):
    return int(round(float(t) * 255))


def label_instances(p_u8):
    """Loose-rule instances of the map, relabelled 1..n; -> (label map, strict flags)."""
    lab, n = ndi.label(p_u8 > u8(T_LO), structure=np.ones((3, 3), bool))
    lv, pv = lab.ravel(), p_u8.ravel()
    nz = np.flatnonzero(lv)
    area = np.bincount(lv[nz], minlength=n + 1)
    psum = np.bincount(lv[nz], weights=pv[nz], minlength=n + 1)
    pmax = np.zeros(n + 1, np.uint8)
    np.maximum.at(pmax, lv[nz], pv[nz])
    meanp = psum / np.maximum(area, 1) / 255.0

    def ok(rule):
        return (pmax > u8(rule["t_hi"])) & (area >= rule["min_area"]) & (meanp >= rule["min_meanp"])

    loose, strict = ok(LOOSE), ok(STRICT)
    loose[0] = False
    assert not (strict[1:] & ~loose[1:]).any()
    ids = np.flatnonzero(loose)
    remap = np.zeros(n + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[lab], [bool(strict[i]) for i in ids]


def gt_rles(anns, h, w):
    out = []
    for a in anns:
        if a.get("segmentation"):
            r = mu.frPyObjects(a["segmentation"], h, w)
            out.append(mu.merge(r) if isinstance(r, list) else r)
    return out


def record(name, tag, fold, avg):
    g = fp.read_gray(os.path.join(train_dir, name))
    assert g.shape == avg.shape == (H, W) and avg.dtype == np.uint8, (name, g.shape, avg.shape, avg.dtype)
    lab, strict = label_instances(avg)
    inst, disk = feats.image_instances(g, lab, lab, avg.astype(np.float32) / 255.0)
    pred = [{"size": [H, W], "counts": c.encode("ascii")} for c in fp.label_map_to_rles(lab)]
    assert len(pred) == len(inst) == len(strict)
    for r, s in zip(inst, strict):
        r["iou"], r["strict"] = [], s
    counts = np.zeros(4)
    for pid, anns in sorted(per_file[name].items(), key=lambda kv: str(kv[0])):
        gt = gt_rles(anns, H, W)
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
    ref = fm.pq_counts([p["counts"].decode("ascii") for p in pred], per_file[name], H, W)
    assert abs(ref[0] - counts[0]) < 1e-6 and tuple(ref[1:]) == tuple(int(c) for c in counts[1:]), (name, ref, counts)
    return {"name": name, "set": tag, "fold": fold, "n_pass": len(per_file[name]),
            "counts": [float(counts[0]), *map(int, counts[1:])], "disk": [round(v, 2) for v in disk], "instances": inst}


ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)
names = sorted(per_file)
holdout = fm.holdout_names(names)
assert len(names) == 707 and len(holdout) == 50

found = {}
for tag in ("oof", "holdout"):
    for p in sorted(glob.glob(os.path.join(INPUT, "**", f"effv2s_maps_fold*_{tag}.npz"), recursive=True)):
        f = int(re.search(r"fold(\d+)_", os.path.basename(p)).group(1))
        assert (tag, f) not in found, ("duplicate fold", p, found[(tag, f)])
        found[(tag, f)] = p
folds = sorted(f for t, f in found if t == "oof")
assert folds and folds == sorted(f for t, f in found if t == "holdout"), found
print("folds found:", folds, flush=True)
# FAIL FAST before any heavy work: every map file opens, has the expected images and format
for (tag, f), p in sorted(found.items()):
    z = np.load(p)
    assert len(z.files) == (50 if tag == "holdout" else len(z.files)) and len(z.files) >= 50, (p, len(z.files))
    m = z[z.files[0]]
    assert m.shape == (H, W) and m.dtype == np.uint8 and m.max() > 128, (p, m.shape, m.dtype, int(m.max()))
if os.environ.get("REQUIRE_ALL") == "1":
    assert folds == [1, 2, 3, 4, 5], folds
print("fail-fast OK:", len(found), "map files", flush=True)

results, seen = [], set()
for f in folds:
    z = np.load(found[("oof", f)])
    for stem in sorted(z.files):
        name = stem + ".jpeg"
        assert name in per_file and name not in holdout and name not in seen, name
        seen.add(name)
        results.append(record(name, "oof", f, z[stem]))
    print(f"fold {f}: {len(z.files)} images, total {len(results)}, {time.time() - T0:.0f}s", flush=True)
    json.dump({"status": "running", "folds": folds[:folds.index(f) + 1], "results": results}, open(f"{OUT}/effv2s_instances.json", "w"))
    assert time.time() - T0 < BUDGET_S, "time budget exceeded: stopping (partial file kept)"
if folds == [1, 2, 3, 4, 5]:
    assert len(seen) == 657 and seen == set(names) - set(holdout), len(seen)

zs = [np.load(found[("holdout", f)]) for f in folds]
for z in zs:
    assert sorted(z.files) == sorted(n[:-5] for n in holdout)
for stem in sorted(zs[0].files):
    avg = np.clip(np.round(np.mean([z[stem].astype(np.float32) for z in zs], axis=0)), 0, 255).astype(np.uint8)
    results.append(record(stem + ".jpeg", "holdout", 0, avg))
status = "done" if folds == [1, 2, 3, 4, 5] else "partial"
json.dump({"status": status, "folds": folds, "results": results}, open(f"{OUT}/effv2s_instances.json", "w"))
print(status, len(results), "images,", sum(len(r["instances"]) for r in results), "instances,", round(time.time() - T0), "s", flush=True)
