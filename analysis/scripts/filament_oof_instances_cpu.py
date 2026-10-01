"""
Solar Filament Segmentation 2026 -- per-instance features and per-pass IoU of every predicted
instance, on the stored out-of-fold (657) and hold-out (50) predictions. CPU-ONLY, no inference,
no network.

Purpose: training data for a learned instance scorer. Under PQ, keeping an instance adds
IoU * [IoU > 0.5] to the numerator and exactly 0.5 to the denominator in every annotator pass
(TP: +1 TP, -1 FN -> +0.5; FP: +0.5), so an instance is worth keeping iff its expected matched IoU
exceeds PQ / 2. Earlier filters were fitted on about 420 hold-out candidates; this gives about 8,000.

Per image (released pipeline with the near-limb filter, drop@0.88):
  instances: features computable at inference time (stage-A probability statistics, shape, intensity,
             position on the disk, neighbourhood) and, per pass, the instance's maximum IoU with the
             ground truth; `counts` recomputed from these equal filament_metric.pq_counts (asserted);
  gts:       per pass, each ground-truth instance's area, disk position and maximum IoU with the
             predictions (for the false-negative breakdown).
The scorer is fitted and evaluated locally from oof_instances.json (grouped CV by image).
"""

import glob
import json
import math
import os
import time

import numpy as np
import pycocotools.mask as mu

import filament_metric as fm
import filament_pipeline as fp
import instance_features as feats

INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()


def gt_rles(anns, h, w):
    out = []
    for a in anns:
        if a.get("segmentation"):
            r = mu.frPyObjects(a["segmentation"], h, w)
            out.append(mu.merge(r) if isinstance(r, list) else r)
    return out


ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)
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
    name, z = stem + ".jpeg", files[stores[(tag, stem)]]
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    lab = fp.drop_near_limb(z[stem + "__lab"].astype(np.int32), g)
    cand, prob = z[stem + "__cand"].astype(np.int32), z[stem + "__prob"].astype(np.float32) / 255.0
    inst, disk = feats.image_instances(g, lab, cand, prob)
    # per-pass IoU against the ground truth
    pred = [{"size": [h, w], "counts": c.encode("ascii")} for c in fp.label_map_to_rles(lab)]
    assert len(pred) == len(inst)
    for r in inst:
        r["iou"] = []
    counts, gts = np.zeros(4), []
    for pid, anns in sorted(per_file[name].items(), key=lambda kv: str(kv[0])):
        gt = gt_rles(anns, h, w)
        g_area = [float(a) for a in mu.area(gt)] if gt else []
        g_dist = [float(math.hypot(b[0] + b[2] / 2 - disk[0], b[1] + b[3] / 2 - disk[1]) / disk[2]) for b in mu.toBbox(gt)] if gt else []
        if gt and pred:
            iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)), np.float64)
            hit = iou > 0.5
            counts += [iou[hit].sum(), hit.sum(), (hit.sum(1) == 0).sum(), (hit.sum(0) == 0).sum()]
            for r, row in zip(inst, iou):
                r["iou"].append(float(row.max()))
            g_max = iou.max(0).tolist()
        else:
            counts += [0, 0, len(pred) if not gt else 0, len(gt) if not pred else 0]
            for r in inst:
                r["iou"].append(0.0)
            g_max = [0.0] * len(gt)
        gts.append({"pass": str(pid), "gt": [[a, round(d, 4), round(m, 4)] for a, d, m in zip(g_area, g_dist, g_max)]})
    ref = fm.pq_counts([p["counts"].decode("ascii") for p in pred], per_file[name], h, w)
    assert abs(ref[0] - counts[0]) < 1e-6 and tuple(ref[1:]) == tuple(int(c) for c in counts[1:]), (name, ref, counts)
    results.append({"name": name, "set": tag, "n_pass": len(per_file[name]), "counts": [float(counts[0]), *map(int, counts[1:])],
                    "disk": [round(v, 2) for v in disk], "instances": inst, "passes": gts})
    if k % 50 == 0 or k == len(stores):
        print(f"{k}/{len(stores)} {time.time() - t0:.0f}s instances so far {sum(len(r['instances']) for r in results)}", flush=True)
        json.dump({"status": "running" if k < len(stores) else "done", "results": results}, open(f"{OUT}/oof_instances.json", "w"))
print("done", round(time.time() - t0), "s", flush=True)
