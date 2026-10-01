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

import cv2
import numpy as np
import pycocotools.mask as mu
from scipy import ndimage

import filament_metric as fm
import filament_pipeline as fp

INPUT, OUT = "/kaggle/input", "/kaggle/working"
PAD = 32
K_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
t0 = time.time()


def crop(sl, shape):
    y0, y1 = max(0, sl[0].start - PAD), min(shape[0], sl[0].stop + PAD)
    x0, x1 = max(0, sl[1].start - PAD), min(shape[1], sl[1].stop + PAD)
    return slice(y0, y1), slice(x0, x1)


def shape_feats(comp):
    """Geometry of a binary crop (largest contour) -- no ground truth involved."""
    u8 = comp.astype(np.uint8)
    n_cc = cv2.connectedComponents(u8, connectivity=8)[0] - 1
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    ca, per = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    rw, rh = cv2.minAreaRect(cnt)[1]
    ha = cv2.contourArea(cv2.convexHull(cnt))
    ys, xs = np.nonzero(comp)
    return {"n_cc": int(n_cc),
            "circularity": float(4 * math.pi * ca / per ** 2) if per > 0 and ca > 0 else 0.0,
            "aspect": float(max(rw, rh) / max(min(rw, rh), 1e-3)), "length": float(max(rw, rh)),
            "width": float(min(rw, rh)), "solidity": float(ca / ha) if ha > 0 else 1.0,
            "extent": float(comp.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))}


def instance_feats(i, lab, cand, prob, g, sl, sl_c, disk, g_med):
    cx, cy, R = disk
    cs = crop(sl, lab.shape)
    comp, pc, gc = lab[cs] == i, prob[cs], g[cs].astype(np.float32)
    ys, xs = np.nonzero(comp)
    pv = pc[comp]
    f = {"area": int(comp.sum()), "mean_p": float(pv.mean()), "max_p": float(pv.max()),
         "p10_p": float(np.percentile(pv, 10)), "p50_p": float(np.median(pv)), "frac_p40": float((pv > 0.4).mean()),
         "cy": float(ys.mean() + cs[0].start), "cx": float(xs.mean() + cs[1].start)}
    f["dist_r"] = float(math.hypot(f["cx"] - cx, f["cy"] - cy) / R)
    f.update(shape_feats(comp))
    if sl_c is not None:                                       # the stage-A candidate the refiner redrew
        cc = crop(sl_c, lab.shape)
        cm = cand[cc] == i
        cv_ = prob[cc][cm]
        inter = int(((cand == i)[cs] & comp).sum())
        f.update({"cand_area": int(cm.sum()), "cand_mean_p": float(cv_.mean()), "cand_p10_p": float(np.percentile(cv_, 10)),
                  "cand_iou": float(inter / (cm.sum() + comp.sum() - inter)),
                  **{f"cand_{k}": v for k, v in shape_feats(cm).items() if k in ("aspect", "solidity", "length", "width")}})
    else:
        f.update({"cand_area": 0, "cand_mean_p": 0.0, "cand_p10_p": 0.0, "cand_iou": 0.0,
                  "cand_aspect": 0.0, "cand_solidity": 0.0, "cand_length": 0.0, "cand_width": 0.0})
    f["refined_ratio"] = float(f["area"] / max(f["cand_area"], 1))
    u8 = comp.astype(np.uint8)
    ring = (cv2.dilate(u8, K_OUT) > 0) & (cv2.dilate(u8, K_IN) == 0)
    inside = gc[comp]
    rg = gc[ring] if ring.sum() > 20 else inside
    f.update({"contrast": float((rg.mean() - inside.mean()) / (rg.std() + 1e-3)), "ring_mean": float(rg.mean() / g_med),
              "ring_std": float(rg.std() / g_med), "inside_mean": float(inside.mean() / g_med),
              "inside_std": float(inside.std() / g_med), "ring_p": float(pc[ring].mean()) if ring.any() else 0.0})
    return f


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
    disk = fp.estimate_disk(g)
    sub = g[::4, ::4]
    g_med = float(np.median(sub[sub > 20]))
    objs, objs_c = ndimage.find_objects(lab), ndimage.find_objects(cand)
    ids = [i for i, sl in enumerate(objs, 1) if sl is not None]
    inst = [{"id": i, "feats": instance_feats(i, lab, cand, prob, g, objs[i - 1],
                                              objs_c[i - 1] if i <= len(objs_c) else None, disk, g_med)} for i in ids]
    # neighbourhood features
    cen = np.array([[r["feats"]["cx"], r["feats"]["cy"]] for r in inst]).reshape(-1, 2)
    ar = np.array([r["feats"]["area"] for r in inst], float)
    for j, r in enumerate(inst):
        d = np.hypot(*(cen - cen[j]).T)
        d[j] = np.inf
        nn = int(np.argmin(d)) if len(inst) > 1 else j
        r["feats"].update({"n_inst": len(inst), "nn_dist": float(d[nn]) if len(inst) > 1 else 4096.0,
                           "nn_area": float(ar[nn]) if len(inst) > 1 else 0.0, "n_within_150": int((d < 150).sum()),
                           "area_rank": float((ar < ar[j]).mean()), "img_pred_area": float(ar.sum()),
                           "img_mean_p": float(prob[::4, ::4].mean())})
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
