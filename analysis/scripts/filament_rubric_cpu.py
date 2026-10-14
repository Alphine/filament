"""
Solar Filament Segmentation 2026 -- raw overlap data for the organizers' rubric metrics, CPU-ONLY, no inference, no network.

The final evaluation (Overview > Evaluation) lists, besides PQ: the mean Dice score, the distributions of Dice and IoU
scores, and the distribution of one-to-many and many-to-one relations between ground-truth and predicted
filaments. The organizers' self-evaluation notebook defines the pairwise quantities: per annotator-image, the
(n_gt x n_pred) IoU and Dice matrices; distributions use the pairs with a score > 0; one-to-many / many-to-one count,
per ground-truth filament and per prediction, the partners with IoU > 0.

For every annotator pass of every out-of-fold (657) and hold-out (50) image this stores what those quantities (and
the image-level Dice of the union masks) are computed from, for the released predictions with the near-limb filter
(the scorer's keep decisions are applied afterwards, locally, because predictions do not overlap each other):
  gt_area[j], pred_area[i], inter[j][i]   pairwise intersections (exact integers), pred ids in label-map order
  gt_union                                area of the union of the pass's ground-truth masks
  pred_gtu[i]                             |prediction i  intersect  union of the ground truth|
Integrity: PQ counts recomputed from the stored intersections equal filament_metric.pq_counts on every image.
"""

import glob
import json
import os
import time

import numpy as np
import pycocotools.mask as mu

import filament_metric as fm
import filament_pipeline as fp

INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()


def gt_rles(anns, h, w):
    out = []
    for a in anns:
        if a.get("segmentation"):
            r = mu.frPyObjects(a["segmentation"], h, w)
            out.append(mu.merge(r) if isinstance(r, list) else r)
    return out


def inter_from_iou(iou, a, b):
    """exact integer intersections from IoU and areas: I = iou * (a + b) / (1 + iou)"""
    s = np.asarray(a, np.float64)[:, None] + np.asarray(b, np.float64)[None, :]
    return np.rint(iou * s / (1.0 + iou)).astype(np.int64)


def pass_record(pred, gt, h, w):
    """pred, gt: lists of RLE dicts -> the stored quantities of one annotator-image"""
    p_area = [int(a) for a in mu.area(pred)] if pred else []
    g_area = [int(a) for a in mu.area(gt)] if gt else []
    if gt and pred:
        iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)), np.float64).T          # (n_gt, n_pred)
        inter = inter_from_iou(iou, g_area, p_area)
        gu = mu.merge(gt, intersect=False)
        gt_union = int(mu.area(gu))
        pred_gtu = inter_from_iou(np.asarray(mu.iou(pred, [gu], [0]), np.float64).T, [gt_union], p_area)[0].tolist()
    else:
        inter = np.zeros((len(gt), len(pred)), np.int64)
        gt_union = int(mu.area(mu.merge(gt, intersect=False))) if gt else 0
        pred_gtu = [0] * len(pred)
    return {"gt_area": g_area, "pred_area": p_area, "inter": inter.tolist(), "gt_union": gt_union, "pred_gtu": pred_gtu}


def pq_from_record(rec):
    """(sum IoU of TP, TP, FP, FN) of one annotator-image from the stored intersections (order of fm.pq_counts)"""
    g, p = np.asarray(rec["gt_area"], np.float64), np.asarray(rec["pred_area"], np.float64)
    if len(g) == 0 or len(p) == 0:
        return 0.0, 0, len(p), len(g)
    inter = np.asarray(rec["inter"], np.float64)
    iou = inter / (g[:, None] + p[None, :] - inter)
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())


def main():
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
        pred = [{"size": [h, w], "counts": c.encode("ascii")} for c in fp.label_map_to_rles(lab)]
        passes = []
        for pid, anns in sorted(per_file[name].items(), key=lambda kv: str(kv[0])):
            rec = pass_record(pred, gt_rles(anns, h, w), h, w)
            rec["pass"] = str(pid)
            passes.append(rec)
        ref = fm.pq_counts([p["counts"].decode("ascii") for p in pred], per_file[name], h, w)
        tot = np.sum([pq_from_record(r) for r in passes], axis=0)
        assert abs(ref[0] - tot[0]) < 1e-6 and (int(ref[1]), int(ref[2]), int(ref[3])) == (int(tot[1]), int(tot[2]), int(tot[3])), \
            (name, ref, tot.tolist())
        results.append({"name": name, "set": tag, "h": h, "w": w, "n_pred": len(pred), "passes": passes})
        if k % 50 == 0 or k == len(stores):
            print(f"{k}/{len(stores)} {time.time() - t0:.0f}s", flush=True)
            json.dump({"status": "running" if k < len(stores) else "done", "results": results}, open(f"{OUT}/rubric_raw.json", "w"))
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
