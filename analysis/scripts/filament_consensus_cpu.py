"""
Solar Filament Segmentation 2026 -- ANNOTATION CONSENSUS-TARGET TEST, CPU-only,
annotation file only (no images, no model, no training).

QUESTION. Stage-A is trained on the pixel UNION of all annotator passes
(filament_unet_ensemble.py: `m |= ...`), while the official metric is computed
per pass and then averaged. Is the union target inherently worse than other
consensus targets? Quick derivation: adding a prediction that is marked in a
fraction f of the passes raises the PQ numerator by f*IoU and the denominator
by a constant 0.5, so it pays off only if f > 0.5*PQ/IoU (~0.32).

METHOD. For images with 3 passes (A,B,C): hold one pass out as ground truth (C),
build the "prediction" ONLY from the other two passes (A,B) with several target
constructors, score it against C with the official rule (IoU>0.5 is a hit;
PQ = sumIoU/(TP+0.5FP+0.5FN)), and pool the three choices of C over all images.
This simulates the TARGET (an upper bound if a model learned that target
perfectly); it is not a model result.

Constructors:
  single_A / single_B   : one pass alone (baseline; ~0.34 in the earlier test)
  pix_union_cc          : connected components of the pixel union A|B
                          (exactly the stage-A training target)
  pix_inter_cc          : connected components of the pixel intersection A&B
  inst_union_keep       : instances of A and B are paired (IoU>0.3, greedy
                          one-to-one); a pair -> union mask; unpaired
                          instances are KEPT
  inst_inter_keep       : pair -> intersection mask; unpaired instances kept
  inst_union_drop       : pair -> union; unpaired instances DROPPED
                          (only objects both passes agree on)
  inst_inter_drop       : pair -> intersection; unpaired instances dropped
Components/intersections with area < MIN_AREA pixels are dropped (rasterisation noise).
"""

import json
import os
import time

import cv2
import numpy as np
import pycocotools.mask as mask_util

OUT = "/kaggle/working"
MATCH_IOU = 0.3
MIN_AREA = 100


def discover():
    ann = None
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".json") and "nnotation" in f:
                ann = os.path.join(root, f)
    if not ann:
        raise SystemExit("annotation file not found")
    return ann


def load(ann_path):
    coco = json.load(open(ann_path))
    images = {im["id"]: im for im in coco["images"]}
    by_file = {}
    for a in coco["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            continue
        fn = os.path.basename(im["file_name"])
        by_file.setdefault(fn, {}).setdefault(a["image_id"], []).append(
            (a, im["height"], im["width"]))
    return by_file


def rles_of(items):
    out = []
    for a, h, w in items:
        seg = a.get("segmentation")
        if not seg:
            continue
        r = mask_util.frPyObjects(seg, h, w)
        r = mask_util.merge(r) if isinstance(r, list) else r
        out.append(r)
    return out


def compare(pred, gt):
    n_p, n_g = len(pred), len(gt)
    if n_p == 0 or n_g == 0:
        return (0.0, 0, n_p, n_g)
    iou = np.asarray(mask_util.iou(pred, gt, [0] * n_g), dtype=np.float64)
    hit = iou > 0.5
    return (float(iou[hit].sum()), int(hit.sum()),
            int((hit.sum(1) == 0).sum()), int((hit.sum(0) == 0).sum()))


def pooled(stats):
    S = sum(s[0] for s in stats)
    tp = sum(s[1] for s in stats)
    fp = sum(s[2] for s in stats)
    fn = sum(s[3] for s in stats)
    d = tp + 0.5 * fp + 0.5 * fn
    return (S / d if d else 0.0), tp, fp, fn


def keep_big(rles):
    return [r for r in rles if mask_util.area(r) >= MIN_AREA]


def cc_rles(mask_u8):
    n, lab = cv2.connectedComponents(mask_u8, connectivity=8)
    out = []
    for i in range(1, n):
        r = mask_util.encode(np.asfortranarray((lab == i).astype(np.uint8)))
        if mask_util.area(r) >= MIN_AREA:
            out.append(r)
    return out


def match(ra, rb):
    """Greedy one-to-one pairing by largest IoU (> MATCH_IOU)."""
    if not ra or not rb:
        return [], list(range(len(ra))), list(range(len(rb)))
    iou = np.asarray(mask_util.iou(ra, rb, [0] * len(rb)), dtype=np.float64)
    cand = sorted(((iou[i, j], i, j) for i in range(len(ra)) for j in range(len(rb))
                   if iou[i, j] > MATCH_IOU), reverse=True)
    ua, ub, pairs = set(), set(), []
    for v, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); pairs.append((i, j))
    return (pairs, [i for i in range(len(ra)) if i not in ua],
            [j for j in range(len(rb)) if j not in ub])


def pass_mask(rles, h, w):
    if not rles:
        return np.zeros((h, w), np.uint8)
    return mask_util.decode(mask_util.merge(rles)).astype(np.uint8)


def build(ra, rb, ma, mb):
    """All constructors from two passes (ra, rb = lists of instance RLEs; ma, mb =
    merged binary masks)."""
    out = {}
    out["single_A"] = ra
    out["single_B"] = rb
    out["pix_union_cc"] = cc_rles(np.maximum(ma, mb))
    out["pix_inter_cc"] = cc_rles(np.minimum(ma, mb))
    pairs, ua, ub = match(ra, rb)
    uni = [mask_util.merge([ra[i], rb[j]]) for i, j in pairs]
    ist = [mask_util.merge([ra[i], rb[j]], intersect=True) for i, j in pairs]
    ist = keep_big(ist)
    lone = [ra[i] for i in ua] + [rb[j] for j in ub]
    out["inst_union_keep"] = uni + lone
    out["inst_inter_keep"] = ist + lone
    out["inst_union_drop"] = uni
    out["inst_inter_drop"] = ist
    out["_n"] = (len(ra), len(rb), len(pairs))
    return out


def main():
    t0 = time.time()
    by_file = load(discover())
    variants = ["single_A", "single_B", "pix_union_cc", "pix_inter_cc",
                "inst_union_keep", "inst_inter_keep", "inst_union_drop", "inst_inter_drop"]
    per_file = {}
    n_used = 0
    match_stats = []
    for fn, passes in by_file.items():
        ids = sorted(passes, key=str)
        if len(ids) != 3:
            continue
        n_used += 1
        h, w = passes[ids[0]][0][1], passes[ids[0]][0][2]
        rl = {p: rles_of(passes[p]) for p in ids}
        mk = {p: pass_mask(rl[p], h, w) for p in ids}
        rec = {v: [] for v in variants}
        n_inst = {v: [] for v in variants}
        for c in ids:
            a, b = [p for p in ids if p != c]
            built = build(rl[a], rl[b], mk[a], mk[b])
            match_stats.append(built["_n"])
            for v in variants:
                rec[v].append(compare(built[v], rl[c]))
                n_inst[v].append(len(built[v]))
        per_file[fn] = (rec, n_inst, [len(rl[p]) for p in ids])
        if n_used % 25 == 0:
            print(f"  {n_used} 3-pass images  {time.time()-t0:.0f}s", flush=True)

    keys = list(per_file)
    rng = np.random.default_rng(0)
    res = {"n_files_3pass": len(keys), "min_area": MIN_AREA, "match_iou": MATCH_IOU,
           "variants": {}}
    for v in variants:
        flat = [t for k in keys for t in per_file[k][0][v]]
        pq, tp, fp, fn = pooled(flat)
        bs = []
        for _ in range(1500):
            pick = rng.integers(0, len(keys), len(keys))
            bs.append(pooled([t for k in pick for t in per_file[keys[k]][0][v]])[0])
        n_pred = float(np.mean([x for k in keys for x in per_file[k][1][v]]))
        res["variants"][v] = {"pq": round(pq, 4), "se": round(float(np.std(bs)), 4),
                              "tp": tp, "fp": fp, "fn": fn,
                              "mean_pred_instances_per_case": round(n_pred, 2)}
    # paired difference against pix_union_cc (our training target), bootstrap over images
    base = "pix_union_cc"
    res["paired_vs_pix_union_cc"] = {}
    for v in variants:
        if v == base:
            continue
        diffs = []
        for _ in range(1500):
            pick = rng.integers(0, len(keys), len(keys))
            a = pooled([t for k in pick for t in per_file[keys[k]][0][v]])[0]
            b = pooled([t for k in pick for t in per_file[keys[k]][0][base]])[0]
            diffs.append(a - b)
        res["paired_vs_pix_union_cc"][v] = {
            "diff_mean": round(float(np.mean(diffs)), 4),
            "ci95": [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]}
    res["mean_instances_per_pass_gt"] = round(float(np.mean(
        [x for k in keys for x in per_file[k][2]])), 2)
    ms = np.array(match_stats)
    res["mean_match"] = {"n_A": round(float(ms[:, 0].mean()), 2),
                         "n_B": round(float(ms[:, 1].mean()), 2),
                         "n_matched": round(float(ms[:, 2].mean()), 2)}
    res["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(os.path.join(OUT, "gpu_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
