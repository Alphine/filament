"""
Solar Filament Segmentation 2026 -- prediction-morphology figures for the report, CPU-ONLY,
no inference: the released 5-model ensemble's stored hold-out label maps (preds_chunk*.npz from
filament_merge_cpu.py) with the near-limb filter of the final pipeline applied on top.

Reference annotation per image: the annotator pass with the most instances (as in the report's
example figure). Instance categories against that pass:
  match       prediction with IoU > 0.5 (a true positive under the official rule)
  near-miss   prediction whose best IoU is in (0.25, 0.5]
  phantom     prediction that touches no annotated filament (best IoU < 0.02)
  missed      annotated filament whose best IoU with any prediction is < 0.1
  fragmented  annotated filament overlapped (IoU > 0.1) by >= 2 predictions
  complex     matched filaments with the most branched annotation (lowest solidity) -- barbs
  limb-drop   prediction removed by the near-limb filter (centroid beyond 0.88 R)
Outputs: fig_morph_overview.png, fig_morph_gallery.png, morph_examples.json (per-category counts
and the chosen examples).
"""

import glob
import json
import os

import cv2
import matplotlib
import numpy as np
import pycocotools.mask as mu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import filament_metric as fm  # noqa: E402
import filament_pipeline as fp  # noqa: E402

INPUT, OUT = "/kaggle/input", "/kaggle/working"
N_PER_ROW = 6
COL = {"gt": (60, 220, 60), "match": (0, 200, 255), "near-miss": (255, 170, 0), "phantom": (255, 50, 50),
       "missed": (255, 230, 0), "limb-drop": (230, 60, 230), "pred": (0, 200, 255)}


def display(gray):
    lo, hi = np.percentile(gray, [1, 99.5])
    x = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) ** 0.8
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def contour(img, mask, color, th):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, th)


def solidity(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 1.0
    c = max(cs, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(mask.sum() / hull) if hull > 0 else 1.0


def bbox_of(*masks):
    m = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(m)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def analyse(name, gray, lab_rel, passes):
    """Per-instance records against the largest annotator pass, for the final (limb-filtered) prediction."""
    h, w = gray.shape
    final = fp.drop_near_limb(lab_rel, gray)
    dropped = [i for i in range(1, int(lab_rel.max()) + 1) if (lab_rel == i).any() and not (final == i).any()]
    anns = max(passes.values(), key=len)
    gt = [mu.decode(r).astype(bool) for r in fm._gt_rles(anns, h, w)]
    pid = [i for i in range(1, int(final.max()) + 1) if (final == i).any()]
    pm = [final == i for i in pid]
    iou = np.zeros((len(pm), len(gt)))
    for a, p in enumerate(pm):
        for b, g in enumerate(gt):
            inter = np.logical_and(p, g).sum()
            if inter:
                iou[a, b] = inter / np.logical_or(p, g).sum()
    recs = []
    for a, p in enumerate(pm):
        best = iou[a].max() if gt else 0.0
        cat = "match" if best > 0.5 else "near-miss" if best > 0.25 else "phantom" if best < 0.02 else None
        if cat:
            b = int(iou[a].argmax()) if gt else -1
            recs.append({"cat": cat, "iou": float(best), "pred": [a], "gt": [b] if b >= 0 and best > 0 else []})
    for b, g in enumerate(gt):
        col = iou[:, b] if len(pm) else np.zeros(0)
        best = col.max() if col.size else 0.0
        if best < 0.1:
            recs.append({"cat": "missed", "iou": float(best), "pred": [], "gt": [b]})
        over = [a for a in range(len(pm)) if col[a] > 0.1]
        if len(over) >= 2:
            recs.append({"cat": "fragmented", "iou": float(best), "pred": over, "gt": [b]})
        if best > 0.5:
            recs.append({"cat": "complex", "iou": float(best), "pred": [int(col.argmax())], "gt": [b], "solidity": solidity(g)})
    for i in dropped:
        m = lab_rel == i
        ious = [np.logical_and(m, g).sum() / np.logical_or(m, g).sum() for g in gt]
        best = float(max(ious, default=0.0))
        recs.append({"cat": "limb-drop", "iou": best, "drop_mask": m, "pred": [],
                     "gt": [int(np.argmax(ious))] if ious and best > 0 else []})
    return final, pm, gt, recs, len(dropped), iou


def crop_panel(gray, rec, pm, gt, size=256):
    masks = [pm[a] for a in rec["pred"]] + [gt[b] for b in rec["gt"]] + ([rec["drop_mask"]] if "drop_mask" in rec else [])
    y0, y1, x0, x1 = bbox_of(*masks)
    side = int(max(y1 - y0, x1 - x0) * 1.6) + 40
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h, w = gray.shape
    ya, xa = max(0, cy - side // 2), max(0, cx - side // 2)
    yb, xb = min(h, ya + side), min(w, xa + side)
    img = display(gray[ya:yb, xa:xb]).copy()
    th = max(1, int(round((yb - ya) / 180)))
    for b in rec["gt"]:
        contour(img, gt[b][ya:yb, xa:xb], COL["gt"], th)
    pc = COL.get(rec["cat"], COL["pred"])
    for a in rec["pred"]:
        contour(img, pm[a][ya:yb, xa:xb], pc, th)
    if "drop_mask" in rec:
        contour(img, rec["drop_mask"][ya:yb, xa:xb], COL["limb-drop"], th)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
    train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
    per_file, dims = fm.load_annotations(ann_json, train_dir)
    store = {}
    for p in glob.glob(f"{INPUT}/**/preds_chunk*.npz", recursive=True):
        for k in np.load(p).files:
            if k.endswith("__lab"):
                store[k[:-5]] = p
    names = fm.holdout_names(list(per_file))
    assert all(n.rsplit(".", 1)[0] in store for n in names), "stored hold-out label maps missing"
    pools = {c: [] for c in ("complex", "near-miss", "phantom", "missed", "fragmented", "limb-drop")}
    counts, overview = {}, []
    for n in names:
        stem = n.rsplit(".", 1)[0]
        gray = fp.read_gray(os.path.join(train_dir, n))
        lab = np.load(store[stem])[stem + "__lab"].astype(np.int32)
        final, pm, gt, recs, n_drop, iou = analyse(n, gray, lab, per_file[n])
        for r in recs:
            counts[r["cat"]] = counts.get(r["cat"], 0) + 1
            if r["cat"] in pools:
                r2 = dict(r, name=n)
                r2["panel"] = crop_panel(gray, r, pm, gt)
                r2.pop("drop_mask", None)
                pools[r["cat"]].append(r2)
        if len(overview) < 3 and n_drop and len(gt) >= 5:
            img = display(gray).copy()
            for b, g in enumerate(gt):          # annotation: green if found (IoU > 0.5), yellow if missed
                hit = iou.shape[0] and iou[:, b].max() > 0.5
                contour(img, g, COL["gt"] if hit else COL["missed"], 6)
            for a, p in enumerate(pm):          # prediction: cyan if matched, red otherwise
                hit = iou.shape[1] and iou[a].max() > 0.5
                contour(img, p, COL["match"] if hit else COL["phantom"], 6)
            for i in range(1, int(lab.max()) + 1):
                m = lab == i
                if m.any() and not (final == i).any():
                    contour(img, m, COL["limb-drop"], 8)
            overview.append((n, cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)))
    # deterministic selection per category
    pick = {"complex": lambda r: r["solidity"], "near-miss": lambda r: -r["iou"], "phantom": lambda r: r["name"],
            "missed": lambda r: r["name"], "fragmented": lambda r: -len(r["pred"])}
    chosen = {c: sorted(v, key=pick[c])[:N_PER_ROW] for c, v in pools.items() if c != "limb-drop"}
    ld = sorted(pools["limb-drop"], key=lambda r: r["iou"])          # balanced: removed FPs and lost TPs
    chosen["limb-drop"] = ld[:N_PER_ROW // 2] + ld[::-1][:N_PER_ROW - N_PER_ROW // 2]
    counts["limb-drop_false_positive"] = sum(r["iou"] <= 0.5 for r in ld)
    counts["limb-drop_true_positive"] = sum(r["iou"] > 0.5 for r in ld)
    titles = {"complex": "Matched, most curved/branched", "near-miss": "Near-miss (0.25 < IoU <= 0.5)",
              "phantom": "Phantom false positive", "missed": "Missed filament", "fragmented": "Fragmented (>= 2 predictions)",
              "limb-drop": "Removed near the limb:\nleft FP, right lost TP"}
    fig, axes = plt.subplots(len(chosen), N_PER_ROW, figsize=(2.0 * N_PER_ROW, 2.15 * len(chosen)))
    for r_i, (c, recs) in enumerate(chosen.items()):
        for c_i in range(N_PER_ROW):
            ax = axes[r_i, c_i]
            ax.set_xticks([]); ax.set_yticks([])
            if c_i < len(recs):
                ax.imshow(recs[c_i]["panel"])
                ax.set_xlabel(f"IoU {recs[c_i]['iou']:.2f}", fontsize=7)
            else:
                ax.axis("off")
        axes[r_i, 0].set_ylabel(titles[c], fontsize=7)
    fig.suptitle("Hold-out predictions of the final pipeline (green: annotation; cyan: matched; orange: near-miss; "
                 "red: phantom; magenta: removed near the limb)", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_morph_gallery.png", dpi=150)
    fig, axes = plt.subplots(1, len(overview), figsize=(4.2 * len(overview), 4.6))
    for ax, (n, img) in zip(np.atleast_1d(axes), overview):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(n[:16], fontsize=8)
    fig.suptitle("Annotation: green found, yellow missed. Prediction: cyan matched (IoU > 0.5), red unmatched, "
                 "magenta removed by the near-limb filter", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_morph_overview.png", dpi=150)
    json.dump({"counts": counts, "overview": [n for n, _ in overview],
               "chosen": {c: [{k: v for k, v in r.items() if k != "panel"} for r in recs] for c, recs in chosen.items()}},
              open(f"{OUT}/morph_examples.json", "w"), indent=1, default=float)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
