"""Inspect the spine annotations before any GPU training uses them (CPU, a few minutes).

Reports: annotation keys; the type/shape of the spine field; the share of annotations with a spine;
and, for rasterised spines (both x,y orders tried), the share of spine pixels that fall inside the
filament's own mask -- the order with ~all pixels inside is the correct one.
"""
import json
import os

import cv2
import numpy as np
import pycocotools.mask as mu

INPUT = "/kaggle/input"
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
coco = json.load(open(ann_json))
anns = coco["annotations"]
images = {im["id"]: im for im in coco["images"]}
print("top-level keys:", list(coco))
print("annotation keys:", sorted({k for a in anns[:500] for k in a}))
cand = [k for k in anns[0] if "spine" in k.lower() or "skel" in k.lower() or "line" in k.lower()]
print("spine-like keys:", cand)
key = cand[0] if cand else None
out = {"keys": sorted({k for a in anns[:500] for k in a}), "spine_key": key, "n_annotations": len(anns)}
if key:
    have = [a for a in anns if a.get(key)]
    out["share_with_spine"] = len(have) / len(anns)
    s = have[0][key]
    out["example_type"] = type(s).__name__
    out["example"] = s if len(json.dumps(s)) < 600 else json.dumps(s)[:600]
    print("share with spine:", out["share_with_spine"], "| example:", out["example"])
    print("categories:", coco.get("categories"))

    def flat(s):
        """-> list of polylines, each an (N, 2) float array in the order stored."""
        if isinstance(s, dict):
            s = s.get("points") or s.get("coordinates") or list(s.values())[0]
        if s and isinstance(s[0], (int, float)):
            s = [s]
        lines = []
        for part in s:
            arr = np.asarray(part, float)
            lines.append(arr.reshape(-1, 2) if arr.ndim == 1 else arr[:, :2])
        return lines

    inside = {"xy": [], "yx": []}
    for a in have[:400]:
        im = images[a["image_id"]]
        h, w = im["height"], im["width"]
        r = mu.frPyObjects(a["segmentation"], h, w)
        m = mu.decode(mu.merge(r) if isinstance(r, list) else r).astype(bool)
        for order in ("xy", "yx"):
            canvas = np.zeros((h, w), np.uint8)
            for line in flat(a[key]):
                pts = line if order == "xy" else line[:, ::-1]
                cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 1, 1)
            n = canvas.sum()
            if n:
                inside[order].append(float((canvas.astype(bool) & m).sum() / n))
    for order, v in inside.items():
        print(f"order {order}: median share of spine pixels inside the mask {np.median(v):.3f} (n={len(v)})")
    out["inside_share_median"] = {o: float(np.median(v)) if v else None for o, v in inside.items()}
json.dump(out, open("/kaggle/working/spine_inspect.json", "w"), indent=1, default=str)
print("done")
