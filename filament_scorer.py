"""Learned instance scorer: the last stage of the pipeline.

Under PQ, keeping a predicted instance adds its matched IoU to the numerator and 0.5 to the denominator
in every annotator pass, so an instance is worth keeping only if its expected matched IoU exceeds PQ / 2.
A gradient-boosted regressor (LightGBM) predicts that expected matched IoU from features available at
inference time -- stage-A probability statistics, shape, local contrast, position on the disk and
neighbourhood -- and instances below the threshold are dropped. It was fitted on 4,694 out-of-fold
instances (analysis/scripts/analyze_instance_scorer.py; out of fold +0.0144 PQ, leaderboard 0.36 -> 0.37).

The feature functions here are the single implementation used for fitting and for inference.
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
from scipy import ndimage

import filament_pipeline as fp

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

PAD = 32
K_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))


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


def image_instances(g, lab, cand, prob):
    """-> (instances, disk): one {"id", "feats"} per non-empty instance of `lab` (ids ascending, the order of
    filament_pipeline.label_map_to_rles). `cand` is the stage-A candidate map with the same ids, `prob` the
    stage-A probability map quantised to 1/255 steps."""
    disk = fp.estimate_disk(g)
    sub = g[::4, ::4]
    g_med = float(np.median(sub[sub > 20]))
    objs, objs_c = ndimage.find_objects(lab), ndimage.find_objects(cand)
    ids = [i for i, sl in enumerate(objs, 1) if sl is not None]
    inst = [{"id": i, "feats": instance_feats(i, lab, cand, prob, g, objs[i - 1],
                                              objs_c[i - 1] if i <= len(objs_c) else None, disk, g_med)} for i in ids]
    cen = np.array([[r["feats"]["cx"], r["feats"]["cy"]] for r in inst]).reshape(-1, 2)
    ar = np.array([r["feats"]["area"] for r in inst], float)
    for j, r in enumerate(inst):                                  # neighbourhood features
        d = np.hypot(*(cen - cen[j]).T)
        d[j] = np.inf
        nn = int(np.argmin(d)) if len(inst) > 1 else j
        r["feats"].update({"n_inst": len(inst), "nn_dist": float(d[nn]) if len(inst) > 1 else 4096.0,
                           "nn_area": float(ar[nn]) if len(inst) > 1 else 0.0, "n_within_150": int((d < 150).sum()),
                           "area_rank": float((ar < ar[j]).mean()), "img_pred_area": float(ar.sum()),
                           "img_mean_p": float(prob[::4, ::4].mean())})
    return inst, disk


def quantise(prob: np.ndarray) -> np.ndarray:
    """Probability map in 1/255 steps: the precision of the stored maps the scorer was fitted on."""
    return (prob * 255).round().astype(np.uint8).astype(np.float32) / 255.0


class InstanceScorer:
    """Predicted expected matched IoU per instance; `filter` drops the instances below `tau`."""

    def __init__(self, booster, features: list[str], tau: float) -> None:
        self.booster, self.features, self.tau = booster, features, tau

    @classmethod
    def load(cls, model_dir: str = MODEL_DIR, tau: float | None = None, name: str = "instance_scorer") -> "InstanceScorer":
        """The stored model. `tau` overrides the stored threshold (0.1972, the out-of-fold PQ / 2): a lower
        value keeps more instances (0.15: out-of-fold PQ -0.0025, per-image Dice +0.0072; report section 4).
        `name="effv2s_scorer"` loads the scorer of the EfficientNetV2-S stage-A (filament_effv2s.py; tau 0.1885)."""
        import lightgbm as lgb
        with open(os.path.join(model_dir, f"{name}.json")) as f:
            meta = json.load(f)
        return cls(lgb.Booster(model_file=os.path.join(model_dir, f"{name}.txt")), meta["features"],
                   meta["tau"] if tau is None else float(tau))

    def scores(self, instances: list[dict]) -> np.ndarray:
        if not instances:
            return np.zeros(0)
        return self.booster.predict(np.array([[r["feats"][k] for k in self.features] for r in instances], float))

    def filter(self, gray: np.ndarray, labels: np.ndarray, candidates: np.ndarray, prob: np.ndarray) -> np.ndarray:
        """Label map without the instances scored below tau. `candidates` is the stage-A candidate map
        (same ids as `labels`), `prob` the stage-A probability map."""
        inst, _ = image_instances(gray, labels, candidates, quantise(prob))
        drop = [r["id"] for r, v in zip(inst, self.scores(inst)) if v < self.tau]
        return np.where(np.isin(labels, drop), 0, labels) if drop else labels


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def refiner_features(gray: np.ndarray, candidates: np.ndarray, ids: list[int], refiner, device) -> dict[int, dict]:
    """Refiner-confidence and mask-stability features per instance id (study features, not part of the
    released scorer): statistics of the refiner probability inside the mask, and how much the mask
    changes when the refiner sees the mirrored crop or a smaller / larger seed box."""
    out = {}
    objs = ndimage.find_objects(candidates)
    for i in ids:
        sl = objs[i - 1]
        bbox = [sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start]
        p, _ = fp.refiner_prob(gray, bbox, refiner, device)
        m = p > fp.REFINE_THR
        f = {"ref_mean_p": float(p[m].mean()) if m.any() else 0.0, "ref_p10_p": float(np.percentile(p[m], 10)) if m.any() else 0.0,
             "ref_max_p": float(p.max()), "ref_band": float(((p > 0.3) & (p < 0.7)).sum() / max(int(m.sum()), 1)),
             "ref_area_70_30": float((p > 0.7).sum() / max(int((p > 0.3).sum()), 1))}
        pf, _ = fp.refiner_prob(gray, bbox, refiner, device, flip=True)
        ps, _ = fp.refiner_prob(gray, bbox, refiner, device, seed_pad=0.2)
        pl, _ = fp.refiner_prob(gray, bbox, refiner, device, seed_pad=0.6)
        ious = [_iou(m, q > fp.REFINE_THR) for q in (pf, ps, pl)]
        mean_p = (p + pf + ps + pl) / 4
        f.update({"ref_iou_flip": ious[0], "ref_iou_seed20": ious[1], "ref_iou_seed60": ious[2], "ref_iou_min": min(ious),
                  "ref_absdiff_flip": float(np.abs(p - pf)[m | (pf > fp.REFINE_THR)].mean()) if (m | (pf > fp.REFINE_THR)).any() else 0.0,
                  "ref_iou_mean4": _iou(m, mean_p > fp.REFINE_THR)})
        out[i] = f
    return out
