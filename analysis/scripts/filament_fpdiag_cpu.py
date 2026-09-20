"""
Solar Filament Segmentation 2026 -- per-image & per-candidate FP DIAGNOSTICS,
CPU-ONLY, NO TRAINING, 4 parallel chunks.

GOALS (steps 1-3 of the multi-agent audit plan):
 1. Save per-image/per-pair results (not just totals) so that paired
    comparisons + bootstrap CIs can be computed, and so that the effect of
    DROPPING specific candidates on the official PQ can be simulated EXACTLY
    offline.
 2. Re-dissect TP/FP/FN of the 0.4325 pipeline (phase 1): pure FP (touches no
    GT) vs localization FP (touches GT but IoU<0.5); pure FN vs localization FN.
 3. Per-candidate stage-A features (before the refiner) to falsify the
    FP-filter idea: AUC of TP vs FP; if below ~0.8 the idea is dropped.

EXACT phase-1 pipeline (stage-A 5-fold + TTA, instances(thr=0.4,
min_area=400, min_mean_prob=0.7), phase-1 refiner) -- checkpoints are loaded
via kernel_sources, NOT retrained. Same sacred holdout (seed 1234, 50 images),
split into CHUNK_TOTAL pieces; each kernel writes
/kaggle/working/holdout_chunk{CHUNK_INDEX}.json (small, numbers only).
Merging + analysis is done LOCALLY from those JSON files (without downloading
the competition data to the local machine).

Integrity reference: phase-1 proxy PQ = 0.4236656564767375 (noise-floor
tolerance 0.0016 across CPU/GPU), checked after all chunks are merged.
"""

import json
import math
import os
import time

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch"], check=True)
    import segmentation_models_pytorch as smp

OUT = "/kaggle/working"

CHUNK_INDEX = 0     # 0-based -- CHANGED per kernel copy (0..3)
CHUNK_TOTAL = 4

PATCH = 512
STRIDE = 384
INFER_BATCH = 4
N_VAL = 50
REF_PARAMS = dict(thr=0.4, min_area=400, min_mean_prob=0.7)

CONTEXT = 1.5
MIN_CROP = 96
PAD_RATIO = 0.40
REFINE_CROP = 512
REFINE_THR = 0.50

DEV = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() or 4))


def discover():
    ann = train_dir = test_dir = None
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".json") and "nnotation" in f:
                ann = os.path.join(root, f)
        b = os.path.basename(root).lower()
        if b == "train_images":
            train_dir = root
        elif b == "test_images":
            test_dir = root
    if not (ann and train_dir):
        for root, _, files in os.walk("/kaggle/input"):
            print(root, len(files))
        raise SystemExit("data not found")
    return ann, train_dir, test_dir


ANN, TRAIN_IMG, TEST_IMG = discover()


def load_annotations():
    with open(ANN) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    per_file, dims = {}, {}
    for img_id, anns in by_img.items():
        im = images.get(img_id)
        if im is None:
            continue
        fn = os.path.basename(im["file_name"])
        if not os.path.exists(os.path.join(TRAIN_IMG, fn)):
            continue
        per_file.setdefault(fn, []).extend(anns)
        dims[fn] = (im["height"], im["width"])
    names = sorted(per_file)
    print(f"annotations ready: {len(names)} unique images", flush=True)
    return names, per_file, dims


# ------------------------------------------------------------------ stage-A
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True))


class ResUNet(nn.Module):
    def __init__(self, arch="resnet34", pretrained=False):
        super().__init__()
        import torchvision.models as tvm
        ctor, (c0, c1, c2, c3, c4) = {
            "resnet34": (tvm.resnet34, (64, 64, 128, 256, 512)),
        }[arch]
        r = ctor(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2 = r.layer1, r.layer2
        self.l3, self.l4 = r.layer3, r.layer4
        up = lambda i, o: nn.ConvTranspose2d(i, o, 2, stride=2)
        self.u4, self.d4 = up(c4, 256), block(256 + c3, 256)
        self.u3, self.d3 = up(256, 128), block(128 + c2, 128)
        self.u2, self.d2 = up(128, 64), block(64 + c1, 64)
        self.u1, self.d1 = up(64, 64), block(64 + c0, 64)
        self.u0 = up(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        y = self.d4(torch.cat([self.u4(s4), s3], 1))
        y = self.d3(torch.cat([self.u3(y), s2], 1))
        y = self.d2(torch.cat([self.u2(y), s1], 1))
        y = self.d1(torch.cat([self.u1(y), s0], 1))
        return self.head(self.u0(y))


UNet = ResUNet

TTA_TFMS = [
    ("id",   lambda a: a),
    ("lr",   lambda a: a[:, ::-1]),
    ("ud",   lambda a: a[::-1, :]),
    ("r180", lambda a: a[::-1, ::-1]),
]


@torch.no_grad()
def predict_prob_ens_std(models, gray):
    """Identical to predict_prob_ens(tta=True) -- mean of 20 predictions
    (5 models x 4 flips) -- but also returns the per-pixel std across the 20
    predictions (a model-disagreement feature)."""
    acc = acc2 = None
    n = 0
    for m in models:
        for _, f in TTA_TFMS:
            p = predict_prob(m, np.ascontiguousarray(f(gray)))
            p = np.ascontiguousarray(f(p))
            if acc is None:
                acc = p.copy()
                acc2 = p * p
            else:
                acc += p
                acc2 += p * p
            n += 1
    mean = acc / n
    var = np.maximum(acc2 / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


def find_r34_checkpoints():
    hits = []
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("unet_fold") and f.endswith(".pt"):
                hits.append(os.path.join(root, f))
    return sorted(hits)


def find_phase1_refiner():
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "refiner_best.pt":
                return os.path.join(root, f)
    return None


@torch.no_grad()
def predict_prob(model, gray):
    H, W = gray.shape
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H - PATCH, 0) + 1, STRIDE))
    xs = list(range(0, max(W - PATCH, 0) + 1, STRIDE))
    if ys[-1] != H - PATCH:
        ys.append(H - PATCH)
    if xs[-1] != W - PATCH:
        xs.append(W - PATCH)
    batch, coords = [], []
    for y in ys:
        for x in xs:
            p = gray[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0
            p = (p - p.mean()) / (p.std() + 1e-6)
            batch.append(p)
            coords.append((y, x))
            if len(batch) == INFER_BATCH:
                _flush(model, batch, coords, acc, cnt)
                batch, coords = [], []
    if batch:
        _flush(model, batch, coords, acc, cnt)
    return acc / np.maximum(cnt, 1e-6)


def _flush(model, batch, coords, acc, cnt):
    t = torch.from_numpy(np.stack(batch))[:, None].float()
    pr = torch.sigmoid(model(t)).float().numpy()[:, 0]
    for (y, x), m in zip(coords, pr):
        acc[y:y + PATCH, x:x + PATCH] += m
        cnt[y:y + PATCH, x:x + PATCH] += 1


def instances(prob, thr, min_area, min_mean_prob=0.0, max_area=40000, close_k=3):
    m = (prob > thr).astype(np.uint8)
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros(lab.shape, np.int32)
    nid = 1
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area or a > max_area:
            continue
        comp = lab == i
        if min_mean_prob > 0 and prob[comp].mean() < min_mean_prob:
            continue
        out[comp] = nid
        nid += 1
    return out


def load_models(items, arch):
    ms = []
    for it in items:
        m = UNet(arch, pretrained=False)
        m.load_state_dict(torch.load(it, map_location=DEV))
        m.eval()
        ms.append(m)
    return ms


# --------------------------------------------------------------- phase-1 refiner
def _bbox(m):
    ys = np.flatnonzero(m.any(1))
    if not ys.size:
        return None
    xs = np.flatnonzero(m.any(0))
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def square_bounds_from_bbox(bbox, h, w, context=CONTEXT, minimum=MIN_CROP):
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    side = int(math.ceil(max(bw, bh) * context))
    side = max(side, minimum)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    x0, y0 = max(0, x0), max(0, y0)
    return x0, y0, x1, y1


def seed_from_box(h, w, bbox, pad_ratio=PAD_RATIO):
    x, y, bw, bh = bbox
    px, py = bw * pad_ratio, bh * pad_ratio
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
    seed = np.zeros((h, w), np.uint8)
    seed[y0:y1, x0:x1] = 1
    return seed


@torch.no_grad()
def refine_label_map_conf(gray, coarse_lab, refiner, thr=REFINE_THR):
    """Identical to phase-1 refine_label_map (same label map), plus the
    refiner confidence per instance: mean/max refiner prob inside the resulting mask."""
    h, w = gray.shape
    out = np.zeros_like(coarse_lab)
    conf = {}
    clahe_op = cv2.createCLAHE(2.0, (8, 8))
    n_inst = int(coarse_lab.max())
    if n_inst == 0:
        return out, conf
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    for i in range(1, n_inst + 1):
        m = coarse_lab == i
        b = _bbox(m)
        if b is None:
            continue
        by0, by1, bx0, bx1 = b
        bbox = [bx0, by0, bx1 - bx0, by1 - by0]
        seed_full = seed_from_box(h, w, bbox, pad_ratio=PAD_RATIO)
        x0, y0, x1, y1 = square_bounds_from_bbox(bbox, h, w)
        cs = REFINE_CROP
        img_c = cv2.resize(gray[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR)
        seed_c = cv2.resize(seed_full[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
        clahe_c = clahe_op.apply(img_c)
        x = np.stack([img_c, clahe_c, seed_c * 255], 0).astype(np.float32) / 255.0
        x = (x - mean) / std
        xt = torch.from_numpy(x)[None]
        prob = torch.sigmoid(refiner(xt)).float().numpy()[0, 0]
        crop_mask = (prob > thr)
        if crop_mask.any():
            conf[i] = (float(prob[crop_mask].mean()), float(prob.max()))
        else:
            conf[i] = (0.0, float(prob.max()))
        full_mask = cv2.resize(crop_mask.astype(np.uint8), (x1 - x0, y1 - y0),
                               interpolation=cv2.INTER_NEAREST)
        out[y0:y1, x0:x1][full_mask.astype(bool)] = i
    return out, conf


# ------------------------------------------------------------ candidate features
def candidate_features(gray, prob, std_map, comp, coarse_area_ref=None):
    """Features of one stage-A candidate (binary mask `comp`, bool HxW) -- all
    computed from the image/prob itself, WITHOUT touching GT."""
    ys, xs = np.nonzero(comp)
    area = int(comp.sum())
    pv = prob[comp]
    f = {
        "area": area,
        "mean_p": float(pv.mean()),
        "max_p": float(pv.max()),
        "p10_p": float(np.percentile(pv, 10)),
        "std_views": float(std_map[comp].mean()),
        "cy": float(ys.mean()), "cx": float(xs.mean()),
    }
    f["dist_center"] = float(math.hypot(f["cy"] - 1024.0, f["cx"] - 1024.0) / 1000.0)
    u8 = comp.astype(np.uint8)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        ca = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        f["circularity"] = float(4 * math.pi * ca / (per ** 2)) if per > 0 and ca > 0 else 0.0
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        f["aspect"] = float(max(rw, rh) / max(min(rw, rh), 1e-3))
        f["length"] = float(max(rw, rh))
        f["width"] = float(min(rw, rh))
        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        f["solidity"] = float(ca / ha) if ha > 0 else 1.0
    else:
        f.update(circularity=0.0, aspect=1.0, length=0.0, width=0.0, solidity=1.0)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    f["extent"] = float(area / max((bx1 - bx0) * (by1 - by0), 1))
    # local contrast: inside intensity vs the surrounding ring (normalized by ring std)
    pad = 32
    x0, x1 = max(0, bx0 - pad), min(gray.shape[1], bx1 + pad)
    y0, y1 = max(0, by0 - pad), min(gray.shape[0], by1 + pad)
    sub = comp[y0:y1, x0:x1].astype(np.uint8)
    g = gray[y0:y1, x0:x1].astype(np.float32)
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(sub, k_out) > 0) & (cv2.dilate(sub, k_in) == 0)
    inside = g[sub > 0]
    if ring.sum() > 20:
        rg = g[ring]
        f["contrast"] = float((rg.mean() - inside.mean()) / (rg.std() + 1e-3))
        f["ring_mean"] = float(rg.mean())
    else:
        f["contrast"] = 0.0
        f["ring_mean"] = 0.0
    f["inside_mean"] = float(inside.mean())
    f["inside_std"] = float(inside.std())
    return f


# ----------------------------------------------------- detailed official evaluation
def official_detail(name, per_file, dims, pred_rles):
    """Official metric per annotator-pass, but keeping the details: hit pairs
    (pred_idx, gt_idx, iou), n_gt, n_pred, max-IoU of each pred and each GT.
    Sufficient to recompute the exact official PQ after dropping candidates."""
    by_pass = {}
    for a_ in per_file[name]:
        by_pass.setdefault(a_["image_id"], []).append(a_)
    h, w = dims[name]
    passes = []
    for pid, anns in sorted(by_pass.items(), key=lambda kv: str(kv[0])):
        gt_rles = []
        for a_ in anns:
            seg = a_.get("segmentation")
            if not seg:
                continue
            r = mask_util.frPyObjects(seg, h, w)
            r = mask_util.merge(r) if isinstance(r, list) else r
            c = r["counts"]
            gt_rles.append(c.decode("utf-8") if isinstance(c, bytes) else c)
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        rec = {"pass": str(pid), "n_gt": n_gt, "n_pred": n_pred,
               "pairs": [], "pred_max_iou": [], "gt_max_iou": []}
        if n_gt == 0 or n_pred == 0:
            rec["pred_max_iou"] = [0.0] * n_pred
            rec["gt_max_iou"] = [0.0] * n_gt
            passes.append(rec)
            continue
        g = mask_util.decode([{"size": [h, w], "counts": r} for r in gt_rles])
        g = g.transpose(2, 0, 1).astype(np.float32)
        p = mask_util.decode([{"size": [h, w], "counts": r} for r in pred_rles])
        p = p.transpose(2, 0, 1).astype(np.float32)
        gt_t, pr_t = torch.from_numpy(g), torch.from_numpy(p)
        inter = torch.matmul(gt_t.reshape(n_gt, -1), pr_t.reshape(n_pred, -1).t())
        ga = gt_t.reshape(n_gt, -1).sum(1).view(-1, 1)
        pa = pr_t.reshape(n_pred, -1).sum(1).view(1, -1)
        union = ga + pa - inter
        iou = torch.where(union == 0, torch.tensor(0.), inter / union)
        hit = iou > 0.5
        gi, pi = torch.nonzero(hit, as_tuple=True)
        rec["pairs"] = [[int(b), int(a), float(iou[a, b])] for a, b in zip(gi.tolist(), pi.tolist())]
        rec["pred_max_iou"] = [float(x) for x in iou.max(0).values.tolist()]
        rec["gt_max_iou"] = [float(x) for x in iou.max(1).values.tolist()]
        passes.append(rec)
    return passes


def val_gt(fn, per_file, dims):
    h, w = dims[fn]
    by_pass = {}
    for a in per_file[fn]:
        by_pass.setdefault(a["image_id"], []).append(a)
    anns = max(by_pass.values(), key=len)
    gt = []
    for a in anns:
        seg = a.get("segmentation")
        if not seg:
            continue
        r = mask_util.frPyObjects(seg, h, w)
        r = mask_util.merge(r) if isinstance(r, list) else r
        m = mask_util.decode(r).astype(bool)
        b = _bbox(m)
        if b is None:
            continue
        gt.append((b, np.ascontiguousarray(m[b[0]:b[1], b[2]:b[3]]), int(m.sum())))
    return gt


def pq_counts(pred, gt, iou_thr=0.5):
    used, tp, fp, s = set(), 0, 0, 0.0
    for i in range(1, int(pred.max()) + 1):
        pm = pred == i
        pb = _bbox(pm)
        if pb is None:
            continue
        py0, py1, px0, px1 = pb
        ps_ = pm[py0:py1, px0:px1]
        ps = int(ps_.sum())
        best, bj = 0.0, -1
        for j, (gb, gs_, gs) in enumerate(gt):
            if j in used:
                continue
            gy0, gy1, gx0, gx1 = gb
            y0, y1 = max(py0, gy0), min(py1, gy1)
            x0, x1 = max(px0, gx0), min(px1, gx1)
            if y1 <= y0 or x1 <= x0:
                continue
            inter = int(np.logical_and(
                ps_[y0 - py0:y1 - py0, x0 - px0:x1 - px0],
                gs_[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]).sum())
            if not inter:
                continue
            iou = inter / (ps + gs - inter)
            if iou > best:
                best, bj = iou, j
        if best >= iou_thr:
            tp += 1
            s += best
            used.add(bj)
        else:
            fp += 1
    return s, tp, fp, len(gt) - len(used)


def main():
    t0 = time.time()
    print(f"device: {DEV} | threads={torch.get_num_threads()} | cpu_count={os.cpu_count()}",
          flush=True)
    print(f"CHUNK_INDEX={CHUNK_INDEX} CHUNK_TOTAL={CHUNK_TOTAL}", flush=True)

    r34_paths = find_r34_checkpoints()
    print(f"ResNet34 (stage-A) checkpoints found: {len(r34_paths)}", flush=True)
    if len(r34_paths) != 5:
        raise SystemExit("STOP: kernel_sources did not mount the 5 stage-A weights.")
    stage_a = load_models(r34_paths, "resnet34")

    p1_path = find_phase1_refiner()
    print(f"phase-1 refiner checkpoint: {p1_path}", flush=True)
    if p1_path is None:
        raise SystemExit("STOP: kernel_sources did not mount refiner_best.pt.")
    refiner_p1 = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                          in_channels=3, classes=1)
    refiner_p1.load_state_dict(torch.load(p1_path, map_location=DEV))
    refiner_p1.eval()

    names, per_file, dims = load_annotations()
    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(names))
    val_names = [names[i] for i in idx[:N_VAL]]
    chunks = np.array_split(np.arange(len(val_names)), CHUNK_TOTAL)
    my = [val_names[i] for i in chunks[CHUNK_INDEX]]
    print(f"holdout total {len(val_names)}; chunk {CHUNK_INDEX}: {len(my)} images", flush=True)

    results = []
    for k, fn in enumerate(my, 1):
        g = cv2.imread(os.path.join(TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        prob, std_map = predict_prob_ens_std(stage_a, g)
        cand = instances(prob, **REF_PARAMS)
        lab, conf = refine_label_map_conf(g, cand, refiner_p1)

        pred_ids, pred_rles = [], []
        cand_recs = []
        for i in range(1, int(cand.max()) + 1):
            comp = cand == i
            if not comp.any():
                continue
            feats = candidate_features(g, prob, std_map, comp)
            rm = (lab == i)
            feats["refined_area"] = int(rm.sum())
            feats["refined_ratio"] = float(rm.sum() / max(comp.sum(), 1))
            cm, cmax = conf.get(i, (0.0, 0.0))
            feats["ref_mean_p"] = cm
            feats["ref_max_p"] = cmax
            cand_recs.append({"id": i, "feats": feats, "in_pred": bool(rm.any())})
            if rm.any():
                pred_ids.append(i)
                pred_rles.append(mask_util.encode(np.asfortranarray(rm.astype(np.uint8)))["counts"].decode("utf-8"))

        passes = official_detail(fn, per_file, dims, pred_rles)
        s, tp, fp, fnn = pq_counts(lab, val_gt(fn, per_file, dims))
        results.append({"name": fn, "pred_cand_ids": pred_ids, "cands": cand_recs,
                        "passes": passes, "proxy": [float(s), int(tp), int(fp), int(fnn)]})
        el = time.time() - t0
        print(f"  {k}/{len(my)}  {el:.0f}s  (~{el/k:.1f}s/image)  n_cand={len(cand_recs)} "
              f"n_pred={len(pred_rles)}", flush=True)
        json.dump({"chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL,
                   "status": "running" if k < len(my) else "done",
                   "elapsed_s": el, "results": results},
                  open(os.path.join(OUT, f"holdout_chunk{CHUNK_INDEX}.json"), "w"))
        json.dump({"status": "running" if k < len(my) else "done",
                   "progress": f"{k}/{len(my)}", "elapsed_s": el},
                  open(os.path.join(OUT, "gpu_summary.json"), "w"))

    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
