"""
Solar Filament Segmentation 2026 -- two-stage pipeline: a crop-refiner for
IoU quality (phase 1: refiner only, detection is left unchanged).

BACKGROUND. The PQ loss breakdown (code/metric/where_score_lost.py) shows two
untouched levers with the highest ceilings: mean IoU on TP pairs is only 0.635
(ceiling +57.5% if perfect) and FPs (ceiling +52.6%). The last three runs
(Tversky sweep, balanced-loss ensemble) showed that the "tune parameters/loss"
route saturates around holdout PQ 0.40 / Kaggle score 0.34.

EVIDENCE FROM TWO PUBLIC NOTEBOOKS that changed the expectations for this run.
The public notebook "lb-1st-solar-filament-segmentation-2026" turns out to be
just a plain U-Net baseline (misleading name, not a winning solution). The
notebook "best-score-magfilo-yolo11s-detect-crop-u-net" is a YOLO11s
(detection) -> crop-refiner U-Net (segmentation inside the crop) cascade with
fully open code. Its own local holdout PQ is only ~0.409 -- just 1.5% above
our ensemble+TTA (0.4026). So a two-stage architecture does NOT automatically
jump to 0.55 (the top-3 cluster); something else separates them, and it is
not visible in either public notebook. This run does NOT promise a jump to the
top 3 -- it looks for a measurable incremental improvement over 0.34, with the
same evaluation discipline used since the start of the project.

WHY NOT YOLO. Our stage-A (5-fold ensemble + TTA, PQ 0.4026) is already
comparable to the FINAL result of the reference YOLO+refiner system (0.409).
Replacing stage-A with a new detector adds engineering risk (ultralytics
dependency, bbox data format, separate training) for a thin potential gain
given the evidence above. Stage-A is used AS-IS via kernel_sources and is NOT
retrained.

PHASE 1 DESIGN: REFINER ONLY. For every instance ALREADY accepted by stage-A
at fixed parameters (thr=0.4/min_area=400/min_mean_prob=0.7 -- the same
reference point used in earlier kernels, reproducing PQ 0.4026 exactly), cut a
square crop around its bounding box and run a small U-Net that ONLY redraws
its shape. This isolates the "does the shape improve" test from "does
detection improve" -- FPs/FNs are still determined by stage-A as usual; the
refiner is not tasked with rejecting candidates (that needs hard negative
examples, deferred to a later phase if this phase proves helpful).

REFINER RECIPE FOLLOWS THE REFERENCE NOTEBOOK (already proven stable):
  - segmentation_models_pytorch.Unet(encoder_name="resnet34",
    encoder_weights="imagenet", in_channels=3, classes=1).
  - 3-channel input: [grayscale crop, CLAHE crop, seed mask x255], normalized
    with ImageNet mean/std.
  - SEED = JITTERED filled box, NOT the GT mask. This is the reference's key
    point: training seeds taken from the true GT mask make the refiner
    memorize the answer and fail at inference when given a coarse seed from
    stage-A. Jitter (shift up to 15% of box size, scale 0.85-1.25x, random
    padding) simulates real box imperfections.
  - Square crop: side ~1.5x the largest box dimension (minimum 96 px),
    resized to 512x512.
  - Loss: 0.45xBCE + 0.55xsoft-Dice.
  - PER-INSTANCE training data (not per-image) from the 657-image pool (the
    50-image holdout is sacrosanct, never touched) -- the COCO annotations
    already provide segmentation + bbox per instance per annotator pass, used
    directly.

EVALUATION & GATE (same pattern since the Tversky kernel): on the 50 holdout
images, stage-A at REF_PARAMS must reproduce PQ 0.4026 EXACTLY (integrity
check). For the SAME set of instances, redraw each mask with the refiner ->
recompute the official PQ + proxy. Submit only if
refined_PQ > baseline_PQ + noise floor 0.0016.
"""

import json
import math
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch"], check=True)
    import segmentation_models_pytorch as smp

OUT = "/kaggle/working"

PATCH = 512            # stage-A patch (sliding window, main U-Net)
STRIDE = 384
N_VAL = 50
REF_PARAMS = dict(thr=0.4, min_area=400, min_mean_prob=0.7)
REF_CLASSICAL_PQ = 0.19985337293988686
REF_PREVIOUS_ENS_PQ = 0.4026232478903755   # previous ensemble x5 + TTA, reproduced 2x before
NOISE_FLOOR = 0.0016

# ---- refiner configuration (follows the reference notebook) ----
CONTEXT = 1.5           # crop side ~ CONTEXT * largest bbox dimension
MIN_CROP = 96
PAD_RATIO = 0.40
REFINE_CROP = 512
REFINE_EPOCHS = 13
REFINE_BATCH = 8
REFINE_LR = 3e-4
REFINE_THR = 0.50       # refiner's own decision threshold (not the stage-A thr)
TRAIN_BUDGET = 3.0 * 3600
SEED = 42

BAR = "=" * 66
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- data discovery
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
    if not (ann and train_dir and test_dir):
        for root, _, files in os.walk("/kaggle/input"):
            print(root, len(files))
        raise SystemExit("data not found")
    return ann, train_dir, test_dir


ANN, TRAIN_IMG, TEST_IMG = discover()


def load_annotations():
    """Parse annotations into names/per_file/dims -- WITHOUT writing mask PNGs.

    Unlike build_masks() in the main U-Net training kernel: this kernel does
    not retrain stage-A, so merged per-image masks are not needed. The refiner
    needs PER-INSTANCE annotations (bbox + segmentation), which are available
    directly from per_file without merging.
    """
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
    print(f"annotations ready: {len(names)} unique images, "
          f"{sum(len(v) for v in per_file.values())} annotation entries", flush=True)
    return names, per_file, dims


# ------------------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True))


class ResUNet(nn.Module):
    """U-Net with a pretrained ResNet34 encoder -- the stage-A architecture,
    NOT retrained in this kernel. Copied verbatim from the ensemble kernel so
    old checkpoints still load with strict=True."""

    def __init__(self, arch="resnet34", pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        ctor, wcls, (c0, c1, c2, c3, c4) = {
            "resnet34": (tvm.resnet34, tvm.ResNet34_Weights,
                         (64, 64, 128, 256, 512)),
        }[arch]
        r = None
        if pretrained:
            try:
                r = ctor(weights=wcls.IMAGENET1K_V1)
            except Exception as e:
                print(f"WARNING: failed to download pretrained weights ({type(e).__name__}: "
                      f"{str(e)[:120]}) -> using RANDOM initialization", flush=True)
        if r is None:
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
def predict_prob_ens(models, gray, tta=False):
    tfms = TTA_TFMS if tta else TTA_TFMS[:1]
    acc = None
    for m in models:
        for _, f in tfms:
            p = predict_prob(m, np.ascontiguousarray(f(gray)))
            p = np.ascontiguousarray(f(p))
            acc = p if acc is None else acc + p
    return acc / (len(models) * len(tfms))


def find_r34_checkpoints():
    hits = []
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("unet_fold") and f.endswith(".pt"):
                hits.append(os.path.join(root, f))
    return sorted(hits)


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
            if len(batch) == 16:
                _flush(model, batch, coords, acc, cnt)
                batch, coords = [], []
    if batch:
        _flush(model, batch, coords, acc, cnt)
    return acc / np.maximum(cnt, 1e-6)


def _flush(model, batch, coords, acc, cnt):
    t = torch.from_numpy(np.stack(batch))[:, None].float().to(DEV)
    with torch.autocast("cuda", enabled=DEV.type == "cuda"):
        pr = torch.sigmoid(model(t)).float().cpu().numpy()[:, 0]
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


# ---------------------------------------------------------------------------- PQ
def _bbox(m):
    ys = np.flatnonzero(m.any(1))
    if not ys.size:
        return None
    xs = np.flatnonzero(m.any(0))
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


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


def pq(s, tp, fp, fn):
    d = tp + 0.5 * fp + 0.5 * fn
    return s / d if d else 0.0


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


def official_pq_holdout(val_names, per_file, dims, pred_by_stem):
    """OFFICIAL PQ metric: each annotator pass is scored SEPARATELY (not merged)."""
    tp_iou, fp, fn = [], 0, 0
    for name in val_names:
        by_pass = {}
        for a_ in per_file[name]:
            by_pass.setdefault(a_["image_id"], []).append(a_)
        pred_rles = pred_by_stem.get(name, [])
        h, w = dims[name]
        for _, anns in by_pass.items():
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
            if n_gt == 0:
                fp += n_pred
                continue
            if n_pred == 0:
                fn += n_gt
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
            tp_iou.extend(iou[hit].tolist())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    return (sum(tp_iou) / den if den > 0 else 0.0), tp, fp, fn


# ------------------------------------------------- classical baseline (for comparison)
def classical_pq(names, per_file, dims):
    P = dict(darkness_strict=0.9316482255232238, darkness_loose=0.99,
             ridge_pct=89, line_len=15, min_area=500, max_area=15000,
             dilate=1, open_k=3, close_k=3)
    S = T = FP = FN = 0.0
    for fn in names:
        gray = cv2.imread(os.path.join(TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        lab = _classical(gray, P)
        s, tp, fp, fnn = pq_counts(lab, val_gt(fn, per_file, dims))
        S, T, FP, FN = S + s, T + tp, FP + fp, FN + fnn
    return pq(S, T, FP, FN)


def _classical(gray, p):
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (51, 51)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    disk = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    h, w = disk.shape
    ff = disk.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    disk = (disk | (1 - ff))
    disk = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool)
    small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 51), (w, h),
                    interpolation=cv2.INTER_LINEAR).astype(np.float32)
    bg[bg < 1] = 1
    ratio = gray.astype(np.float32) / bg
    ridge = np.zeros(gray.shape, np.uint8)
    for a in np.linspace(0, 180, 6, endpoint=False):
        L = p["line_len"] | 1
        c = np.zeros((L, L), np.uint8)
        cv2.line(c, (0, L // 2), (L - 1, L // 2), 1, 1)
        M = cv2.getRotationMatrix2D((L // 2, L // 2), float(a), 1.0)
        k = cv2.warpAffine(c, M, (L, L), flags=cv2.INTER_NEAREST)
        np.maximum(ridge, cv2.subtract(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k), gray),
                   out=ridge)
    rt = np.percentile(ridge[disk], p["ridge_pct"])
    cand = ((ratio < p["darkness_strict"]) |
            ((ratio < p["darkness_loose"]) & (ridge >= rt))) & disk
    cand = cand.astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (p["open_k"],) * 2))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (p["close_k"],) * 2))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros(lab.shape, np.int32)
    nid = 1
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < p["min_area"] or a > p["max_area"]:
            continue
        out[lab == i] = nid
        nid += 1
    if p["dilate"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["dilate"] * 2 + 1,) * 2)
        g = np.zeros_like(out)
        for i in range(1, nid):
            m = cv2.dilate((out == i).astype(np.uint8), k)
            g[(m > 0) & (g == 0)] = i
        out = g
    return out


def load_models(items, arch):
    ms = []
    for it in items:
        m = UNet(arch, pretrained=False).to(DEV)
        m.load_state_dict(torch.load(it, map_location=DEV)
                          if isinstance(it, str) else it)
        m.eval()
        ms.append(m)
    return ms


# =========================================================================
#                       NEW PART: CROP-REFINER
# =========================================================================
def square_bounds_from_bbox(bbox, h, w, context=CONTEXT, minimum=MIN_CROP):
    """bbox = (x, y, bw, bh) in COCO format. Returns a square (x0,y0,x1,y1)."""
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
    """Seed = padded filled box -- NOT the GT mask (see module docstring:
    a seed from the true mask makes the refiner memorize and fail on real data)."""
    x, y, bw, bh = bbox
    px, py = bw * pad_ratio, bh * pad_ratio
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
    seed = np.zeros((h, w), np.uint8)
    seed[y0:y1, x0:x1] = 1
    return seed


def jitter_bbox(bbox, h, w, rng):
    """Perturb the GT box so the training seed resembles an imperfect stage-A
    box (not the precise GT box)."""
    x, y, bw, bh = map(float, bbox)
    x += float(rng.uniform(-0.15, 0.15) * bw)
    y += float(rng.uniform(-0.15, 0.15) * bh)
    sx = float(rng.uniform(0.85, 1.25))
    sy = float(rng.uniform(0.85, 1.25))
    cx, cy = x + bw / 2, y + bh / 2
    bw, bh = max(8.0, bw * sx), max(8.0, bh * sy)
    x, y = cx - bw / 2, cy - bh / 2
    x = float(np.clip(x, 0, w - 2))
    y = float(np.clip(y, 0, h - 2))
    bw = float(min(bw, w - x))
    bh = float(min(bh, h - y))
    pad = float(rng.uniform(PAD_RATIO * 0.5, PAD_RATIO * 1.5))
    return [x, y, bw, bh], pad


def ann_to_mask(ann, h, w):
    seg = ann.get("segmentation")
    if seg:
        r = mask_util.frPyObjects(seg, h, w)
        r = mask_util.merge(r) if isinstance(r, list) else r
        m = mask_util.decode(r)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return m.astype(np.uint8)
    x, y, bw, bh = map(int, ann["bbox"])
    m = np.zeros((h, w), np.uint8)
    m[y:y + bh, x:x + bw] = 1
    return m


def collect_instances(names_subset, per_file, dims):
    """(image path, bbox, segmentation) for every annotation (all passes) in
    the given set of images."""
    items = []
    for fn in names_subset:
        h, w = dims[fn]
        path = os.path.join(TRAIN_IMG, fn)
        for a in per_file[fn]:
            if not a.get("segmentation"):
                continue
            items.append((path, a["bbox"], a, h, w))
    return items


class RefineDataset(Dataset):
    """3-channel crop [gray, CLAHE, seed] -> target crop mask. One sample =
    one instance (not one image)."""

    def __init__(self, items, crop_size, augment):
        self.items = items
        self.crop_size = crop_size
        self.augment = augment
        self._cache = {}
        self._clahe = cv2.createCLAHE(2.0, (8, 8))

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        if path not in self._cache:
            self._cache[path] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return self._cache[path]

    def __getitem__(self, idx):
        path, bbox, ann, h, w = self.items[idx]
        img = self._load(path)
        target = ann_to_mask(ann, h, w)

        if self.augment:
            rng = np.random.default_rng(SEED + idx + random.randint(0, 10_000))
            box_j, pad_j = jitter_bbox(bbox, h, w, rng)
            seed_full = seed_from_box(h, w, box_j, pad_ratio=pad_j)
        else:
            seed_full = seed_from_box(h, w, bbox, pad_ratio=PAD_RATIO)

        x0, y0, x1, y1 = square_bounds_from_bbox(bbox, h, w)
        cs = self.crop_size
        img_c = cv2.resize(img[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR)
        tgt_c = cv2.resize(target[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
        seed_c = cv2.resize(seed_full[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
        clahe = self._clahe.apply(img_c)

        if self.augment and random.random() < 0.5:
            img_c, clahe = np.fliplr(img_c).copy(), np.fliplr(clahe).copy()
            tgt_c, seed_c = np.fliplr(tgt_c).copy(), np.fliplr(seed_c).copy()
        if self.augment and random.random() < 0.5:
            img_c, clahe = np.flipud(img_c).copy(), np.flipud(clahe).copy()
            tgt_c, seed_c = np.flipud(tgt_c).copy(), np.flipud(seed_c).copy()

        x = np.stack([img_c, clahe, seed_c * 255], 0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        x = (x - mean) / std
        y = torch.from_numpy(tgt_c.astype(np.float32))[None]
        return torch.from_numpy(x), y


def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum((2, 3))
    den = probs.sum((2, 3)) + targets.sum((2, 3))
    return 1.0 - ((2 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def batch_dice(logits, targets, thr=0.5):
    preds = (torch.sigmoid(logits) > thr).float()
    inter = (preds * targets).sum((2, 3))
    den = preds.sum((2, 3)) + targets.sum((2, 3))
    return ((2 * inter + 1e-6) / (den + 1e-6)).mean().item()


@torch.no_grad()
def refine_label_map(gray, coarse_lab, refiner, thr=REFINE_THR):
    """Replace each instance mask in coarse_lab with the version redrawn by
    the refiner, PRESERVING instance ids & count (clean comparison).

    If the refiner decides the crop is empty (the resulting mask is all zeros
    after thresholding), that id slot stays but has no pixels -- automatically
    counted as an FN by pq_counts, with no special handling needed.
    """
    h, w = gray.shape
    out = np.zeros_like(coarse_lab)
    clahe_op = cv2.createCLAHE(2.0, (8, 8))
    n_inst = int(coarse_lab.max())
    if n_inst == 0:
        return out
    for i in range(1, n_inst + 1):
        m = coarse_lab == i
        b = _bbox(m)
        if b is None:
            continue
        by0, by1, bx0, bx1 = b
        bbox = [bx0, by0, bx1 - bx0, by1 - by0]   # x,y,bw,bh
        seed_full = seed_from_box(h, w, bbox, pad_ratio=PAD_RATIO)
        x0, y0, x1, y1 = square_bounds_from_bbox(bbox, h, w)
        cs = REFINE_CROP
        img_c = cv2.resize(gray[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR)
        seed_c = cv2.resize(seed_full[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
        clahe_c = clahe_op.apply(img_c)
        x = np.stack([img_c, clahe_c, seed_c * 255], 0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        x = (x - mean) / std
        xt = torch.from_numpy(x)[None].to(DEV)
        with torch.autocast("cuda", enabled=DEV.type == "cuda"):
            prob = torch.sigmoid(refiner(xt)).float().cpu().numpy()[0, 0]
        crop_mask = (prob > thr).astype(np.uint8)
        full_mask = cv2.resize(crop_mask, (x1 - x0, y1 - y0),
                               interpolation=cv2.INTER_NEAREST)
        out[y0:y1, x0:x1][full_mask.astype(bool)] = i
    return out


def label_to_rles(lab):
    rles = []
    for i in range(1, int(lab.max()) + 1):
        m = np.asfortranarray((lab == i).astype(np.uint8))
        if m.sum():
            rles.append(mask_util.encode(m)["counts"].decode("utf-8"))
    return rles


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    print(f"device: {DEV} | "
          f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else ''}",
          flush=True)
    print(f"segmentation_models_pytorch version: {smp.__version__}", flush=True)

    r34_paths = find_r34_checkpoints()
    print(f"ResNet34 (stage-A) checkpoints found: {len(r34_paths)}", flush=True)
    if len(r34_paths) != 5:
        raise SystemExit(
            "STOP: kernel_sources did not mount the 5 stage-A ResNet34 weights. "
            "Check kernel-metadata.json.")
    stage_a = load_models(r34_paths, "resnet34")

    names, per_file, dims = load_annotations()
    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(names))
    val_names = [names[i] for i in idx[:N_VAL]]
    pool = [names[i] for i in idx[N_VAL:]]
    print(f"holdout (sacrosanct): {len(val_names)} | refiner pool: {len(pool)}",
          flush=True)

    # Selecting the best checkpoint needs a validation set SEPARATE from the
    # sacrosanct holdout (which must not be touched until the final
    # evaluation) and from the refiner training data (so the model is not
    # scored on the very images it learned from). Split at the IMAGE level,
    # not the instance level -- instances from the same image must not be
    # separated into train & val, otherwise validation leaks. Lesson learned:
    # the last epoch is not necessarily the best (the 72-epoch run overfit) --
    # the checkpoint is SELECTED, not taken as-is at the end of training.
    rng2 = np.random.default_rng(7)
    pool_idx = rng2.permutation(len(pool))
    n_rval = max(1, int(0.15 * len(pool)))
    refine_val_names = [pool[i] for i in pool_idx[:n_rval]]
    refine_trn_names = [pool[i] for i in pool_idx[n_rval:]]

    # ------------------------------------------------ train refiner (new phase)
    train_items = collect_instances(refine_trn_names, per_file, dims)
    val_items = collect_instances(refine_val_names, per_file, dims)
    print(f"refiner instance samples: {len(train_items)} training, "
          f"{len(val_items)} validation ({len(refine_val_names)} images)", flush=True)

    refiner = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                       in_channels=3, classes=1).to(DEV)
    opt = torch.optim.AdamW(refiner.parameters(), lr=REFINE_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=REFINE_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")

    train_ds = RefineDataset(train_items, REFINE_CROP, augment=True)
    train_dl = DataLoader(train_ds, batch_size=REFINE_BATCH, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_ds = RefineDataset(val_items, REFINE_CROP, augment=False)
    val_dl = DataLoader(val_ds, batch_size=REFINE_BATCH * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    print(f"\n{BAR}\nTRAINING REFINER ({REFINE_EPOCHS} epochs, budget "
          f"{TRAIN_BUDGET/3600:.1f} h)\n{BAR}", flush=True)
    ep_dur = []
    best = (-1.0, -1, None)
    ep = 0
    while ep < REFINE_EPOCHS:
        ep_t0 = time.time()
        refiner.train()
        losses, dices = [], []
        for x, y in train_dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                logits = refiner(x)
                loss = (0.45 * F.binary_cross_entropy_with_logits(logits, y)
                        + 0.55 * soft_dice_loss(logits, y))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            dices.append(batch_dice(logits.detach(), y))
        sched.step()

        refiner.eval()
        va_losses, va_dices = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                logits = refiner(x)
                loss = (0.45 * F.binary_cross_entropy_with_logits(logits, y)
                        + 0.55 * soft_dice_loss(logits, y))
                va_losses.append(loss.item())
                va_dices.append(batch_dice(logits, y))
        va_dice = float(np.mean(va_dices))
        star = ""
        if va_dice > best[0]:
            best = (va_dice, ep + 1,
                    {k: t.detach().cpu().clone() for k, t in refiner.state_dict().items()})
            star = "  <-- best"

        dur = time.time() - ep_t0
        ep_dur.append(dur)
        print(f"  epoch {ep+1:2d}/{REFINE_EPOCHS}  loss={np.mean(losses):.4f}  "
              f"dice={np.mean(dices):.4f}  val_loss={np.mean(va_losses):.4f}  "
              f"val_dice={va_dice:.4f}  duration={dur:.0f}s  "
              f"total={time.time()-t0:.0f}s{star}", flush=True)
        ep += 1

        elapsed = time.time() - t0
        avg_ep = sum(ep_dur) / len(ep_dur)
        if ep < REFINE_EPOCHS and elapsed + avg_ep > TRAIN_BUDGET:
            print(f"\nTIME GATE: stopping at epoch {ep} ({elapsed:.0f}s "
                  f"used).", flush=True)
            break

    print(f"\nusing best checkpoint: epoch {best[1]} (val_dice {best[0]:.4f})",
          flush=True)
    refiner.load_state_dict(best[2])
    torch.save(refiner.state_dict(), os.path.join(OUT, "refiner_best.pt"))
    refiner.eval()
    del train_dl, train_ds, val_dl, val_ds
    print(f"\nrefiner training finished | {time.time()-t0:.0f}s", flush=True)

    # -------------------------------------------------------- holdout evaluation
    print(f"\ncomputing stage-A + refiner on {len(val_names)} holdout "
          f"images...", flush=True)
    coarse_by_stem, refined_by_stem, gts = {}, {}, []
    coarse_labs, refined_labs = {}, {}
    for i, fn in enumerate(val_names, 1):
        g = cv2.imread(os.path.join(TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        prob = predict_prob_ens(stage_a, g, tta=True)
        lab_coarse = instances(prob, **REF_PARAMS)
        lab_refined = refine_label_map(g, lab_coarse, refiner)
        coarse_labs[fn] = lab_coarse
        refined_labs[fn] = lab_refined
        coarse_by_stem[fn] = label_to_rles(lab_coarse)
        refined_by_stem[fn] = label_to_rles(lab_refined)
        gts.append(val_gt(fn, per_file, dims))
        if i % 10 == 0:
            print(f"  {i}/{len(val_names)}  {time.time()-t0:.0f}s", flush=True)

    def score_proxy(pred_labs):
        S = T = FP = FN = 0.0
        for fn, g in zip(val_names, gts):
            s_, tp, fp_, fn_ = pq_counts(pred_labs[fn], g)
            S, T, FP, FN = S + s_, T + tp, FP + fp_, FN + fn_
        return pq(S, T, FP, FN), int(T), int(FP), int(FN)

    v_coarse_proxy, tp_c, fp_c, fn_c = score_proxy(coarse_labs)
    v_refined_proxy, tp_r, fp_r, fn_r = score_proxy(refined_labs)
    v_coarse_off, tpo_c, fpo_c, fno_c = official_pq_holdout(
        val_names, per_file, dims, coarse_by_stem)
    v_refined_off, tpo_r, fpo_r, fno_r = official_pq_holdout(
        val_names, per_file, dims, refined_by_stem)

    base = classical_pq(val_names, per_file, dims)
    ok_c = abs(base - REF_CLASSICAL_PQ) < 1e-9

    print(f"\n{BAR}\nHASIL\n{BAR}")
    print(f"  proxy coarse (stage-A only)    PQ={v_coarse_proxy:.4f}  "
          f"TP={tp_c} FP={fp_c} FN={fn_c}")
    print(f"  proxy refined (+ crop-refiner) PQ={v_refined_proxy:.4f}  "
          f"TP={tp_r} FP={fp_r} FN={fn_r}")
    print(f"  official coarse                PQ={v_coarse_off:.4f}  "
          f"TP={tpo_c} FP={fpo_c} FN={fno_c}")
    print(f"  official refined               PQ={v_refined_off:.4f}  "
          f"TP={tpo_r} FP={fpo_r} FN={fno_r}")

    print(f"\n{BAR}\nINTEGRITY CHECK\n{BAR}")
    print(f"  classical  {base:.17f} vs {REF_CLASSICAL_PQ:.17f}  "
          f"{'MATCH' if ok_c else '*** MISMATCH ***'}")
    # IMPORTANT: REF_PREVIOUS_ENS_PQ is a PROXY-metric number (from the
    # fixed_param_pq table of the balanced-ensemble/tta_r50 kernels -- those
    # kernels have NO official_pq_holdout). So it must be compared with
    # v_coarse_proxy, NOT v_coarse_off (official metric) -- the two use
    # different formulas and are not comparable at all. The noise floor is used
    # (not 1e-9) because this run also does not quantize probabilities to uint8
    # like tta_r50/balanced-ensemble, so the reproduction is not expected to
    # match exactly to the last digit. This mistake slipped through in the
    # first run (compared against v_coarse_off, wrong metric) -- the submission
    # gate below is NOT affected because it compares coarse-vs-refined within
    # the same metric & run, and does not use REF_PREVIOUS_ENS_PQ at all.
    ok_r = abs(v_coarse_proxy - REF_PREVIOUS_ENS_PQ) < NOISE_FLOOR
    print(f"  stage-A (proxy) {v_coarse_proxy:.17f} vs {REF_PREVIOUS_ENS_PQ:.17f}  "
          f"{'MATCH (within noise floor)' if ok_r else '*** MISMATCH ***'}")
    if not (ok_c and ok_r):
        print("  WARNING: the split, weights, or stage-A params have drifted. This "
              "run's numbers are NOT comparable with previous runs.", flush=True)
    print(BAR, flush=True)

    margin = v_refined_off - v_coarse_off
    passes_gate = margin > NOISE_FLOOR
    print(f"\n{BAR}\nSUBMISSION GATE\n{BAR}")
    print(f"  baseline (stage-A, official) PQ={v_coarse_off:.4f}")
    print(f"  refined  (+ refiner, official) PQ={v_refined_off:.4f}")
    print(f"  margin={margin:+.4f}  (noise floor {NOISE_FLOOR:.4f})  -> "
          f"{'SUBMIT' if passes_gate else 'DO NOT submit'}")
    print(BAR, flush=True)

    json.dump(dict(
        proxy_coarse_pq=float(v_coarse_proxy), proxy_refined_pq=float(v_refined_proxy),
        official_coarse_pq=float(v_coarse_off), official_refined_pq=float(v_refined_off),
        margin=float(margin), noise_floor=NOISE_FLOOR, submitted=bool(passes_gate),
        classical_pq=float(base), integrity_classical=bool(ok_c),
        integrity_stage_a=bool(ok_r), ref_previous_ens_pq=REF_PREVIOUS_ENS_PQ,
        n_train_instances=len(train_items), n_val=len(val_names),
        refine_epochs=len(ep_dur), refine_crop=REFINE_CROP,
    ), open(os.path.join(OUT, "gpu_summary.json"), "w"), indent=2)

    if not passes_gate:
        print(f"\nNOTE: the crop-refiner does NOT beat pure stage-A by a margin "
              f"above the noise floor {NOISE_FLOOR:.4f}. No submission was "
              f"created. The findings are still recorded in gpu_summary.json.",
              flush=True)
        print(f"total {time.time()-t0:.0f}s", flush=True)
        return

    # ------------------------------------------------------------- submission
    print("\nbuilding submission with stage-A + refiner...", flush=True)
    tf = sorted(f for f in os.listdir(TEST_IMG)
                if f.lower().endswith((".jpeg", ".jpg")))
    rows = []
    for k, fname in enumerate(tf, 1):
        g = cv2.imread(os.path.join(TEST_IMG, fname), cv2.IMREAD_GRAYSCALE)
        prob = predict_prob_ens(stage_a, g, tta=True)
        lab_coarse = instances(prob, **REF_PARAMS)
        lab_refined = refine_label_map(g, lab_coarse, refiner)
        iid = fname.split(".")[0]
        for i in range(1, int(lab_refined.max()) + 1):
            m = np.asfortranarray((lab_refined == i).astype(np.uint8))
            if not m.sum():
                continue
            rows.append({"filament_id": f"{iid}_{i}",
                         "segmentation_rle":
                             mask_util.encode(m)["counts"].decode("utf-8")})
        if k % 20 == 0:
            print(f"  test inference {k}/{len(tf)}  {time.time()-t0:.0f}s",
                  flush=True)

    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
    print(f"submission.csv: {len(sub)} filaments / {len(tf)} images", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
