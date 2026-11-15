"""Two-stage solar filament instance segmentation (inference).

Stage-A : 5-fold ResNet34 U-Net ensemble, dihedral TTA, sliding window -> probability map.
Stage-B : connected components of the thresholded map become candidates; a crop
          refiner (smp.Unet, ResNet34) redraws every candidate's mask from a
          box-shaped seed.
Limb    : instances whose centroid lies beyond 0.88 of the solar-disk radius are
          dropped (near-limb false positives; see report).
Scales  : optionally the probability maps of several image scales are averaged (multi-scale TTA).
Scorer  : optionally, a learned instance scorer (filament_scorer.py) drops the instances
          whose predicted matched IoU is below the PQ break-even.

Everything here is inference-only. Weights (`unet_fold1..5.pt`, `refiner_best.pt`): see the
repository README for the Kaggle dataset and SHA256 checksums. This is the earlier pipeline
(public leaderboard 0.37); the final submission (0.38) is filament_effv2s.py.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

# ----------------------------------------------------------------- configuration
PATCH = 512          # stage-A window (px)
STRIDE = 384         # stage-A window stride (px)

# candidate extraction (tuned on the held-out split; see report)
CAND_THR = 0.4
CAND_MIN_AREA = 400
CAND_MAX_AREA = 40000
CAND_MIN_MEAN_PROB = 0.7
CAND_CLOSE_K = 3

# refiner
REFINE_CROP = 512
REFINE_CONTEXT = 1.5     # crop side = context * longest bbox side
REFINE_MIN_CROP = 96
REFINE_SEED_PAD = 0.40   # seed box = bbox padded by 40 % of its size
REFINE_THR = 0.50
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

# near-limb filter: drop instances whose centroid is beyond LIMB_DROP_T * disk radius
# (chosen out of fold: +0.0065 PQ on 657 images, +0.0059 on the hold-out; see report)
LIMB_DROP_T = 0.88

# multi-scale TTA: scales of the released multi-scale setting (Pipeline(scales=MSC_SCALES))
MSC_SCALES = (0.75, 1.0, 1.25)

# dihedral TTA: (name, forward transform); every transform used here is its own inverse
TTA = (
    ("identity", lambda a: a),
    ("flip_lr", lambda a: a[:, ::-1]),
    ("flip_ud", lambda a: a[::-1, :]),
    ("rot180", lambda a: a[::-1, ::-1]),
)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_batch(device: torch.device) -> int:
    # CPU segfaulted on larger batches in earlier runs; keep it small there.
    return 16 if device.type == "cuda" else 4


# ------------------------------------------------------------------ stage-A model
def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResUNet(nn.Module):
    """U-Net with a ResNet34 encoder; 1 input channel, 1 logit channel."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        r = tvm.resnet34(weights=None)
        c0, c1, c2, c3, c4 = 64, 64, 128, 256, 512
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        up = lambda i, o: nn.ConvTranspose2d(i, o, 2, stride=2)
        self.u4, self.d4 = up(c4, 256), _block(256 + c3, 256)
        self.u3, self.d3 = up(256, 128), _block(128 + c2, 128)
        self.u2, self.d2 = up(128, 64), _block(64 + c1, 64)
        self.u1, self.d1 = up(64, 64), _block(64 + c0, 64)
        self.u0 = up(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def load_stage_a(paths: list[str], device: torch.device | None = None) -> list[nn.Module]:
    device = device or default_device()
    models = []
    for p in paths:
        m = ResUNet()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        models.append(m.eval().to(device))
    return models


def load_refiner(path: str, device: torch.device | None = None) -> nn.Module:
    import segmentation_models_pytorch as smp
    device = device or default_device()
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.eval().to(device)


# ------------------------------------------------------------------- stage-A infer
@torch.no_grad()
def predict_prob(model: nn.Module, gray: np.ndarray, device: torch.device,
                 batch: int) -> np.ndarray:
    """Sliding-window probability map of one model; every patch is z-scored on its own."""
    h, w = gray.shape
    ys = list(range(0, max(h - PATCH, 0) + 1, STRIDE))
    xs = list(range(0, max(w - PATCH, 0) + 1, STRIDE))
    if ys[-1] != h - PATCH:
        ys.append(h - PATCH)
    if xs[-1] != w - PATCH:
        xs.append(w - PATCH)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    tiles, coords = [], []

    def flush() -> None:
        t = torch.from_numpy(np.stack(tiles))[:, None].float().to(device)
        p = torch.sigmoid(model(t)).float().cpu().numpy()[:, 0]
        for (y, x), m in zip(coords, p):
            acc[y:y + PATCH, x:x + PATCH] += m
            cnt[y:y + PATCH, x:x + PATCH] += 1
        tiles.clear()
        coords.clear()

    for y in ys:
        for x in xs:
            p = gray[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0
            tiles.append((p - p.mean()) / (p.std() + 1e-6))
            coords.append((y, x))
            if len(tiles) == batch:
                flush()
    if tiles:
        flush()
    return acc / np.maximum(cnt, 1e-6)


def predict_ensemble(models: list[nn.Module], gray: np.ndarray, device: torch.device | None = None,
                     batch: int | None = None) -> np.ndarray:
    """Mean probability over models x dihedral TTA."""
    device = device or default_device()
    batch = batch or default_batch(device)
    acc = None
    for m in models:
        for _, f in TTA:
            p = predict_prob(m, np.ascontiguousarray(f(gray)), device, batch)
            p = np.ascontiguousarray(f(p))
            acc = p if acc is None else acc + p
    return acc / (len(models) * len(TTA))


def predict_scaled(models: list[nn.Module], gray: np.ndarray, scale: float, device: torch.device | None = None,
                   batch: int | None = None) -> np.ndarray:
    """predict_ensemble on the image resized by `scale` (area interpolation down, linear up), resampled
    back to the image size; scale 1.0 is exactly predict_ensemble."""
    if scale == 1.0:
        return predict_ensemble(models, gray, device, batch)
    size = int(round(gray.shape[0] * scale))
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    p = predict_ensemble(models, resized, device, batch)
    return cv2.resize(p, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def predict_multiscale(models: list[nn.Module], gray: np.ndarray, scales: tuple[float, ...] = MSC_SCALES,
                       device: torch.device | None = None, batch: int | None = None) -> np.ndarray:
    """Mean of the per-scale probability maps (out of fold, scales (0.75, 1.0, 1.25) beat one scale by +0.0117 PQ)."""
    return np.mean([predict_scaled(models, gray, s, device, batch) for s in scales], axis=0)


# -------------------------------------------------------------------- candidates
def extract_candidates(prob: np.ndarray, thr: float = CAND_THR, min_area: int = CAND_MIN_AREA,
                       min_mean_prob: float = CAND_MIN_MEAN_PROB) -> np.ndarray:
    """Label map (0 = background, 1..n = candidate instances) from a probability map."""
    m = (prob > thr).astype(np.uint8)
    if CAND_CLOSE_K:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CAND_CLOSE_K,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros(lab.shape, np.int32)
    nid = 1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > CAND_MAX_AREA:
            continue
        comp = lab == i
        if prob[comp].mean() < min_mean_prob:
            continue
        out[comp] = nid
        nid += 1
    return out


# ----------------------------------------------------------------------- refiner
def _bbox(mask: np.ndarray):
    ys = np.flatnonzero(mask.any(1))
    if not ys.size:
        return None
    xs = np.flatnonzero(mask.any(0))
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def square_bounds(bbox_xywh, h: int, w: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox_xywh
    cx, cy = x + bw / 2, y + bh / 2
    side = max(int(math.ceil(max(bw, bh) * REFINE_CONTEXT)), REFINE_MIN_CROP)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1, x0 = x1 - x0, 0
    if y0 < 0:
        y1, y0 = y1 - y0, 0
    if x1 > w:
        x0, x1 = x0 - (x1 - w), w
    if y1 > h:
        y0, y1 = y0 - (y1 - h), h
    return max(0, x0), max(0, y0), x1, y1


def seed_box(h: int, w: int, bbox_xywh, pad: float = REFINE_SEED_PAD) -> np.ndarray:
    x, y, bw, bh = bbox_xywh
    px, py = bw * pad, bh * pad
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
    seed = np.zeros((h, w), np.uint8)
    seed[y0:y1, x0:x1] = 1
    return seed


@torch.no_grad()
def refiner_prob(gray: np.ndarray, bbox_xywh, refiner: nn.Module, device: torch.device,
                 seed_pad: float = REFINE_SEED_PAD, flip: bool = False) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Refiner probability of one candidate box on its REFINE_CROP x REFINE_CROP crop, and the crop's
    bounds (x0, y0, x1, y1) in the image. `flip` runs the refiner on the left-right mirrored crop and
    mirrors the result back (used only for the scorer's stability features)."""
    h, w = gray.shape
    seed = seed_box(h, w, bbox_xywh, seed_pad)
    x0, y0, x1, y1 = square_bounds(bbox_xywh, h, w)
    cs = REFINE_CROP
    img = cv2.resize(gray[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR)
    sd = cv2.resize(seed[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
    x = np.stack([img, cv2.createCLAHE(2.0, (8, 8)).apply(img), sd * 255], 0).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    if flip:
        x = np.ascontiguousarray(x[:, :, ::-1])
    prob = torch.sigmoid(refiner(torch.from_numpy(x)[None].to(device))).float().cpu().numpy()[0, 0]
    return (np.ascontiguousarray(prob[:, ::-1]) if flip else prob), (x0, y0, x1, y1)


def refine_candidates(gray: np.ndarray, candidates: np.ndarray, refiner: nn.Module,
                      device: torch.device | None = None) -> np.ndarray:
    """Redraw every candidate with the crop refiner; returns a label map with the same ids."""
    device = device or default_device()
    out = np.zeros_like(candidates)
    for i in range(1, int(candidates.max()) + 1):
        box = _bbox(candidates == i)
        if box is None:
            continue
        y0_, y1_, x0_, x1_ = box
        prob, (x0, y0, x1, y1) = refiner_prob(gray, [x0_, y0_, x1_ - x0_, y1_ - y0_], refiner, device)
        crop_mask = (prob > REFINE_THR).astype(np.uint8)
        full = cv2.resize(crop_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        out[y0:y1, x0:x1][full.astype(bool)] = i
    return out


# ---------------------------------------------------------------- near-limb filter
def estimate_disk(gray: np.ndarray) -> tuple[float, float, float]:
    """Solar disk (centre x, centre y, radius) from an Otsu threshold: largest component,
    holes (dark filaments, sunspots) filled, radius from its area."""
    _, b = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    m = m | (1 - ff)
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(m.sum() / np.pi))


def drop_near_limb(labels: np.ndarray, gray: np.ndarray, t: float = LIMB_DROP_T) -> np.ndarray:
    """Remove every instance whose centroid lies farther than t * disk radius from the disk centre."""
    cx, cy, r = estimate_disk(gray)
    out = labels.copy()
    for i in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == i)
        if xs.size and math.hypot(xs.mean() - cx, ys.mean() - cy) / r > t:
            out[labels == i] = 0
    return out


# ------------------------------------------------------------------ end-to-end API
@dataclass
class Pipeline:
    stage_a: list[nn.Module]
    refiner: nn.Module
    device: torch.device
    limb_drop: float | None = LIMB_DROP_T   # None disables the near-limb filter
    scorer: object | None = None            # filament_scorer.InstanceScorer; None disables the scorer
    scales: tuple[float, ...] = (1.0,)       # stage-A scales; MSC_SCALES enables multi-scale TTA

    def segment(self, gray: np.ndarray) -> np.ndarray:
        """uint8 grayscale image -> instance label map (0 = background)."""
        prob = predict_multiscale(self.stage_a, gray, self.scales, self.device)
        candidates = extract_candidates(prob)
        labels = refine_candidates(gray, candidates, self.refiner, self.device)
        if self.limb_drop is not None:
            labels = drop_near_limb(labels, gray, self.limb_drop)
        if self.scorer is not None:
            labels = self.scorer.filter(gray, labels, candidates, prob)
        return labels


def label_map_to_rles(labels: np.ndarray) -> list[str]:
    """One compressed COCO RLE `counts` string per non-empty instance."""
    rles = []
    for i in range(1, int(labels.max()) + 1):
        m = np.asfortranarray((labels == i).astype(np.uint8))
        if m.any():
            rles.append(mask_util.encode(m)["counts"].decode("ascii"))
    return rles


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def validate_submission_rows(rows: list[dict], test_ids: set[str], size=(2048, 2048)) -> dict:
    """Format checks: unique ids, known images, non-empty masks of the expected size."""
    ids = [r["filament_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate filament_id")
    stems = {i.rsplit("_", 1)[0] for i in ids}
    unknown = stems - test_ids
    if unknown:
        raise ValueError(f"unknown image ids: {sorted(unknown)[:3]}")
    for r in rows:
        rle = {"size": list(size), "counts": r["segmentation_rle"].encode("ascii")}
        if mask_util.area(rle) == 0:
            raise ValueError(f"empty mask: {r['filament_id']}")
    return {"rows": len(rows), "images_with_predictions": len(stems),
            "images_without_predictions": len(test_ids - stems)}
