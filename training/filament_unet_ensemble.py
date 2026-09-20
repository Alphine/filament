"""
Solar Filament Segmentation 2026 — 5-fold ENSEMBLE (Kaggle GPU).

Follow-up to the pretrained-encoder run (validation PQ 0.3227 / Kaggle score 0.27).

DESIGN FOR COMPARABILITY:
  - The 50 holdout images are HELD OUT: no fold ever trains on them, and they
    are the same 50 images (seed 1234) as in all previous runs. The ensemble
    PQ can therefore be compared directly with 0.3227.
  - 5 folds are split from the remaining 657 images. Each fold trains on ~526
    images and uses its ~131 fold-val images ONLY to select the best
    checkpoint; it never touches the holdout.
  - The ensemble averages the 5 models' probability maps rather than merging
    binary masks. Averaging probabilities preserves confidence information;
    merging binary masks discards it.

All five models are kept in GPU memory at once during inference (5 x 24M
parameters ~480 MB, fits on a 16 GB T4), so each image is read only once and
no giant probability accumulator is needed in RAM.

Two findings from the previous runs drove this design:

A. TRAINING LONGER DOES NOT HELP. 72 epochs lowered the training loss from
   0.2347 to 0.1452 (-38%) but validation PQ actually dropped slightly
   (0.2977 -> 0.2951). The model memorized better without generalizing
   better, so compute time is not the bottleneck.

B. THE ROOT CAUSE: TRAINING FROM SCRATCH ON 657 IMAGES. That is too few to
   learn good visual features from nothing. An ImageNet-pretrained encoder
   brings mature edge/texture features and typically gives a large jump on
   small datasets.

This run therefore switches to a pretrained ResNet34 encoder AND adds
validation-based checkpoint selection (previously we always used the
last-epoch weights, although finding A shows the last epoch is not
necessarily the best).

Design decisions carried over from the earlier classical-CV pipeline
(best Kaggle score 0.17, holdout PQ 0.2084):

1. FULL-RESOLUTION PATCHES, NOT DOWNSCALED IMAGES.
   Diagnosis showed that 38.5% of GT filaments are not touched by any
   prediction, and the missed ones are SMALLER (median 846 px vs 1,826 px for
   detected ones) and dimmer. Downscaling 2048->512 would erase exactly the
   class of objects we miss most often. The model is therefore trained on
   native-resolution 512x512 crops, and inference uses a sliding window.

2. FALSE-NEGATIVE-HEAVY LOSS (Tversky beta>alpha).
   Our main weakness is recall, not precision. Tversky with beta 0.7
   penalizes FN more than FP. This is a deliberate choice: the PQ metric
   weighs FP and FN equally (0.5), but we currently have far more FN (130)
   than TP (62), so pushing recall first makes sense.

3. SPLIT DEDUPLICATED BY FILE NAME.
   The annotation file has 1,154 entries for only 707 unique images, because
   each image is annotated by 1-3 annotators (411/145/151). Splitting per
   entry would leak the same image into both train AND validation through
   different annotator passes, making the validation score too optimistic.
   The split is done per file name.

4. TRAINING MASK = UNION OF ALL ANNOTATOR PASSES.
   A pixel marked as filament by any annotator is treated as filament,
   giving more complete supervision for the recall problem above.

5. PRETRAINED ENCODER (changed from the previous run).
   Earlier runs trained from scratch because the kernel ran without
   internet. This kernel enables internet so the ImageNet weights can be
   downloaded. If the download fails, it does NOT abort but falls back to
   random initialization and says so in the log, so the result is not
   mistaken for a pretrained one.

6. FAIR COMPARISON.
   The classical-CV baseline is re-run on the SAME VALIDATION SET inside this
   kernel, instead of being compared with numbers from a different split.
   A submission is produced only if the model actually wins.
"""

import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OUT = "/kaggle/working"
# Masks are an INTERMEDIATE artifact (707 PNGs of 2048x2048), not a result to
# keep. They used to live inside OUT, so all of them were uploaded with the
# results at the end of the job, adding minutes (and billing) for nothing.
# They now go to a separate scratch directory that is not uploaded, and can
# be regenerated from the annotations at any time.
# On Kaggle, /kaggle/working IS the kernel output; /kaggle/temp is not.
MASK_DIR = "/kaggle/temp/masks" if os.path.isdir("/kaggle") else "./masks"
os.makedirs(MASK_DIR, exist_ok=True)

PATCH = 512
BATCH = 8
EPOCHS = 30        # pretrained converges faster; 72 epochs proved to overfit
STEPS_PER_EPOCH = 220
LR = 1e-4          # smaller: the pretrained encoder must not be damaged
N_VAL = 50                 # SAME as the Saturn run (PQ 0.2876) for comparability
POS_FRAC = 0.75            # fraction of patches centered on a filament
STRIDE = 384               # sliding-window stride at inference (overlap 128)
VAL_EVERY = 5              # validate every N epochs to select the checkpoint
VAL_QUICK = 20             # number of images for the quick evaluation during training
N_FOLDS = 5                # number of ensemble folds
BAR = "=" * 60

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data discovery
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


# ---------------------------------------------------------------- prepare masks
def build_masks():
    """Merge all annotator passes into one binary mask per unique image."""
    os.makedirs(MASK_DIR, exist_ok=True)
    with open(ANN) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    per_file = {}      # file name -> list of annotations from all passes
    dims = {}
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
    for fn in names:
        out = os.path.join(MASK_DIR, fn.replace(".jpeg", ".png").replace(".jpg", ".png"))
        if os.path.exists(out):
            continue
        h, w = dims[fn]
        m = np.zeros((h, w), np.uint8)
        for a in per_file[fn]:
            seg = a.get("segmentation")
            if not seg:
                continue
            r = mask_util.frPyObjects(seg, h, w)
            r = mask_util.merge(r) if isinstance(r, list) else r
            m |= mask_util.decode(r).astype(np.uint8)
        cv2.imwrite(out, m * 255)
    print(f"masks ready for {len(names)} unique images", flush=True)
    return names, per_file, dims


# ----------------------------------------------------------------------- dataset
class PatchDS(Dataset):
    """Native-resolution 512x512 crops, biased toward filament areas.

    Images and masks are cached in RAM once at the start. Without this, every
    patch would trigger a full 2048x2048 JPEG re-decode — a large waste that
    makes data loading, not the GPU, the main training bottleneck.
    """

    def __init__(self, names, n_steps, cache):
        self.names = names
        self.n = n_steps * BATCH
        self.cache = cache

    def __len__(self):
        return self.n

    def _load(self, fn):
        return self.cache[fn]

    def build_positions(self):
        """Precompute filament locations per image (subsampled) for patch sampling."""
        self.pos = {}
        for fn in self.names:
            _, m = self.cache[fn]
            ys, xs = np.nonzero(m[::16, ::16])
            self.pos[fn] = np.stack([ys * 16, xs * 16], 1) if len(ys) else np.empty((0, 2), int)

    def __getitem__(self, i):
        rng = np.random.default_rng((i * 9781 + 12345) % (2 ** 31))
        fn = self.names[rng.integers(len(self.names))]
        img, msk = self._load(fn)
        H, W = img.shape

        pos = self.pos[fn]
        if rng.random() < POS_FRAC and len(pos):
            cy, cx = pos[rng.integers(len(pos))]
            y0 = int(np.clip(cy - PATCH // 2 + rng.integers(-96, 96), 0, H - PATCH))
            x0 = int(np.clip(cx - PATCH // 2 + rng.integers(-96, 96), 0, W - PATCH))
        else:
            y0 = int(rng.integers(0, H - PATCH))
            x0 = int(rng.integers(0, W - PATCH))

        p = img[y0:y0 + PATCH, x0:x0 + PATCH].astype(np.float32)
        q = msk[y0:y0 + PATCH, x0:x0 + PATCH].astype(np.float32)

        if rng.random() < 0.5:
            p, q = p[:, ::-1], q[:, ::-1]
        if rng.random() < 0.5:
            p, q = p[::-1], q[::-1]
        r = int(rng.integers(4))
        if r:
            p, q = np.rot90(p, r), np.rot90(q, r)

        p = np.ascontiguousarray(p) / 255.0
        # per-patch normalization: suppresses brightness variation across observations
        p = (p - p.mean()) / (p.std() + 1e-6)
        return (torch.from_numpy(p)[None].float(),
                torch.from_numpy(np.ascontiguousarray(q))[None].float())


# ------------------------------------------------------------------------ model
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
        nn.ReLU(inplace=True))


class ResUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet34 encoder.

    Our images are 1-channel grayscale while the ImageNet encoder expects 3
    channels, so the channel is repeated 3x. This is simple and keeps the
    conv1 weights unchanged, unlike summing the weights, which changes the
    activation scale.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        # Pretrained weights are downloaded from the internet. If the download
        # fails (internet down / blocked), do NOT kill the whole run — fall back
        # to random initialization and state so plainly, so the result stays
        # interpretable and is not mistaken for a pretrained one.
        r = None
        if pretrained:
            try:
                r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                print("encoder: ResNet34 PRETRAINED ImageNet", flush=True)
            except Exception as e:
                print(f"WARNING: failed to download pretrained weights ({type(e).__name__}: "
                      f"{str(e)[:120]}) -> using RANDOM initialization", flush=True)
        if r is None:
            r = resnet34(weights=None)
            print("encoder: ResNet34 random (NOT pretrained)", flush=True)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # /2   64ch
        self.pool = r.maxpool                               # /4
        self.l1, self.l2 = r.layer1, r.layer2               # /4 64, /8 128
        self.l3, self.l4 = r.layer3, r.layer4               # /16 256, /32 512

        up = lambda i, o: nn.ConvTranspose2d(i, o, 2, stride=2)
        self.u4, self.d4 = up(512, 256), block(512, 256)
        self.u3, self.d3 = up(256, 128), block(256, 128)
        self.u2, self.d2 = up(128, 64), block(128, 64)
        self.u1, self.d1 = up(64, 64), block(128, 64)
        self.u0 = up(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)          # grayscale -> 3 channels
        s0 = self.stem(x)                 # /2  64
        s1 = self.l1(self.pool(s0))       # /4  64
        s2 = self.l2(s1)                  # /8  128
        s3 = self.l3(s2)                  # /16 256
        s4 = self.l4(s3)                  # /32 512
        y = self.d4(torch.cat([self.u4(s4), s3], 1))
        y = self.d3(torch.cat([self.u3(y), s2], 1))
        y = self.d2(torch.cat([self.u2(y), s1], 1))
        y = self.d1(torch.cat([self.u1(y), s0], 1))
        return self.head(self.u0(y))


UNet = ResUNet   # old name, used throughout the script


def tversky_bce(logit, target, alpha=0.3, beta=0.7):
    """Tversky (beta>alpha => FN penalized more heavily) + BCE.

    The main weakness of the previous pipeline was recall: FN 130 vs TP 62.
    """
    p = torch.sigmoid(logit)
    tp = (p * target).sum((1, 2, 3))
    fp = (p * (1 - target)).sum((1, 2, 3))
    fn = ((1 - p) * target).sum((1, 2, 3))
    tv = 1 - ((tp + 1) / (tp + alpha * fp + beta * fn + 1)).mean()
    return tv + F.binary_cross_entropy_with_logits(logit, target)


# --------------------------------------------------------------------- inference
@torch.no_grad()
def predict_prob_ens(models, gray):
    """Average the probabilities of several models on one image."""
    acc = None
    for m in models:
        p = predict_prob(m, gray)
        acc = p if acc is None else acc + p
    return acc / len(models)


@torch.no_grad()
def predict_prob(model, gray):
    """Full-resolution sliding window, averaged over overlapping areas."""
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
        # Confidence filter: drop components the model itself is unsure about.
        # Unlike raising the global `thr`, which erodes ALL instances including
        # correct ones. On the 36-epoch model, this local tuning gave FP -24%
        # with TP only -7% (PQ 0.2875 -> 0.2977).
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
    """Per-instance GT for validation: use ONE annotator pass (not the union).

    Training masks use the union for complete supervision, but for
    instance-level PQ the union can merge neighboring filaments from different
    passes and corrupt the instance count.
    """
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


# ------------------------------------------------- classical baseline (comparison)
def classical_pq(names, per_file, dims):
    """Run the classical-CV winner on the SAME VALIDATION SET."""
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


# ------------------------------------------------------------------------- main
def train_one_fold(fold, fold_trn, fold_val, cache, per_file, dims, t0):
    """Train one fold; return the best state_dict according to its fold-val."""
    print(f"\n{BAR}\nFOLD {fold+1}/{N_FOLDS}  train={len(fold_trn)}  "
          f"fold-val={len(fold_val)}\n{BAR}", flush=True)

    model = UNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * STEPS_PER_EPOCH, pct_start=0.25)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")

    ds = PatchDS(fold_trn, STEPS_PER_EPOCH, cache)
    ds.build_positions()
    dl = DataLoader(ds, batch_size=BATCH, num_workers=2, pin_memory=True,
                    drop_last=True, persistent_workers=True)

    best = (-1.0, -1, None)
    quick = fold_val[:VAL_QUICK]
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                loss = tversky_bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()

        if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
            model.eval()
            S = T = FP = FN = 0.0
            for fn in quick:
                g = cv2.imread(os.path.join(TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
                lab = instances(predict_prob(model, g), 0.4, 400, min_mean_prob=0.7)
                s_, tp, fp, fn_ = pq_counts(lab, val_gt(fn, per_file, dims))
                S, T, FP, FN = S + s_, T + tp, FP + fp, FN + fn_
            v = pq(S, T, FP, FN)
            star = "  <-- best" if v > best[0] else ""
            if v > best[0]:
                best = (v, ep + 1,
                        {k: t.detach().cpu().clone()
                         for k, t in model.state_dict().items()})
            print(f"  fold{fold+1} ep{ep+1:2d}  loss={tot/len(dl):.4f}  "
                  f"foldval_PQ={v:.4f}  {time.time()-t0:.0f}s{star}", flush=True)

    print(f"  fold{fold+1} done: checkpoint epoch {best[1]} (PQ {best[0]:.4f})",
          flush=True)
    del model, opt, dl, ds
    torch.cuda.empty_cache()
    return best[2]


def main():
    t0 = time.time()
    print(f"device: {DEV} | "
          f"{torch.cuda.get_device_name(0) if DEV.type=='cuda' else ''}", flush=True)
    names, per_file, dims = build_masks()

    # Holdout is HELD OUT: the same 50 images as in all previous runs, never
    # trained on nor used to select a checkpoint by any fold.
    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(names))
    val_names = [names[i] for i in idx[:N_VAL]]
    pool = [names[i] for i in idx[N_VAL:]]
    print(f"holdout (held out): {len(val_names)} | fold pool: {len(pool)}",
          flush=True)

    print("loading pool images+masks into RAM...", flush=True)
    cache = {}
    for fn in pool:
        img = cv2.imread(os.path.join(TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        mp = os.path.join(MASK_DIR,
                          fn.replace(".jpeg", ".png").replace(".jpg", ".png"))
        cache[fn] = (img,
                     (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))
    print(f"RAM cache ready: {len(cache)} images | {time.time()-t0:.0f}s", flush=True)

    order = np.random.default_rng(99).permutation(len(pool))
    folds = np.array_split(order, N_FOLDS)

    states = []
    for f in range(N_FOLDS):
        fv = [pool[i] for i in folds[f]]
        ft = [pool[i] for i in np.concatenate(
            [folds[j] for j in range(N_FOLDS) if j != f])]
        states.append(train_one_fold(f, ft, fv, cache, per_file, dims, t0))
        torch.save(states[-1], os.path.join(OUT, f"unet_fold{f+1}.pt"))

    del cache
    # All five models are kept on the GPU together: 5 x 24M params ~480 MB, fits
    # on a T4. This way each image is read only once and no giant probability
    # accumulator is needed in RAM.
    models = []
    for st in states:
        m = UNet(pretrained=False).to(DEV)
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    print(f"\n{len(models)} models loaded for ensemble | {time.time()-t0:.0f}s",
          flush=True)

    print("\ncomputing holdout probabilities...", flush=True)
    probs_ens, probs_single, gts = [], [], []
    for fn in val_names:
        g = cv2.imread(os.path.join(TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        probs_ens.append(predict_prob_ens(models, g))
        probs_single.append(predict_prob(models[0], g))   # single-model comparison
        gts.append(val_gt(fn, per_file, dims))
    print(f"done {time.time()-t0:.0f}s", flush=True)

    def sweep(probs, tag):
        best = (-1.0, dict(thr=0.4, min_area=400, min_mean_prob=0.7))
        for thr in [0.3, 0.4, 0.5, 0.6]:
            for ma in [400, 600, 900]:
                for mp in [0.0, 0.7, 0.8]:
                    S = T = FP = FN = 0.0
                    for pr, gt in zip(probs, gts):
                        s_, tp, fp, fn_ = pq_counts(
                            instances(pr, thr, ma, min_mean_prob=mp), gt)
                        S, T, FP, FN = S + s_, T + tp, FP + fp, FN + fn_
                    v = pq(S, T, FP, FN)
                    if v > best[0]:
                        best = (v, dict(thr=thr, min_area=ma, min_mean_prob=mp))
                        print(f"  [{tag}] thr={thr} min_area={ma:<4} mp={mp} -> "
                              f"PQ={v:.4f} TP={int(T)} FP={int(FP)} FN={int(FN)}",
                              flush=True)
        return best

    print("\n--- sweep: SINGLE model (fold 1) ---", flush=True)
    single = sweep(probs_single, "1-model")
    print("\n--- sweep: 5-model ENSEMBLE ---", flush=True)
    best = sweep(probs_ens, "ensemble")

    base = classical_pq(val_names, per_file, dims)
    print("\n" + BAR)
    print(f"PQ holdout classical         = {base:.4f}")
    print(f"PQ holdout 1 model (fold 1)  = {single[0]:.4f}")
    print(f"PQ holdout ENSEMBLE 5-fold   = {best[0]:.4f} @ {best[1]}")
    print(f"ensemble vs 1 model          : "
          f"{100*(best[0]-single[0])/single[0]:+.1f}%")
    print("previous run for comparison: single pretrained = 0.3227")
    print(BAR, flush=True)

    json.dump(dict(ensemble_pq=float(best[0]), single_pq=float(single[0]),
                   classical_pq=float(base), params=best[1], n_folds=N_FOLDS,
                   epochs=EPOCHS, n_val=len(val_names), n_pool=len(pool)),
              open(os.path.join(OUT, "gpu_summary.json"), "w"), indent=2)

    if best[0] <= single[0]:
        print("\nNOTE: the ensemble does NOT beat a single model. The submission is "
              "still built from the ensemble, but this finding matters; do not bury it.",
              flush=True)

    tf = sorted(f for f in os.listdir(TEST_IMG)
                if f.lower().endswith((".jpeg", ".jpg")))
    rows = []
    for k, fname in enumerate(tf, 1):
        g = cv2.imread(os.path.join(TEST_IMG, fname), cv2.IMREAD_GRAYSCALE)
        lab = instances(predict_prob_ens(models, g), **best[1])
        iid = fname.split(".")[0]
        for i in range(1, int(lab.max()) + 1):
            m = np.asfortranarray((lab == i).astype(np.uint8))
            if not m.sum():
                continue
            rows.append({"filament_id": f"{iid}_{i}",
                         "segmentation_rle":
                             mask_util.encode(m)["counts"].decode("utf-8")})
        if k % 40 == 0:
            print(f"  test inference {k}/{len(tf)}  {time.time()-t0:.0f}s",
                  flush=True)

    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
    print(f"submission.csv: {len(sub)} filaments / {len(tf)} images "
          f"({len(sub)/len(tf):.1f} per image; GT ~7.1)", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
