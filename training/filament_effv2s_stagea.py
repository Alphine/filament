"""
Solar Filament Segmentation 2026 -- a stronger stage-A trained by us: U-Net with an EfficientNetV2-S encoder (Kaggle GPU).

Why: the stage-A map is our bottleneck (HANDOFF 4.3), and a public model card (dariushafshar/solar-filament-unet-
weights, 17 October) reports that this encoder with a longer schedule scores public 0.39 where ResNet34 scores 0.37.
We train our own weights with our own code on OUR splits (hold-out 50 images, seed 1234; 5 folds of the 657-image pool,
seed 99), so the result is comparable with every out-of-fold number in this repository. The recipe follows the public
card's description (disclosed in the report): 512 px crops at full resolution, 70 % centred on a filament; target = mean
of the annotator passes' union masks; per-image normalisation by the mean/SD inside a 950 px disk; rot90 + flips, gamma
and contrast/brightness jitter; BCE on the soft target + soft Dice; AdamW 3e-4, weight decay 1e-4, 300 warm-up steps then
cosine to 1 %; batch 16 as two accumulated micro-batches of 8; 10,500 steps; float16 autocast. Inference: full image,
4 flip/rotation views. One fold per kernel (env FOLD).

Compute safety -- STOP ON ERROR (any exception is logged and re-raised: the kernel ends):
  * FAIL FAST before training: split identical to the repository's; two real optimiser steps; a checkpoint save/load
    round trip; one full-image forward pass (the inference path and its memory) -- all in the first minutes.
  * non-finite loss stops the run. NOT LEARNING stops the run: the mean loss of steps 801-1000 must be below 0.9 x the
    mean loss of steps 1-100.
  * checkpoint (model, optimiser, scheduler, scaler, step, loss history) every CKPT_EVERY steps, written atomically;
    a checkpoint or a finished fold found under INPUT_ROOT (the previous run attached as a kernel source) is resumed /
    skipped. Samples are a pure function of (seed, fold, sample index), so a resumed run sees the same data.
  * TIME BUDGET (env BUDGET_S, whole kernel): training stops at a checkpoint when the remaining time would not cover
    the next checkpoint interval plus inference; status "stopped" -- attach the output and rerun to continue.
Outputs: effv2s_fold<F>.json (log, status), effv2s_fold<F>.pt (final weights, only when training finished),
effv2s_ckpt_fold<F>.pt (only while unfinished), effv2s_maps_fold<F>_{oof,holdout,test}.npz (uint8 probability maps).
Env SYNTH=1: tiny synthetic data, CPU, a few steps -- the local smoke test of every code path.
"""

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

T_START = time.time()
SYNTH = os.environ.get("SYNTH") == "1"
FOLD = int(os.environ.get("FOLD", "1"))
STEPS = int(os.environ.get("STEPS", "10500"))
BATCH, MICRO = int(os.environ.get("BATCH", "16")), int(os.environ.get("MICRO", "8"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "500"))
BUDGET_S = float(os.environ.get("BUDGET_S", str(4.0 * 3600)))
INFER_RESERVE_S = float(os.environ.get("INFER_RESERVE_S", "1500"))
INPUT_ROOT = os.environ.get("INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "3"))
ENCODER = os.environ.get("ENCODER", "tu-tf_efficientnetv2_s")
PRETRAINED = os.environ.get("PRETRAINED", "imagenet")
SEED = 20261110
CROP, POS_FRAC, LR, WD, WARMUP = 512, 0.7, 3e-4, 1e-4, 300
N_HOLD, N_FOLDS, TTA = 50, 5, 4
H = W = 2048
DISK_R = 950
TAG = f"effv2s_fold{FOLD}"
assert BATCH % MICRO == 0 and 1 <= FOLD <= N_FOLDS

try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"], check=True)
    import segmentation_models_pytorch as smp

import cv2
import numpy as np
import pycocotools.mask as mu
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, ".")
import filament_metric as fm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"
YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2
D4 = [(0, False), (0, True), (2, False), (2, True)]


class Log:
    def __init__(self):
        self.path = os.path.join(OUT, f"{TAG}.json")
        self.d = {"fold": FOLD, "seed": SEED, "encoder": ENCODER, "steps": STEPS, "batch": BATCH, "status": "running",
                  "events": []}

    def note(self, msg):
        print(f"[{time.time() - T_START:7.0f}s] fold{FOLD} {msg}", flush=True)
        self.d["events"].append([round(time.time() - T_START), msg])
        self.save()

    def save(self):
        json.dump(self.d, open(self.path + ".tmp", "w"), indent=1)
        os.replace(self.path + ".tmp", self.path)


def synth_data(root):
    """12 synthetic 2048 px images with elongated dark blobs, one or two annotator passes each (smoke test only)."""
    rng = np.random.default_rng(0)
    os.makedirs(f"{root}/train_images", exist_ok=True)
    os.makedirs(f"{root}/test_images", exist_ok=True)
    images, anns = [], []
    for i in range(12):
        name = f"2024010100{i:04d}Bh.jpeg"
        img = np.full((H, W), 20, np.uint8)
        cv2.circle(img, (W // 2, H // 2), 930, 150, -1)
        polys = []
        for _ in range(4):
            cx, cy = rng.integers(500, 1500, 2)
            box = cv2.boxPoints(((float(cx), float(cy)), (160.0, 30.0), float(rng.uniform(0, 180))))
            cv2.fillPoly(img, [box.astype(np.int32)], 60)
            polys.append(box)
        cv2.imwrite(f"{root}/train_images/{name}", img)
        if i < 2:
            cv2.imwrite(f"{root}/test_images/{name}", img)
        for p in range(1 + i % 2):
            iid = len(images) + 1
            images.append({"id": iid, "file_name": name, "height": H, "width": W})
            for box in polys[: 4 - p]:
                anns.append({"id": len(anns) + 1, "image_id": iid, "segmentation": [[float(v) for v in box.ravel()]]})
    json.dump({"images": images, "annotations": anns}, open(f"{root}/Annotations_train.json", "w"))


def locate():
    if SYNTH:
        root = os.path.join(OUT, "synth_input")
        if not os.path.exists(f"{root}/Annotations_train.json"):
            synth_data(root)
        return f"{root}/Annotations_train.json", f"{root}/train_images", f"{root}/test_images"
    ann = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT_ROOT) for f in fs if f.endswith(".json") and "nnotation" in f)
    return (ann, next(r for r, _, _ in os.walk(INPUT_ROOT) if os.path.basename(r) == "train_images"),
            next(r for r, _, _ in os.walk(INPUT_ROOT) if os.path.basename(r) == "test_images"))


def soft_target(passes, h, w):
    """Mean over the annotator passes of each pass's union mask, as uint8 (0..255)."""
    acc = np.zeros((h, w), np.float32)
    for anns in passes.values():
        m = np.zeros((h, w), np.uint8)
        for a in anns:
            if a.get("segmentation"):
                r = mu.frPyObjects(a["segmentation"], h, w)
                m |= mu.decode(mu.merge(r) if isinstance(r, list) else r)
        acc += m
    return np.round(acc / len(passes) * 255).astype(np.uint8)


def disk_stats(img):
    v = img[DISK].astype(np.float32) / 255.0
    return float(v.mean()), max(float(v.std()), 1e-3)


class CropDS(Dataset):
    """Sample i is a pure function of (SEED, FOLD, i): image, crop position and augmentation."""

    def __init__(self, names, cache):
        self.names, self.cache = names, cache

    def __len__(self):
        return STEPS * BATCH

    def __getitem__(self, i):
        rng = np.random.default_rng([SEED, FOLD, i])
        img, tgt, (mean, std), pos = self.cache[self.names[rng.integers(len(self.names))]]
        if len(pos) and rng.random() < POS_FRAC:
            cy, cx = pos[rng.integers(len(pos))]
            y0 = int(np.clip(cy - CROP // 2 + rng.integers(-CROP // 4, CROP // 4 + 1), 0, H - CROP))
            x0 = int(np.clip(cx - CROP // 2 + rng.integers(-CROP // 4, CROP // 4 + 1), 0, W - CROP))
        else:
            y0, x0 = (int(v) for v in rng.integers(0, H - CROP + 1, 2))
        x = img[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255.0
        y = tgt[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255.0
        x = np.power(np.clip(x, 0, 1), rng.uniform(0.8, 1.25))                  # gamma on the raw intensities
        x = (x - mean) / std * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)   # contrast / brightness jitter
        k = int(rng.integers(4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(y))[None]


class FromSampler(Sampler):
    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self):
        return self.stop - self.start


def loss_fn(logit, y):
    p = torch.sigmoid(logit.float())
    dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logit.float(), y) + dice.mean()


def lr_factor(step):
    if step < WARMUP:
        return (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def fresh():
    torch.manual_seed(SEED + FOLD)
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None if PRETRAINED == "none" else PRETRAINED,
                     in_channels=1, classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(enabled=AMP)
    return model, opt, sched, scaler


@torch.inference_mode()
def predict(model, img_u8, tta=TTA):
    """Full-image probability, mean over the flip/rotation views, as uint8 (p x 255)."""
    mean, std = disk_stats(img_u8)
    z = ((torch.from_numpy(img_u8.copy()).to(DEV).float() / 255.0 - mean) / std)[None, None]
    acc = torch.zeros(img_u8.shape, device=DEV)
    for k, flip in D4[:tta]:
        x = torch.flip(z, dims=[3]) if flip else z
        x = torch.rot90(x, k, dims=[2, 3]) if k else x
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
            p = torch.sigmoid(model(x).float())
        p = torch.rot90(p, -k, dims=[2, 3]) if k else p
        acc += (torch.flip(p, dims=[3]) if flip else p)[0, 0]
    return (acc / tta * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def main():
    log = Log()
    paths = {"final": os.path.join(OUT, f"{TAG}.pt"), "ckpt": os.path.join(OUT, f"effv2s_ckpt_fold{FOLD}.pt")}
    log.note(f"device {torch.cuda.get_device_name(0) if AMP else 'cpu'} torch {torch.__version__} smp {smp.__version__} "
             f"steps {STEPS} batch {BATCH} budget {BUDGET_S / 3600:.2f} h")
    assert AMP or SYNTH, "no GPU -- refusing to train on CPU"
    prev = glob.glob(f"{INPUT_ROOT}/**/{TAG}.json", recursive=True) if not SYNTH else []
    if prev and json.load(open(prev[0])).get("status") == "done":
        for p in glob.glob(os.path.join(os.path.dirname(prev[0]), f"effv2s_*fold{FOLD}*")):
            shutil.copy(p, OUT)
        print("finished in an attached source -- skipped", flush=True)
        return

    ann, train_dir, test_dir = locate()
    per_file, _ = fm.load_annotations(ann, train_dir)
    names = sorted(per_file)
    n_hold = 2 if SYNTH else N_HOLD
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout, pool = [names[i] for i in idx[:n_hold]], [names[i] for i in idx[n_hold:]]
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), N_FOLDS)
    val = [pool[i] for i in folds[FOLD - 1]]
    train = [pool[i] for i in np.concatenate([folds[j] for j in range(N_FOLDS) if j != FOLD - 1])]
    # ---- FAIL FAST 1: the split is the repository's
    if not SYNTH:
        assert len(names) == 707 and holdout == fm.holdout_names(names), "hold-out mismatch"
    assert not (set(train) & set(val)) and not (set(pool) & set(holdout))
    log.d["split"] = {"train": len(train), "val": len(val), "holdout": len(holdout), "val_first": val[:3]}
    log.note(f"split OK: train {len(train)}, validation {len(val)}, hold-out {len(holdout)}")

    cache = {}
    for fn in train:
        img = cv2.imread(os.path.join(train_dir, fn), cv2.IMREAD_GRAYSCALE)
        assert img is not None and img.shape == (H, W), (fn, None if img is None else img.shape)
        tgt = soft_target(per_file[fn], H, W)
        ys, xs = np.nonzero(tgt[::8, ::8])
        cache[fn] = (img, tgt, disk_stats(img), np.stack([ys * 8, xs * 8], 1).astype(np.int32))
    log.note(f"cached {len(cache)} training images; mean target coverage "
             f"{np.mean([c[1].mean() / 255 for c in cache.values()]):.4f}")

    ds = CropDS(train, cache)
    x0, y0 = ds[0]
    assert x0.shape == (1, CROP, CROP) and 0 <= float(y0.min()) and float(y0.max()) <= 1 and torch.equal(ds[0][0], x0)
    model, opt, sched, scaler = fresh()

    def step(x, y):
        """One optimiser step on a batch of BATCH samples as BATCH / MICRO accumulated micro-batches."""
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for xs_, ys_ in zip(x.split(MICRO), y.split(MICRO)):
            xs_, ys_ = xs_.to(DEV, non_blocking=True), ys_.to(DEV, non_blocking=True)
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
                out = model(xs_)
            loss = loss_fn(out, ys_) * (len(xs_) / len(x))
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.step(opt)
        scaler.update()
        sched.step()
        return tot

    def save_ckpt(step_done, hist, path=None):
        path = path or paths["ckpt"]
        torch.save({"step": step_done, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "hist": hist, "fold": FOLD, "seed": SEED, "encoder": ENCODER,
                    "steps": STEPS, "batch": BATCH}, path + ".tmp")
        os.replace(path + ".tmp", path)

    # ---- FAIL FAST 2: two real steps, checkpoint round trip, one full-image forward pass
    model.train()
    t0 = time.time()
    for j in range(2):
        xb, yb = zip(*(ds[j * BATCH + q] for q in range(BATCH)))
        l0 = step(torch.stack(xb), torch.stack(yb))
        assert math.isfinite(l0), l0
    s_step = (time.time() - t0) / 2
    ff = paths["ckpt"] + ".failfast"                    # never the real checkpoint: a rerun must still find it
    save_ckpt(0, [], ff)
    rt = torch.load(ff, map_location="cpu", weights_only=False)
    assert rt["fold"] == FOLD and set(rt["model"]) == set(model.state_dict())
    os.remove(ff)
    model.eval()
    t0 = time.time()
    pm = predict(model, cache[train[0]][0], tta=1)
    assert pm.shape == (H, W) and pm.dtype == np.uint8
    log.note(f"fail-fast OK: 2 steps ({s_step:.2f} s/step without loader, loss {l0:.3f}), checkpoint round trip, "
             f"full-image forward {time.time() - t0:.1f} s")
    del model, opt, sched, scaler
    model, opt, sched, scaler = fresh()                 # restart cleanly after the smoke steps

    start, hist = 0, []
    prev_ck = [paths["ckpt"]] if os.path.exists(paths["ckpt"]) else \
        ([] if SYNTH and os.environ.get("NO_RESUME") else glob.glob(f"{INPUT_ROOT}/**/effv2s_ckpt_fold{FOLD}.pt", recursive=True))
    if prev_ck:
        c = torch.load(prev_ck[0], map_location="cpu", weights_only=False)
        assert (c["fold"], c["seed"], c["encoder"], c["steps"], c["batch"]) == (FOLD, SEED, ENCODER, STEPS, BATCH), "foreign checkpoint"
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"]); start, hist = c["step"], c["hist"]
        log.note(f"RESUMED at step {start}")

    dl = DataLoader(ds, batch_size=BATCH, sampler=FromSampler(start * BATCH, STEPS * BATCH), num_workers=NUM_WORKERS,
                    pin_memory=AMP, drop_last=True, persistent_workers=NUM_WORKERS > 0)
    model.train()
    t_train, s, stopped, losses = time.time(), start, False, []
    for x, y in dl:
        loss = step(x, y)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {s + 1}")
        losses.append(loss)
        s += 1
        if s % 100 == 0:
            hist.append([s, round(float(np.mean(losses)), 4)])
            losses = []
        if s == 1000 and not SYNTH:
            first, late = hist[0][1], float(np.mean([h[1] for h in hist if 800 < h[0] <= 1000]))
            if not late < 0.9 * first:
                raise RuntimeError(f"NOT LEARNING: loss {first:.4f} (steps 1-100) -> {late:.4f} (steps 801-1000)")
        if s % CKPT_EVERY == 0 and s < STEPS:
            save_ckpt(s, hist)
            per = (time.time() - t_train) / (s - start)
            log.d["loss_history"] = hist
            log.note(f"step {s}/{STEPS} loss {hist[-1][1] if hist else float('nan'):.4f} lr {sched.get_last_lr()[0]:.2e} "
                     f"{per:.2f} s/step, checkpoint saved, ETA {(STEPS - s) * per / 60:.0f} min")
            if time.time() - T_START + CKPT_EVERY * per + INFER_RESERVE_S > BUDGET_S:
                stopped = True
                break
    del dl
    log.d["loss_history"] = hist
    log.d["train"] = {"steps_done": s, "train_seconds": round(time.time() - t_train)}
    if stopped:
        log.d["status"] = "stopped"
        log.note(f"TIME BUDGET: stopped at step {s}; attach this output as a kernel source and rerun to continue")
        return
    assert s == STEPS, (s, STEPS)
    torch.save(model.state_dict(), paths["final"])
    if os.path.exists(paths["ckpt"]):
        os.remove(paths["ckpt"])
    log.note("training finished, final weights saved")

    # ---- inference: out-of-fold (this fold's validation images), hold-out and test maps
    model.eval()
    del cache
    tnames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(test_dir, "*.jpeg")))
    for tag, d, ns in (("oof", train_dir, val), ("holdout", train_dir, holdout), ("test", test_dir, tnames)):
        maps, t0 = {}, time.time()
        for fn in (ns[:1] if SYNTH else ns):
            maps[fn[:-5]] = predict(model, cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE), tta=1 if SYNTH else TTA)
        np.savez_compressed(os.path.join(OUT, f"effv2s_maps_fold{FOLD}_{tag}.npz"), **maps)
        log.note(f"{tag}: {len(maps)} maps in {time.time() - t0:.0f} s")
    log.d["status"] = "done"
    log.note("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                              # STOP ON ERROR: log and end the kernel
        p = os.path.join(OUT, f"{TAG}.json")
        d = json.load(open(p)) if os.path.exists(p) else {"fold": FOLD, "events": []}
        d["status"], d["error"] = "error", f"{type(e).__name__}: {e}"
        json.dump(d, open(p, "w"), indent=1)
        print(f"fold{FOLD} ERROR {type(e).__name__}: {e} -- stopping the kernel", flush=True)
        raise
