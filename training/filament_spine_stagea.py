"""
Solar Filament Segmentation 2026 -- stage-A with SPINE multi-task supervision (Kaggle GPU).

Why: the refiner is saturated (refiner generation 2 gave nothing), so the remaining room is in
stage-A detection (FP and FN). Every training annotation carries a manually drawn spine (centre
line; 100% of 8,199 annotations, flat [x0, y0, x1, y1, ...], x-y order verified: 99.6% of the
rasterised spine pixels fall inside the filament mask). The organizers allow it as supervision.
The spine was never used by any of our models.

Model: the production stage-A (training/filament_unet_ensemble.py, imported) with a 2-channel
head: channel 0 = filament mask (union of passes, as in production), channel 1 = spine (union of
passes, 5-px polylines). Loss = tversky_bce(mask) + 0.5 * tversky_bce(spine). Everything else as
production; same seed and data order as the checkpoint-selection runs, whose EMA weights are the
baseline. Weights are kept as an EMA of every step (decay 0.999) -- the rule that removed the
checkpoint-selection noise. Saved weights keep only head channel 0, so they load unchanged into
the released inference pipeline (filament_pipeline.load_stage_a).

Evaluation per fold (identical to training/filament_ckpt_selection.py): the model alone, 4-flip
TTA, production thresholds, released refiner, on the fold's validation images minus its first 20.
Decision rule fixed beforehand: spine-EMA replaces baseline-EMA only if the paired gain pooled over
the five folds is >= +0.005 with a bootstrap CI95 above 0 (then a hold-out gate follows).

Compute safety -- STOP ON ERROR (any exception ends the kernel after logging):
  * FAIL FAST before training: spine rasterisation sanity (>= 90% of thin spine pixels inside
    the mask); the evaluation path must reproduce the stored out-of-fold counts of the released
    fold model on one image; the sliced 1-channel state must load strictly into the released
    ResUNet; 2 real training steps; checkpoint save/load round trip (model, optimiser, scheduler,
    scaler, EMA).
  * per-epoch checkpoint, resumable from /kaggle/input; finished folds in /kaggle/input skipped;
    per-kernel time budget stops training early but still evaluates.
Outputs per fold: spine_fold<F>.json, spine_fold<F>_ema.pt, spine_fold<F>_last.pt.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

T_START = time.time()
FOLDS = [int(f) for f in os.environ.get("FOLDS", "1").split(",")]
TRAIN_BUDGET_S = float(os.environ.get("TRAIN_BUDGET_S", str(10.0 * 3600)))
SEED = 20260927          # same as the checkpoint-selection runs (baseline)
EMA_DECAY = 0.999
LAMBDA_SPINE = 0.5
SPINE_T = 5              # spine line thickness (px) in the training target

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
import filament_unet_ensemble as tr
import filament_pipeline as fp
import filament_metric as fm

OUT = "/kaggle/working"


class Log:
    def __init__(self, fold):
        self.path = os.path.join(OUT, f"spine_fold{fold}.json")
        self.d = {"fold": fold, "seed": SEED, "lambda_spine": LAMBDA_SPINE, "spine_t": SPINE_T, "events": []}

    def note(self, msg):
        print(f"[{time.time() - T_START:7.0f}s] fold{self.d['fold']} {msg}", flush=True)
        self.d["events"].append([round(time.time() - T_START), msg])
        json.dump(self.d, open(self.path, "w"), indent=1)


def spine_raster(anns, h, w, thickness):
    canvas = np.zeros((h, w), np.uint8)
    for a in anns:
        pts = np.asarray(a.get("spine") or [], float).reshape(-1, 2)
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 1, thickness)
    return canvas


class SpinePatchDS(tr.PatchDS):
    """Production patch sampler; target = mask (bit 0) | spine (bit 1). Patch positions are
    sampled from the mask bit only, exactly as in production."""

    def build_positions(self):
        self.pos = {}
        for fn in self.names:
            _, m = self.cache[fn]
            ys, xs = np.nonzero((m & 1)[::16, ::16])
            self.pos[fn] = np.stack([ys * 16, xs * 16], 1) if len(ys) else np.empty((0, 2), int)


def two_head_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = tr.UNet()
    model.head = nn.Conv2d(32, 2, 1)
    return model.to(tr.DEV)


def sliced_state(module):
    """1-channel (mask) state_dict that loads into the released ResUNet."""
    m = module.module if isinstance(module, AveragedModel) else module
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    sd["head.weight"] = sd["head.weight"][:1].clone()
    sd["head.bias"] = sd["head.bias"][:1].clone()
    return sd


def spine_loss(logits, y):
    ym = torch.remainder(y, 2.0)
    ys = (y >= 2).float()
    return tr.tversky_bce(logits[:, :1].float(), ym) + LAMBDA_SPINE * tr.tversky_bce(logits[:, 1:2].float(), ys)


def evaluate(weight_path, names, S, log, tag):
    device = fp.default_device()
    model = fp.load_stage_a([weight_path], device)
    rows = []
    for k, name in enumerate(names, 1):
        g = fp.read_gray(os.path.join(tr.TRAIN_IMG, name))
        h, w = g.shape
        cand = fp.extract_candidates(fp.predict_ensemble(model, g, device))
        lab = fp.refine_candidates(g, cand, S["refiner"], device)
        a = fm.pq_counts(fp.label_map_to_rles(lab), S["per_file_m"][name], h, w)
        b = fm.pq_counts(fp.label_map_to_rles(cand), S["per_file_m"][name], h, w)
        rows.append({"name": name, "n_pass": len(S["per_file_m"][name]),
                     "with_refiner": [float(a[0]), *map(int, a[1:])],
                     "without_refiner": [float(b[0]), *map(int, b[1:])]})
        if k % 40 == 0:
            log.note(f"eval {tag}: {k}/{len(names)}")
    return rows


def pooled_pq(rows, key):
    t = np.sum([r[key] for r in rows], 0)
    return float(fm.pq(*t))


def run_fold(fold, S):
    log = Log(fold)
    paths = {r: os.path.join(OUT, f"spine_fold{fold}_{r}.pt") for r in ("ema", "last")}
    ckpt = os.path.join(OUT, f"spine_ckpt_fold{fold}.pt")
    prev_json = glob.glob(f"/kaggle/input/**/spine_fold{fold}.json", recursive=True)
    if prev_json and "eval_pq" in json.load(open(prev_json[0])):
        base = os.path.dirname(prev_json[0])
        for r, p in paths.items():
            shutil.copy(os.path.join(base, os.path.basename(p)), p)
        shutil.copy(prev_json[0], log.path)
        print(f"fold{fold}: finished in an attached source -- skipped", flush=True)
        return
    pool, folds = S["pool"], S["folds"]
    fv = [pool[i] for i in folds[fold - 1]]
    ft = [pool[i] for i in np.concatenate([folds[j] for j in range(tr.N_FOLDS) if j != fold - 1])]
    eval_names = fv[tr.VAL_QUICK:]
    log.note(f"train {len(ft)} images, evaluation {len(eval_names)} images")

    # ---- FAIL FAST 1: evaluation path reproduces the stored out-of-fold counts
    released = next(p for p in glob.glob(f"/kaggle/input/**/unet_fold{fold}.pt", recursive=True))
    got = evaluate(released, eval_names[:1], S, log, "fail-fast")[0]["with_refiner"]
    want = S["oof_ref"][eval_names[0]]
    assert got[1:] == want[1:], f"FAIL FAST: eval path {got[1:]} != stored OOF {want[1:]} on {eval_names[0]}"
    log.note(f"fail-fast eval OK on {eval_names[0]}: {got[1:]}")

    total = tr.EPOCHS * tr.STEPS_PER_EPOCH

    def fresh():
        model = two_head_model()
        opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.LR, total_steps=total, pct_start=0.25)
        scaler = torch.amp.GradScaler(enabled=True)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)
        return model, opt, sched, scaler, ema

    model, opt, sched, scaler, ema = fresh()
    ds = SpinePatchDS(ft, tr.STEPS_PER_EPOCH, S["cache"])
    ds.build_positions()
    dl = DataLoader(ds, batch_size=tr.BATCH, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)

    def step(x, y):
        x, y = x.to(tr.DEV, non_blocking=True), y.to(tr.DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            logits = model(x)
        loss = spine_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update_parameters(model)
        return loss.item()

    def save_ckpt(ep_done):
        torch.save({"epoch": ep_done, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "ema": ema.state_dict(), "fold": fold, "seed": SEED}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    # ---- FAIL FAST 2: sliced state loads into the released architecture
    tmp = os.path.join(OUT, f"_slice_test_fold{fold}.pt")
    torch.save(sliced_state(model), tmp)
    fp.load_stage_a([tmp], fp.default_device())
    os.remove(tmp)
    # ---- FAIL FAST 3: two real steps and a checkpoint round trip
    model.train()
    it = iter(dl)
    t0 = time.time()
    for _ in range(2):
        step(*next(it))
    torch.cuda.synchronize()
    del it
    save_ckpt(0)
    rt = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert rt["fold"] == fold and set(rt["model"]) == set(model.state_dict()) and "n_averaged" in rt["ema"]
    log.note(f"sliced state loads OK; 2 steps OK ({(time.time() - t0) / 2:.2f}s/step); checkpoint round trip OK")
    model, opt, sched, scaler, ema = fresh()

    start = 0
    prev = glob.glob(f"/kaggle/input/**/spine_ckpt_fold{fold}.pt", recursive=True)
    if prev:
        c = torch.load(prev[0], map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"]); ema.load_state_dict(c["ema"]); start = c["epoch"]
        log.note(f"RESUMED at epoch {start}")
    t_train = time.time()
    stopped = False
    ep = start - 1                      # defined even if a resumed run has no epochs left
    for ep in range(start, tr.EPOCHS):
        model.train()
        losses = [step(x, y) for x, y in dl]
        loss = float(np.mean(losses))
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {ep + 1}")
        save_ckpt(ep + 1)
        log.note(f"ep{ep + 1:2d} loss={loss:.4f}")
        per_ep = (time.time() - t_train) / (ep + 1 - start)
        if time.time() - T_START + per_ep > TRAIN_BUDGET_S and ep + 1 < tr.EPOCHS:
            stopped = True
            log.note(f"TIME BUDGET: stopping after epoch {ep + 1}")
            break
    torch.save(sliced_state(ema), paths["ema"])
    torch.save(sliced_state(model), paths["last"])
    log.d["train"] = {"epochs_done": ep + 1, "stopped_early": stopped, "train_seconds": round(time.time() - t_train)}
    del dl, ds
    torch.cuda.empty_cache()
    log.d["eval"] = {r: evaluate(p, eval_names, S, log, r) for r, p in paths.items()}
    log.d["eval_pq"] = {r: {k: round(pooled_pq(rows, k), 4) for k in ("with_refiner", "without_refiner")}
                        for r, rows in log.d["eval"].items()}
    log.note(f"DONE {log.d['eval_pq']}")
    if not stopped and os.path.exists(ckpt):
        os.remove(ckpt)


def main():
    print(f"folds={FOLDS} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)
    assert torch.cuda.is_available(), "no GPU -- refusing to train on CPU"
    names, per_file, dims = tr.build_masks()
    per_file_m, _ = fm.load_annotations(tr.ANN, tr.TRAIN_IMG)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout, pool = [names[i] for i in idx[:tr.N_VAL]], [names[i] for i in idx[tr.N_VAL:]]
    assert holdout == fm.holdout_names(list(per_file_m)), "hold-out mismatch"
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), tr.N_FOLDS)
    cache, inside = {}, []
    for k, fn in enumerate(pool):
        h, w = dims[fn]
        img = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        mp = os.path.join(tr.MASK_DIR, fn.replace(".jpeg", ".png").replace(".jpg", ".png"))
        msk = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        sp = spine_raster(per_file[fn], h, w, SPINE_T)
        cache[fn] = (img, (msk | (sp << 1)).astype(np.uint8))
        if k < 60:                      # sanity on the thin (1-px) spine
            thin = spine_raster(per_file[fn], h, w, 1).astype(bool)
            if thin.any():
                inside.append(float((thin & msk.astype(bool)).sum() / thin.sum()))
    share = float(np.median(inside))
    print(f"spine sanity: median share of 1-px spine pixels inside the union mask {share:.3f} (60 images)", flush=True)
    assert share >= 0.90, f"FAIL FAST: spine raster looks wrong (inside share {share:.3f})"
    oof_ref = {}
    for p in glob.glob("/kaggle/input/**/oof_fold*.json", recursive=True):
        oof_ref.update({r["name"]: r["with_refiner"] for r in json.load(open(p))["results"]})
    refiner = fp.load_refiner(next(p for p in glob.glob("/kaggle/input/**/refiner_best.pt", recursive=True)),
                              fp.default_device())
    S = dict(per_file=per_file, dims=dims, per_file_m=per_file_m, pool=pool, folds=folds, cache=cache,
             refiner=refiner, oof_ref=oof_ref)
    for fold in FOLDS:
        try:
            run_fold(fold, S)
        except Exception as e:           # STOP ON ERROR: log and end the kernel
            p = os.path.join(OUT, f"spine_fold{fold}.json")
            d = json.load(open(p)) if os.path.exists(p) else {"fold": fold, "events": []}
            d["error"] = f"{type(e).__name__}: {e}"
            json.dump(d, open(p, "w"), indent=1)
            print(f"fold{fold} ERROR {type(e).__name__}: {e} -- stopping the kernel", flush=True)
            raise


if __name__ == "__main__":
    main()
