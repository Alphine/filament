"""
Solar Filament Segmentation 2026 -- stage-A with per-epoch patch RE-SAMPLING (Kaggle GPU).

Finding (verified on the production code): tr.PatchDS draws each patch from a generator seeded by
the sample index only and the DataLoader does not shuffle, so all 30 epochs of the production
training see the SAME 1,760 patches (52,800 patch draws, 1,760 distinct). This experiment changes
exactly that and nothing else:

    epoch e uses the global sample indices e*1760 ... (e+1)*1760 - 1

Epoch 0 is therefore identical to production; later epochs draw new patches (same crop policy,
same augmentation, same number of steps). Implemented with a sampler in the main process, so it
works with persistent DataLoader workers. Everything else follows the checkpoint-selection runs
(training/filament_ckpt_selection.py): same seed and initialisation, same splits and folds, union
target, Tversky+BCE, AdamW + OneCycle 1e-4, 30 epochs x 220 steps x batch 8, EMA of the weights
(decay 0.999, every step).

Comparison: the baseline is the EMA (and last-epoch) weights of the checkpoint-selection runs,
scored on the same out-of-fold images (each fold's validation images minus the first 20), same
protocol (single model, 4-flip TTA, production thresholds, released crop refiner).
Decision rule fixed beforehand (analysis/scripts/analyze_resample.py): resampled-EMA replaces
baseline-EMA only if the paired gain pooled over the five folds (with refiner) is >= +0.005 with a
bootstrap CI95 above 0; then an ensemble-level hold-out check against the EMA-baseline ensemble.

Compute safety -- STOP ON ERROR (any exception is logged and re-raised: the kernel ends):
  * FAIL FAST before training: the evaluation path must reproduce the stored out-of-fold counts of
    the released fold model on one image; the sampler must equal production at epoch 0, differ at
    epoch 1 and be deterministic; 2 real training steps; checkpoint save/load round trip.
  * non-finite loss stops the run; per-epoch checkpoint (model, optimiser, scheduler, scaler, EMA)
    resumable from INPUT_ROOT; a finished fold found in INPUT_ROOT is skipped; time budget.
Outputs per fold: resample_fold<F>.json, resample_fold<F>_ema.pt, resample_fold<F>_last.pt.
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
TRAIN_BUDGET_S = float(os.environ.get("TRAIN_BUDGET_S", str(3.5 * 3600)))
INPUT_ROOT = os.environ.get("INPUT_ROOT", "/kaggle/input")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))   # production value 2; lowered only for the low-memory local test
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
SEED = 20260927          # same as the checkpoint-selection runs (baseline)
EMA_DECAY = 0.999
RULES = ("ema", "last")

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

import cv2
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, ".")
import filament_unet_ensemble as tr
import filament_ckpt_selection as sel      # evaluate / plain_state / pooled_pq of the baseline runs
import filament_pipeline as fp
import filament_metric as fm


class EpochSampler(Sampler):
    """Yields the global sample indices epoch*n .. (epoch+1)*n - 1 (epoch 0 == production order)."""

    def __init__(self, n):
        self.n = n
        self.epoch = 0

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))

    def __len__(self):
        return self.n


class Log:
    def __init__(self, fold):
        self.path = os.path.join(OUT, f"resample_fold{fold}.json")
        self.d = {"fold": fold, "seed": SEED, "ema_decay": EMA_DECAY, "events": []}

    def note(self, msg):
        print(f"[{time.time() - T_START:7.0f}s] fold{self.d['fold']} {msg}", flush=True)
        self.d["events"].append([round(time.time() - T_START), msg])
        self.save()

    def save(self):
        json.dump(self.d, open(self.path, "w"), indent=1)


def fresh(total_steps):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = tr.UNet().to(tr.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.LR, total_steps=total_steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(enabled=True)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)
    return model, opt, sched, scaler, ema


def run_fold(fold, S):
    log = Log(fold)
    paths = {r: os.path.join(OUT, f"resample_fold{fold}_{r}.pt") for r in RULES}
    ckpt = os.path.join(OUT, f"resample_ckpt_fold{fold}.pt")
    prev_json = glob.glob(f"{INPUT_ROOT}/**/resample_fold{fold}.json", recursive=True)
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

    # ---- FAIL FAST 1: the evaluation path reproduces the stored out-of-fold counts
    released = next(iter(glob.glob(f"{INPUT_ROOT}/**/unet_fold{fold}.pt", recursive=True)))
    got = sel.evaluate(released, eval_names[:1], S["per_file_m"], S["refiner"], log, "fail-fast")[0]["with_refiner"]
    want = S["oof_ref"][eval_names[0]]
    assert got[1:] == want[1:], f"FAIL FAST: eval path {got[1:]} != stored OOF {want[1:]} on {eval_names[0]}"
    log.note(f"fail-fast eval OK on {eval_names[0]}: {got[1:]}")

    total = tr.EPOCHS * tr.STEPS_PER_EPOCH
    model, opt, sched, scaler, ema = fresh(total)
    ds = tr.PatchDS(ft, tr.STEPS_PER_EPOCH, S["cache"])
    ds.build_positions()
    n = len(ds)
    sampler = EpochSampler(n)
    dl = DataLoader(ds, batch_size=tr.BATCH, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                    persistent_workers=True)

    # ---- FAIL FAST 2: sampler semantics
    assert list(EpochSampler(n)) == list(range(n)), "epoch-0 order differs from production"
    e0 = ds[0][0]
    assert torch.equal(ds[0][0], e0), "patch is not deterministic for a fixed index"
    assert not torch.equal(ds[n][0], e0), "epoch 1 draws the same patch as epoch 0"
    log.note(f"sampler OK: epoch 0 == production order, epoch 1 draws new patches ({n} per epoch)")

    def step(x, y):
        x, y = x.to(tr.DEV, non_blocking=True), y.to(tr.DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = tr.tversky_bce(model(x), y)
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

    # ---- FAIL FAST 3: two real steps and a checkpoint round trip
    model.train()
    it = iter(dl)
    t0 = time.time()
    for _ in range(2):
        step(*next(it))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del it
    save_ckpt(0)
    rt = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert rt["fold"] == fold and set(rt["model"]) == set(model.state_dict()) and "n_averaged" in rt["ema"]
    log.note(f"2 steps OK ({(time.time() - t0) / 2:.2f}s/step); checkpoint round trip OK")
    model, opt, sched, scaler, ema = fresh(total)     # restart cleanly after the smoke steps

    start = 0
    prev = glob.glob(f"{INPUT_ROOT}/**/resample_ckpt_fold{fold}.pt", recursive=True)
    if prev:
        c = torch.load(prev[0], map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"]); ema.load_state_dict(c["ema"]); start = c["epoch"]
        log.note(f"RESUMED at epoch {start}")
    t_train = time.time()
    stopped = False
    ep = start - 1                                     # defined even if a resumed run has no epochs left
    for ep in range(start, tr.EPOCHS):
        sampler.epoch = ep
        model.train()
        losses, first = [], None
        for x, y in dl:
            if first is None:
                first = float(x.abs().mean())          # fingerprint: differs between epochs iff patches differ
            losses.append(step(x, y))
        loss = float(np.mean(losses))
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {ep + 1}")
        save_ckpt(ep + 1)
        log.note(f"ep{ep + 1:2d} loss={loss:.4f} first-batch fingerprint {first:.5f}")
        per_ep = (time.time() - t_train) / (ep + 1 - start)
        if time.time() - T_START + per_ep > TRAIN_BUDGET_S and ep + 1 < tr.EPOCHS:
            stopped = True
            log.note(f"TIME BUDGET: stopping after epoch {ep + 1}")
            break
    torch.save(sel.plain_state(ema), paths["ema"])
    torch.save(sel.plain_state(model), paths["last"])
    log.d["train"] = {"epochs_done": ep + 1, "stopped_early": stopped, "train_seconds": round(time.time() - t_train)}
    del dl, ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.d["eval"] = {r: sel.evaluate(p, eval_names, S["per_file_m"], S["refiner"], log, r) for r, p in paths.items()}
    log.d["eval_pq"] = {r: {k: round(sel.pooled_pq(rows, k), 4) for k in ("with_refiner", "without_refiner")}
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
    cache = {}
    for fn in pool:
        img = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        mp = os.path.join(tr.MASK_DIR, fn.replace(".jpeg", ".png").replace(".jpg", ".png"))
        cache[fn] = (img, (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))
    oof_ref = {}
    for p in glob.glob(f"{INPUT_ROOT}/**/oof_fold*.json", recursive=True):
        oof_ref.update({r["name"]: r["with_refiner"] for r in json.load(open(p))["results"]})
    refiner = fp.load_refiner(next(iter(glob.glob(f"{INPUT_ROOT}/**/refiner_best.pt", recursive=True))),
                              fp.default_device())
    S = dict(per_file_m=per_file_m, pool=pool, folds=folds, cache=cache, refiner=refiner, oof_ref=oof_ref)
    for fold in FOLDS:
        try:
            run_fold(fold, S)
        except Exception as e:                    # STOP ON ERROR: log and end the kernel
            p = os.path.join(OUT, f"resample_fold{fold}.json")
            d = json.load(open(p)) if os.path.exists(p) else {"fold": fold, "events": []}
            d["error"] = f"{type(e).__name__}: {e}"
            json.dump(d, open(p, "w"), indent=1)
            print(f"fold{fold} ERROR {type(e).__name__}: {e} -- stopping the kernel", flush=True)
            raise


if __name__ == "__main__":
    main()
