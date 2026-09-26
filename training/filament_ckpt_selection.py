"""
Solar Filament Segmentation 2026 -- checkpoint-SELECTION study (Kaggle GPU).

Motivation: in the 5-fold consensus run the fold-to-fold spread (about +-0.015 PQ) was larger
than the effect being tested, and the two worst folds had their checkpoint picked at epoch 10
by the production rule (best of 20 fold-val images, checked every 5 epochs). A noisy selection
rule adds variance to every stage-A experiment and to the released ensemble.

One production training run per fold (union target, production recipe imported from
training/filament_unet_ensemble.py), and four ways to take weights from the SAME trajectory:
  prod  best fold-val PQ on the first 20 fold-val images, checked every 5 epochs (production)
  last  the weights after the final epoch (30)
  swa   equal-weight average of the end-of-epoch weights of epochs 21-30 (BN buffers averaged)
  ema   exponential moving average of the weights, updated every step, decay 0.999
        (BN buffers averaged)
Evaluation: each rule's model alone (4-flip TTA, production thresholds, released crop refiner)
on the fold's validation images EXCLUDING the 20 images the prod rule selects on.
Decision rule, fixed before the run: a rule replaces prod only if its paired gain pooled over
the five folds (with refiner) is >= +0.005 with a bootstrap CI95 that excludes 0.

Compute safety: fail-fast (one-image evaluation, three timed training steps, checkpoint
round trip) before training; per-epoch checkpoint of model/optimiser/scheduler/scaler and the
SWA/EMA/prod states, resumable from /kaggle/input; a finished fold found in /kaggle/input is
skipped; a time budget stops training early and still evaluates.
Outputs per fold: sel_fold<F>_<rule>.pt (state_dicts) and sel_fold<F>.json.
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
TRAIN_BUDGET_S = float(os.environ.get("TRAIN_BUDGET_S", str(10.5 * 3600)))
SEED = 20260927
RULES = ("prod", "last", "swa", "ema")
SWA_FROM = 21          # average end-of-epoch weights of epochs 21..30
EMA_DECAY = 0.999

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

import cv2
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
import filament_unet_ensemble as tr
import filament_pipeline as fp
import filament_metric as fm

OUT = "/kaggle/working"


class Log:
    def __init__(self, fold):
        self.path = os.path.join(OUT, f"sel_fold{fold}.json")
        self.d = {"fold": fold, "seed": SEED, "rules": RULES, "events": []}

    def note(self, msg):
        print(f"[{time.time() - T_START:7.0f}s] fold{self.d['fold']} {msg}", flush=True)
        self.d["events"].append([round(time.time() - T_START), msg])
        json.dump(self.d, open(self.path, "w"), indent=1)


def plain_state(module):
    """state_dict of a model or of an AveragedModel's inner module, on CPU."""
    m = module.module if isinstance(module, AveragedModel) else module
    return {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}


def evaluate(weight_path, names, per_file_m, refiner, log, tag):
    device = fp.default_device()
    model = fp.load_stage_a([weight_path], device)
    rows = []
    for k, name in enumerate(names, 1):
        g = fp.read_gray(os.path.join(tr.TRAIN_IMG, name))
        h, w = g.shape
        prob = fp.predict_ensemble(model, g, device)
        cand = fp.extract_candidates(prob)
        lab = fp.refine_candidates(g, cand, refiner, device)
        a = fm.pq_counts(fp.label_map_to_rles(lab), per_file_m[name], h, w)
        b = fm.pq_counts(fp.label_map_to_rles(cand), per_file_m[name], h, w)
        rows.append({"name": name, "n_pass": len(per_file_m[name]),
                     "with_refiner": [float(a[0]), *map(int, a[1:])],
                     "without_refiner": [float(b[0]), *map(int, b[1:])]})
        if k % 40 == 0:
            log.note(f"eval {tag}: {k}/{len(names)}")
    return rows


def pooled_pq(rows, key):
    t = np.sum([r[key] for r in rows], 0)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return float(t[0] / d) if d else 0.0


def run_fold(fold, S):
    log = Log(fold)
    paths = {r: os.path.join(OUT, f"sel_fold{fold}_{r}.pt") for r in RULES}
    ckpt = os.path.join(OUT, f"sel_ckpt_fold{fold}.pt")
    prev_json = glob.glob(f"/kaggle/input/**/sel_fold{fold}.json", recursive=True)
    if prev_json and "eval_pq" in json.load(open(prev_json[0])):
        base = os.path.dirname(prev_json[0])
        for r in RULES:
            shutil.copy(os.path.join(base, f"sel_fold{fold}_{r}.pt"), paths[r])
        shutil.copy(prev_json[0], log.path)
        print(f"fold{fold}: already finished in an attached source -- copied, skipped", flush=True)
        return

    pool, folds = S["pool"], S["folds"]
    fv = [pool[i] for i in folds[fold - 1]]
    ft = [pool[i] for i in np.concatenate([folds[j] for j in range(tr.N_FOLDS) if j != fold - 1])]
    quick = fv[:tr.VAL_QUICK]
    eval_names = fv[tr.VAL_QUICK:]
    log.note(f"train {len(ft)}, fold-val {len(fv)} (prod selects on {len(quick)}, evaluation on {len(eval_names)})")

    released = next(p for p in glob.glob(f"/kaggle/input/**/unet_fold{fold}.pt", recursive=True))
    fast = evaluate(released, eval_names[:1], S["per_file_m"], S["refiner"], log, "fail-fast")
    log.note(f"fail-fast eval OK: {fast[0]['with_refiner']}")

    total = tr.EPOCHS * tr.STEPS_PER_EPOCH

    def fresh():
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = tr.UNet().to(tr.DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.LR, total_steps=total, pct_start=0.25)
        scaler = torch.amp.GradScaler(enabled=True)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)
        swa = AveragedModel(model, use_buffers=True)
        return model, opt, sched, scaler, ema, swa

    model, opt, sched, scaler, ema, swa = fresh()
    ds = tr.PatchDS(ft, tr.STEPS_PER_EPOCH, S["cache"])
    ds.build_positions()
    dl = DataLoader(ds, batch_size=tr.BATCH, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)

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

    def save_ckpt(ep_done, best):
        torch.save({"epoch": ep_done, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best,
                    "ema": ema.state_dict(), "swa": swa.state_dict(), "fold": fold, "seed": SEED}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    save_ckpt(0, (-1.0, -1))
    rt = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert rt["fold"] == fold and set(rt["model"]) == set(model.state_dict()) and "n_averaged" in rt["swa"]
    log.note("checkpoint round trip OK (model, optimiser, scheduler, scaler, EMA, SWA)")

    start_ep, best = 0, (-1.0, -1)
    prev = glob.glob(f"/kaggle/input/**/sel_ckpt_fold{fold}.pt", recursive=True)
    if prev:
        ck = torch.load(prev[0], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); ema.load_state_dict(ck["ema"]); swa.load_state_dict(ck["swa"])
        start_ep, best = ck["epoch"], tuple(ck["best"])[:2]
        pb = glob.glob(f"/kaggle/input/**/sel_fold{fold}_prod.pt", recursive=True)
        if pb:
            shutil.copy(pb[0], paths["prod"])
        log.note(f"RESUMED at epoch {start_ep} (prod best {best[0]:.4f} @ {best[1]})")
    else:
        model.train()
        t0 = time.time()
        it = iter(dl)
        for _ in range(3):
            step(*next(it))
        torch.cuda.synchronize()
        del it
        log.note(f"3 training steps OK ({(time.time() - t0) / 3:.2f}s/step incl. warm-up)")
        model, opt, sched, scaler, ema, swa = fresh()

    t_train = time.time()
    stopped_early = False
    for ep in range(start_ep, tr.EPOCHS):
        model.train()
        tot = 0.0
        for x, y in dl:
            tot += step(x, y)
        if ep + 1 >= SWA_FROM:
            swa.update_parameters(model)
        msg = f"ep{ep + 1:2d} loss={tot / len(dl):.4f}"
        if (ep + 1) % tr.VAL_EVERY == 0 or ep == tr.EPOCHS - 1:
            model.eval()
            s_sum = t_sum = fp_sum = fn_sum = 0.0
            for fn in quick:
                g = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
                lab = tr.instances(tr.predict_prob(model, g), 0.4, 400, min_mean_prob=0.7)
                s_, tp, fp_, fn_ = tr.pq_counts(lab, tr.val_gt(fn, S["per_file"], S["dims"]))
                s_sum, t_sum, fp_sum, fn_sum = s_sum + s_, t_sum + tp, fp_sum + fp_, fn_sum + fn_
            v = tr.pq(s_sum, t_sum, fp_sum, fn_sum)
            if v > best[0]:
                best = (v, ep + 1)
                torch.save(plain_state(model), paths["prod"])
            msg += f" foldval_PQ={v:.4f} prod_best={best[0]:.4f}@{best[1]}"
        log.note(msg)
        save_ckpt(ep + 1, best)
        per_ep = (time.time() - t_train) / (ep + 1 - start_ep)
        if time.time() - T_START + per_ep > TRAIN_BUDGET_S and ep + 1 < tr.EPOCHS:
            stopped_early = True
            log.note(f"TIME BUDGET: stopping after epoch {ep + 1}")
            break
    torch.save(plain_state(model), paths["last"])
    torch.save(plain_state(ema), paths["ema"])
    if int(swa.n_averaged) > 0:
        torch.save(plain_state(swa), paths["swa"])
    log.d["train"] = {"prod_best_foldval_pq": best[0], "prod_best_epoch": best[1], "stopped_early": stopped_early,
                      "swa_n_averaged": int(swa.n_averaged), "train_seconds": round(time.time() - t_train)}
    del dl, ds
    torch.cuda.empty_cache()

    log.d["eval"] = {}
    for r in RULES:
        if os.path.exists(paths[r]):
            log.d["eval"][r] = evaluate(paths[r], eval_names, S["per_file_m"], S["refiner"], log, r)
    log.d["eval_pq"] = {r: {k: round(pooled_pq(rows, k), 4) for k in ("with_refiner", "without_refiner")}
                        for r, rows in log.d["eval"].items()}
    log.note(f"DONE {log.d['eval_pq']}")
    if not stopped_early and os.path.exists(ckpt):
        os.remove(ckpt)


def main():
    print(f"folds={FOLDS} device={tr.DEV} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    assert torch.cuda.is_available(), "no GPU -- refusing to train on CPU"
    names, per_file, dims = tr.build_masks()      # production union masks
    per_file_m, _ = fm.load_annotations(tr.ANN, tr.TRAIN_IMG)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout = [names[i] for i in idx[:tr.N_VAL]]
    pool = [names[i] for i in idx[tr.N_VAL:]]
    assert holdout == fm.holdout_names(list(per_file_m)), "hold-out mismatch"
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), tr.N_FOLDS)
    cache = {}
    for fn in pool:
        img = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        mp = os.path.join(tr.MASK_DIR, fn.replace(".jpeg", ".png").replace(".jpg", ".png"))
        cache[fn] = (img, (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))
    refiner = fp.load_refiner(next(p for p in glob.glob("/kaggle/input/**/refiner_best.pt", recursive=True)),
                              fp.default_device())
    S = dict(per_file=per_file, dims=dims, per_file_m=per_file_m, pool=pool, folds=folds, cache=cache, refiner=refiner)
    failed = []
    for fold in FOLDS:
        try:
            run_fold(fold, S)
        except Exception as e:
            print(f"fold{fold} ERROR {type(e).__name__}: {e}", flush=True)
            p = os.path.join(OUT, f"sel_fold{fold}.json")
            d = json.load(open(p)) if os.path.exists(p) else {"fold": fold, "events": []}
            d["error"] = f"{type(e).__name__}: {e}"
            json.dump(d, open(p, "w"), indent=1)
            failed.append(fold)
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"folds failed: {failed}")


if __name__ == "__main__":
    main()
