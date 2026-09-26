"""
Solar Filament Segmentation 2026 -- consensus-target PILOT (Kaggle GPU), one arm per kernel.

Question: does training stage-A on a consensus target beat the pixel-union target?
  ARM=union      control: exactly the production target (pixel union of all passes)
  ARM=consensus  on multi-pass images keep only instances matched (IoU > 0.3) by an instance
                 of at least one OTHER pass; single-pass images unchanged. Generalises the
                 "inst_union_drop" constructor of analysis/scripts/filament_consensus_cpu.py,
                 which scored +0.0145 PQ over the union in an annotation-only simulation.

Everything else is the production recipe of training/filament_unet_ensemble.py (imported,
not copied): fold 1 of the same splits, ResNet34 U-Net, 30 epochs x 220 steps, batch 8,
Tversky+BCE, AdamW + OneCycle 1e-4, checkpoint chosen every 5 epochs on 20 fold-val images.
Both arms use the same torch seed, so their difference isolates the target (the data loader
is already deterministic); ARM=union against the released unet_fold1.pt measures run-to-run
noise.

Evaluation (after training, same protocol as analysis/scripts/filament_oof_cpu.py): the best
checkpoint alone with 4-flip TTA, production thresholds and the released crop refiner, scored
per image on fold 1's 132 validation (out-of-fold) images and on the 50-image hold-out.
Decision rule, fixed before the run: go to 5 folds only if consensus - union >= +0.005 on the
OOF images with a paired bootstrap CI that excludes 0.

Compute safety:
  * FAIL FAST before training: the full evaluation path runs on one image, three training
    steps run, and a checkpoint is saved and re-loaded (round trip).
  * per-epoch checkpoint (model, optimiser, scheduler, scaler, best state) to
    /kaggle/working/ckpt_<arm>.pt; if a ckpt_<arm>.pt is found under /kaggle/input (a
    previous version attached as a source), training resumes from it.
  * time budget: training stops early (keeping the best checkpoint) once TRAIN_BUDGET_S
    is used, so evaluation and the outputs are always written.
"""

import glob
import json
import os
import subprocess
import sys
import time

T_START = time.time()
ARM = os.environ.get("ARM", "union")
assert ARM in ("union", "consensus"), ARM
FOLD = 1
TRAIN_BUDGET_S = float(os.environ.get("TRAIN_BUDGET_S", str(4.5 * 3600)))
SEED = 20260926

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
import filament_unet_ensemble as tr   # production training code (runs data discovery on import)
import filament_pipeline as fp
import filament_metric as fm

OUT = "/kaggle/working"
MASK_DIR = f"/kaggle/temp/masks_{ARM}"
os.makedirs(MASK_DIR, exist_ok=True)
CKPT = os.path.join(OUT, f"ckpt_{ARM}.pt")
BEST = os.path.join(OUT, f"best_{ARM}.pt")
RESULT = os.path.join(OUT, f"pilot_{ARM}.json")
log = {"arm": ARM, "fold": FOLD, "seed": SEED, "events": []}


def note(msg):
    print(f"[{time.time() - T_START:7.0f}s] {msg}", flush=True)
    log["events"].append([round(time.time() - T_START), msg])
    json.dump(log, open(RESULT, "w"), indent=1)


# ------------------------------------------------------------------ targets
def instance_rles(anns, h, w):
    out = []
    for a in anns:
        seg = a.get("segmentation")
        if not seg:
            continue
        r = mask_util.frPyObjects(seg, h, w)
        out.append(mask_util.merge(r) if isinstance(r, list) else r)
    return out


def consensus_mask(passes, h, w, match_iou=0.3):
    """Union of the instances that are matched (IoU > match_iou) by an instance of another pass."""
    pr = [instance_rles(anns, h, w) for anns in passes]
    if len(pr) < 2:
        return pass_union(pr, h, w)
    keep = []
    for i, ri in enumerate(pr):
        others = [r for j, rj in enumerate(pr) if j != i for r in rj]
        if not ri or not others:
            continue
        iou = np.asarray(mask_util.iou(ri, others, [0] * len(others)), dtype=np.float64)
        keep += [ri[k] for k in range(len(ri)) if (iou[k] > match_iou).any()]
    if not keep:
        return np.zeros((h, w), np.uint8)
    return mask_util.decode(mask_util.merge(keep)).astype(np.uint8)


def pass_union(pr, h, w):
    flat = [r for rs in pr for r in rs]
    if not flat:
        return np.zeros((h, w), np.uint8)
    return mask_util.decode(mask_util.merge(flat)).astype(np.uint8)


def build_targets(names, per_file, dims):
    stats = {"pixels_union": 0, "pixels_target": 0, "multi_pass_images": 0}
    for fn in names:
        h, w = dims[fn]
        by_pass = {}
        for a in per_file[fn]:
            by_pass.setdefault(a["image_id"], []).append(a)
        passes = list(by_pass.values())
        union = pass_union([instance_rles(p, h, w) for p in passes], h, w)
        target = union if ARM == "union" else consensus_mask(passes, h, w)
        stats["pixels_union"] += int(union.sum())
        stats["pixels_target"] += int(target.sum())
        stats["multi_pass_images"] += int(len(passes) > 1)
        cv2.imwrite(os.path.join(MASK_DIR, fn.rsplit(".", 1)[0] + ".png"), target * 255)
    return stats


# ------------------------------------------------------------------ evaluation (production protocol)
def evaluate(weight_path, names, per_file_m, refiner, tag):
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
        if k % 25 == 0:
            note(f"eval {tag}: {k}/{len(names)}")
    return rows


def pooled_pq(rows, key):
    t = np.sum([r[key] for r in rows], 0)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return float(t[0] / d) if d else 0.0


# ------------------------------------------------------------------ main
def main():
    note(f"arm={ARM} device={tr.DEV} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    assert torch.cuda.is_available(), "no GPU -- refusing to train on CPU"
    names, per_file, dims = tr.build_masks()          # union masks in tr.MASK_DIR (unused by the consensus arm)
    per_file_m, _ = fm.load_annotations(tr.ANN, tr.TRAIN_IMG)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout = [names[i] for i in idx[:tr.N_VAL]]
    pool = [names[i] for i in idx[tr.N_VAL:]]
    assert holdout == fm.holdout_names(list(per_file_m)), "hold-out mismatch"
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), tr.N_FOLDS)
    fv = [pool[i] for i in folds[FOLD - 1]]
    ft = [pool[i] for i in np.concatenate([folds[j] for j in range(tr.N_FOLDS) if j != FOLD - 1])]
    note(f"fold {FOLD}: train {len(ft)}, fold-val {len(fv)}, hold-out {len(holdout)}")

    stats = build_targets(ft + fv, per_file, dims)
    log["target_stats"] = stats
    note(f"targets built: {stats} (target/union pixels = {stats['pixels_target'] / max(stats['pixels_union'], 1):.3f})")

    refiner = fp.load_refiner(next(p for p in glob.glob("/kaggle/input/**/refiner_best.pt", recursive=True)), fp.default_device())
    released = next(p for p in glob.glob(f"/kaggle/input/**/unet_fold{FOLD}.pt", recursive=True))

    # ---------------- FAIL FAST (minutes, before any real training)
    fast = evaluate(released, fv[:1], per_file_m, refiner, "fail-fast")
    note(f"fail-fast eval OK on {fv[0]}: {fast[0]['with_refiner']}")
    cache = {}
    for fn in ft + fv:
        img = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(MASK_DIR, fn.rsplit(".", 1)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        cache[fn] = (img, (msk > 127).astype(np.uint8))
    note(f"RAM cache: {len(cache)} images")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = tr.UNet().to(tr.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.LR, total_steps=tr.EPOCHS * tr.STEPS_PER_EPOCH, pct_start=0.25)
    scaler = torch.amp.GradScaler(enabled=True)
    ds = tr.PatchDS(ft, tr.STEPS_PER_EPOCH, cache)
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
        return loss.item()

    def save_ckpt(ep_done, best):
        torch.save({"epoch": ep_done, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best,
                    "arm": ARM, "seed": SEED}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)

    # checkpoint round trip on an untrained copy (verifies save + load before spending GPU hours)
    save_ckpt(0, (-1.0, -1, None))
    rt = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert rt["arm"] == ARM and set(rt["model"]) == set(model.state_dict()), "checkpoint round trip failed"
    note("checkpoint round trip OK")

    start_ep, best = 0, (-1.0, -1, None)
    prev = [p for p in glob.glob(f"/kaggle/input/**/ckpt_{ARM}.pt", recursive=True)]
    if prev:
        ck = torch.load(prev[0], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best = ck["epoch"], ck["best"]
        note(f"RESUMED from {prev[0]} at epoch {start_ep} (best {best[0]:.4f} @ {best[1]})")
    else:
        # three real steps, timed, to estimate the run before committing to it
        model.train()
        t0 = time.time()
        it = iter(dl)
        for _ in range(3):
            x, y = next(it)
            step(x, y)
        torch.cuda.synchronize()
        del it
        note(f"3 training steps OK ({(time.time() - t0) / 3:.2f}s/step incl. warm-up)")
        # the smoke steps changed the weights and schedule: restart cleanly from the same seed
        torch.manual_seed(SEED)
        model = tr.UNet().to(tr.DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.LR, total_steps=tr.EPOCHS * tr.STEPS_PER_EPOCH, pct_start=0.25)
        scaler = torch.amp.GradScaler(enabled=True)

    quick = fv[:tr.VAL_QUICK]
    t_train = time.time()
    stopped_early = False
    for ep in range(start_ep, tr.EPOCHS):
        model.train()
        tot = 0.0
        for x, y in dl:
            tot += step(x, y)
        if (ep + 1) % tr.VAL_EVERY == 0 or ep == tr.EPOCHS - 1:
            model.eval()
            S = T = FP = FN = 0.0
            for fn in quick:
                g = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
                lab = tr.instances(tr.predict_prob(model, g), 0.4, 400, min_mean_prob=0.7)
                s_, tp, fp_, fn_ = tr.pq_counts(lab, tr.val_gt(fn, per_file, dims))
                S, T, FP, FN = S + s_, T + tp, FP + fp_, FN + fn_
            v = tr.pq(S, T, FP, FN)
            if v > best[0]:
                best = (v, ep + 1, None)
                torch.save({k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, BEST)
            note(f"ep{ep + 1:2d} loss={tot / len(dl):.4f} foldval_PQ={v:.4f} best={best[0]:.4f}@{best[1]}")
        else:
            note(f"ep{ep + 1:2d} loss={tot / len(dl):.4f}")
        save_ckpt(ep + 1, best)
        per_ep = (time.time() - t_train) / (ep + 1 - start_ep)
        if time.time() - T_START + per_ep > TRAIN_BUDGET_S and ep + 1 < tr.EPOCHS:
            stopped_early = True
            note(f"TIME BUDGET: stopping after epoch {ep + 1} (best checkpoint kept)")
            break
    log["train"] = {"best_foldval_pq": best[0], "best_epoch": best[1], "stopped_early": stopped_early,
                    "train_seconds": round(time.time() - t_train)}
    del dl, ds
    torch.cuda.empty_cache()
    if not os.path.exists(BEST):
        raise SystemExit("no best checkpoint was saved")

    log["oof"] = evaluate(BEST, fv, per_file_m, refiner, "OOF")
    log["holdout"] = evaluate(BEST, holdout, per_file_m, refiner, "hold-out")
    for part in ("oof", "holdout"):
        log[f"{part}_pq"] = {k: round(pooled_pq(log[part], k), 4) for k in ("with_refiner", "without_refiner")}
    note(f"DONE oof={log['oof_pq']} holdout={log['holdout_pq']}")
    if os.path.exists(CKPT):
        os.remove(CKPT)   # training finished: keep only the best weights in the output


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # keep whatever was logged/checkpointed in the output
        note(f"ERROR {type(e).__name__}: {e}")
        raise
