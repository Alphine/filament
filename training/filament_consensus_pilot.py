"""
Solar Filament Segmentation 2026 -- consensus-target training (Kaggle GPU).

ARM=union      control: exactly the production target (pixel union of all passes)
ARM=consensus  on multi-pass images keep only instances matched (IoU > 0.3) by an instance
               of at least one OTHER pass; single-pass images unchanged. Generalises the
               "inst_union_drop" constructor of analysis/scripts/filament_consensus_cpu.py,
               which scored +0.0145 PQ over the union in an annotation-only simulation.
FOLDS          comma-separated fold numbers trained one after another in this kernel
               (the pilot used FOLDS=1 for both arms; the full run uses the consensus arm
               for folds 2-5).

Everything else is the production recipe of training/filament_unet_ensemble.py (imported,
not copied): same splits, ResNet34 U-Net, 30 epochs x 220 steps, batch 8, Tversky+BCE,
AdamW + OneCycle 1e-4, checkpoint chosen every 5 epochs on 20 fold-val images. Every
fold starts from the same torch seed, so arms differ only in the target (the data loader
is already deterministic).

Evaluation per fold (same protocol as analysis/scripts/filament_oof_cpu.py): the best
checkpoint alone with 4-flip TTA, production thresholds and the released crop refiner,
scored per image on the fold's validation (out-of-fold) images and on the 50-image hold-out.

Compute safety:
  * FAIL FAST before training: the full evaluation path runs on one image, three training
    steps run (timed), and a checkpoint is saved and re-loaded (round trip).
  * per-epoch checkpoint (model, optimiser, scheduler, scaler, best) to
    /kaggle/working/ckpt_<arm>_fold<F>.pt; if such a file is found under /kaggle/input (a
    previous version attached as a source), that fold resumes from it; a fold whose
    best_<arm>_fold<F>.pt and result JSON are already in /kaggle/input is skipped.
  * time budget: training stops early (keeping the best checkpoint) when the kernel's
    elapsed time would exceed TRAIN_BUDGET_S, so evaluation and outputs are always written.

Outputs per fold: best_<arm>_fold<F>.pt (state_dict) and pilot_<arm>_fold<F>.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

T_START = time.time()
ARM = os.environ.get("ARM", "union")
assert ARM in ("union", "consensus"), ARM
FOLDS = [int(f) for f in os.environ.get("FOLDS", "1").split(",")]
TRAIN_BUDGET_S = float(os.environ.get("TRAIN_BUDGET_S", str(10.5 * 3600)))
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


class Log:
    def __init__(self, fold):
        self.path = os.path.join(OUT, f"pilot_{ARM}_fold{fold}.json")
        self.d = {"arm": ARM, "fold": fold, "seed": SEED, "events": []}

    def note(self, msg):
        print(f"[{time.time() - T_START:7.0f}s] fold{self.d['fold']} {msg}", flush=True)
        self.d["events"].append([round(time.time() - T_START), msg])
        self.save()

    def save(self):
        json.dump(self.d, open(self.path, "w"), indent=1)


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


def pass_union(pr, h, w):
    flat = [r for rs in pr for r in rs]
    if not flat:
        return np.zeros((h, w), np.uint8)
    return mask_util.decode(mask_util.merge(flat)).astype(np.uint8)


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
        if k % 25 == 0:
            log.note(f"eval {tag}: {k}/{len(names)}")
    return rows


def pooled_pq(rows, key):
    t = np.sum([r[key] for r in rows], 0)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return float(t[0] / d) if d else 0.0


def fresh(ft_len_steps):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = tr.UNet().to(tr.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.LR, total_steps=ft_len_steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(enabled=True)
    return model, opt, sched, scaler


# ------------------------------------------------------------------ one fold
def run_fold(fold, S):
    log = Log(fold)
    ckpt = os.path.join(OUT, f"ckpt_{ARM}_fold{fold}.pt")
    best_path = os.path.join(OUT, f"best_{ARM}_fold{fold}.pt")
    done_w = glob.glob(f"/kaggle/input/**/best_{ARM}_fold{fold}.pt", recursive=True)
    done_j = glob.glob(f"/kaggle/input/**/pilot_{ARM}_fold{fold}.json", recursive=True)
    if done_w and done_j and "oof_pq" in json.load(open(done_j[0])):
        shutil.copy(done_w[0], best_path)
        shutil.copy(done_j[0], log.path)
        print(f"fold{fold}: already finished in an attached source -- copied, skipped", flush=True)
        return

    folds, pool = S["folds"], S["pool"]
    fv = [pool[i] for i in folds[fold - 1]]
    ft = [pool[i] for i in np.concatenate([folds[j] for j in range(tr.N_FOLDS) if j != fold - 1])]
    log.d["target_stats"] = S["stats"]
    log.note(f"train {len(ft)}, fold-val {len(fv)}, hold-out {len(S['holdout'])}")

    released = next(p for p in glob.glob(f"/kaggle/input/**/unet_fold{fold}.pt", recursive=True))
    fast = evaluate(released, fv[:1], S["per_file_m"], S["refiner"], log, "fail-fast")
    log.note(f"fail-fast eval OK on {fv[0]}: {fast[0]['with_refiner']}")

    total_steps = tr.EPOCHS * tr.STEPS_PER_EPOCH
    model, opt, sched, scaler = fresh(total_steps)
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
        return loss.item()

    def save_ckpt(ep_done, best):
        torch.save({"epoch": ep_done, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best,
                    "arm": ARM, "fold": fold, "seed": SEED}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    save_ckpt(0, (-1.0, -1))
    rt = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert rt["arm"] == ARM and rt["fold"] == fold and set(rt["model"]) == set(model.state_dict())
    log.note("checkpoint round trip OK")

    start_ep, best = 0, (-1.0, -1)
    prev = glob.glob(f"/kaggle/input/**/ckpt_{ARM}_fold{fold}.pt", recursive=True)
    prev_best = glob.glob(f"/kaggle/input/**/best_{ARM}_fold{fold}.pt", recursive=True)
    if prev:
        ck = torch.load(prev[0], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best = ck["epoch"], tuple(ck["best"])[:2]
        if prev_best:
            shutil.copy(prev_best[0], best_path)
        log.note(f"RESUMED from {prev[0]} at epoch {start_ep} (best {best[0]:.4f} @ {best[1]})")
    else:
        model.train()
        t0 = time.time()
        it = iter(dl)
        for _ in range(3):
            x, y = next(it)
            step(x, y)
        torch.cuda.synchronize()
        del it
        log.note(f"3 training steps OK ({(time.time() - t0) / 3:.2f}s/step incl. warm-up)")
        model, opt, sched, scaler = fresh(total_steps)   # the smoke steps changed the weights: restart cleanly

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
            s_sum = t_sum = fp_sum = fn_sum = 0.0
            for fn in quick:
                g = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
                lab = tr.instances(tr.predict_prob(model, g), 0.4, 400, min_mean_prob=0.7)
                s_, tp, fp_, fn_ = tr.pq_counts(lab, tr.val_gt(fn, S["per_file"], S["dims"]))
                s_sum, t_sum, fp_sum, fn_sum = s_sum + s_, t_sum + tp, fp_sum + fp_, fn_sum + fn_
            v = tr.pq(s_sum, t_sum, fp_sum, fn_sum)
            if v > best[0]:
                best = (v, ep + 1)
                torch.save({k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, best_path)
            log.note(f"ep{ep + 1:2d} loss={tot / len(dl):.4f} foldval_PQ={v:.4f} best={best[0]:.4f}@{best[1]}")
        else:
            log.note(f"ep{ep + 1:2d} loss={tot / len(dl):.4f}")
        save_ckpt(ep + 1, best)
        per_ep = (time.time() - t_train) / (ep + 1 - start_ep)
        if time.time() - T_START + per_ep > TRAIN_BUDGET_S and ep + 1 < tr.EPOCHS:
            stopped_early = True
            log.note(f"TIME BUDGET: stopping after epoch {ep + 1} (best checkpoint kept)")
            break
    log.d["train"] = {"best_foldval_pq": best[0], "best_epoch": best[1], "stopped_early": stopped_early,
                      "train_seconds": round(time.time() - t_train)}
    del dl, ds
    torch.cuda.empty_cache()
    if not os.path.exists(best_path):
        raise SystemExit(f"fold {fold}: no best checkpoint was saved")

    log.d["oof"] = evaluate(best_path, fv, S["per_file_m"], S["refiner"], log, "OOF")
    log.d["holdout"] = evaluate(best_path, S["holdout"], S["per_file_m"], S["refiner"], log, "hold-out")
    for part in ("oof", "holdout"):
        log.d[f"{part}_pq"] = {k: round(pooled_pq(log.d[part], k), 4) for k in ("with_refiner", "without_refiner")}
    log.note(f"DONE oof={log.d['oof_pq']} holdout={log.d['holdout_pq']}")
    if os.path.exists(ckpt) and not stopped_early:
        os.remove(ckpt)   # finished: keep only the best weights in the output


# ------------------------------------------------------------------ main
def main():
    print(f"arm={ARM} folds={FOLDS} device={tr.DEV} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)
    assert torch.cuda.is_available(), "no GPU -- refusing to train on CPU"
    names, per_file, dims = tr.build_masks()
    per_file_m, _ = fm.load_annotations(tr.ANN, tr.TRAIN_IMG)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout = [names[i] for i in idx[:tr.N_VAL]]
    pool = [names[i] for i in idx[tr.N_VAL:]]
    assert holdout == fm.holdout_names(list(per_file_m)), "hold-out mismatch"
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), tr.N_FOLDS)
    stats = build_targets(pool, per_file, dims)
    print(f"targets built for {len(pool)} pool images: {stats} "
          f"(target/union pixels {stats['pixels_target'] / max(stats['pixels_union'], 1):.3f})", flush=True)
    cache = {}
    for fn in pool:
        img = cv2.imread(os.path.join(tr.TRAIN_IMG, fn), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(MASK_DIR, fn.rsplit(".", 1)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        cache[fn] = (img, (msk > 127).astype(np.uint8))
    refiner = fp.load_refiner(next(p for p in glob.glob("/kaggle/input/**/refiner_best.pt", recursive=True)),
                              fp.default_device())
    S = dict(per_file=per_file, dims=dims, per_file_m=per_file_m, holdout=holdout, pool=pool, folds=folds,
             stats=stats, cache=cache, refiner=refiner)
    failed = []
    for fold in FOLDS:
        try:
            run_fold(fold, S)
        except Exception as e:  # keep the other folds going; record the error in that fold's JSON
            print(f"fold{fold} ERROR {type(e).__name__}: {e}", flush=True)
            p = os.path.join(OUT, f"pilot_{ARM}_fold{fold}.json")
            d = json.load(open(p)) if os.path.exists(p) else {"arm": ARM, "fold": fold, "events": []}
            d["error"] = f"{type(e).__name__}: {e}"
            json.dump(d, open(p, "w"), indent=1)
            failed.append(fold)
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"folds failed: {failed}")


if __name__ == "__main__":
    main()
