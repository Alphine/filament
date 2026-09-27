"""
Solar Filament Segmentation 2026 -- crop-refiner generation 2 (Kaggle GPU), cross-fitted.

Why: 54% of the pipeline's errors are near-misses (a prediction overlaps a filament with
IoU <= 0.5) and the crop refiner is the only change that ever raised PQ (+0.018). The released
refiner was trained on all 657 pool images, so it cannot be evaluated out of fold; this study
cross-fits instead: the pool is split into two halves (fixed seed); refiners are trained on one
half and applied to the out-of-fold stage-A candidates (stored by filament_oof_cpu.py) of the
other half. HALF=0 trains on half 0 and evaluates on half 1; HALF=1 the reverse. Pooling both
kernels scores all 657 images with refiners that never saw them.

Recipes (same seed, same 13-epoch schedule, same data pipeline as training/filament_refiner.py,
whose code is imported):
  gen1    production recipe: smp.Unet(resnet34), 0.45 BCE + 0.55 soft Dice
  lovasz  as gen1 but loss 0.5 BCE + 0.5 Lovasz hinge (per-image IoU surrogate)
  effb3   as gen1 but encoder efficientnet-b3
Every trained refiner is evaluated without and with 4-flip test-time augmentation of the crop.
Decision rule fixed beforehand: an arm replaces gen1 only if its paired gain over the 657 images
(official PQ, pooled over passes) is >= +0.005 with a bootstrap CI95 above 0.

Compute safety:
  * FAIL FAST: the released refiner applied to the first evaluation image's stored candidates
    must reproduce the stored out-of-fold counts; every recipe runs 2 training steps and a
    checkpoint save/load round trip before training.
  * per-epoch checkpoint per recipe (model, optimiser, scheduler, scaler, best), resumable from
    /kaggle/input; a recipe already finished in an attached source is skipped; results are
    written after every recipe; per-recipe time budget.
Outputs: gen2_half<H>.json (per-image counts per arm), ref_<recipe>_half<H>.pt (best weights).
"""

import glob
import json
import os
import random
import shutil
import sys
import time

T_START = time.time()
HALF = int(os.environ.get("HALF", "0"))
RECIPES = os.environ.get("RECIPES", "gen1,lovasz,effb3").split(",")
RECIPE_BUDGET_S = float(os.environ.get("RECIPE_BUDGET_S", str(2.0 * 3600)))
SEED = 20261001

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
import filament_refiner as rf          # production refiner code (data discovery runs on import)
import filament_pipeline as fp
import filament_metric as fm
import segmentation_models_pytorch as smp

OUT = "/kaggle/working"
RESULT = os.path.join(OUT, f"gen2_half{HALF}.json")
DEV = rf.DEV
log = {"half": HALF, "recipes": RECIPES, "seed": SEED, "events": [], "arms": {}}


def note(msg):
    print(f"[{time.time() - T_START:7.0f}s] {msg}", flush=True)
    log["events"].append([round(time.time() - T_START), msg])
    json.dump(log, open(RESULT, "w"))


# ------------------------------------------------------------------ losses
def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    if gt_sorted.numel() > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_hinge(logits, labels):
    """Binary Lovasz hinge (Berman et al. 2018), averaged over the images of a batch."""
    losses = []
    for lg, lb in zip(logits.float(), labels.float()):
        lg, lb = lg.reshape(-1), lb.reshape(-1)
        errors = 1.0 - lg * (2.0 * lb - 1.0)
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(F.relu(errors_sorted), lovasz_grad(lb[perm])))
    return torch.stack(losses).mean()


def loss_fn(recipe, logits, y):
    if recipe == "lovasz":
        return 0.5 * F.binary_cross_entropy_with_logits(logits, y) + 0.5 * lovasz_hinge(logits, y)
    return 0.45 * F.binary_cross_entropy_with_logits(logits, y) + 0.55 * rf.soft_dice_loss(logits, y)


def build(recipe, pretrained=True):
    enc = "efficientnet-b3" if recipe == "effb3" else "resnet34"
    return smp.Unet(encoder_name=enc, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def refine(gray, cand, model, tta):
    """fp.refine_candidates with an arbitrary refiner and optional 4-flip TTA of the crop
    (identical to fp.refine_candidates when tta is False)."""
    h, w = gray.shape
    out = np.zeros_like(cand)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    for i in range(1, int(cand.max()) + 1):
        box = fp._bbox(cand == i)
        if box is None:
            continue
        y0_, y1_, x0_, x1_ = box
        bbox = [x0_, y0_, x1_ - x0_, y1_ - y0_]
        seed = fp.seed_box(h, w, bbox)
        x0, y0, x1, y1 = fp.square_bounds(bbox, h, w)
        cs = fp.REFINE_CROP
        img = cv2.resize(gray[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_LINEAR)
        sd = cv2.resize(seed[y0:y1, x0:x1], (cs, cs), interpolation=cv2.INTER_NEAREST)
        x = np.stack([img, clahe.apply(img), sd * 255], 0).astype(np.float32) / 255.0
        x = torch.from_numpy((x - fp.IMAGENET_MEAN) / fp.IMAGENET_STD)[None].to(DEV)
        if tta:
            views = [x, x.flip(-1), x.flip(-2), x.flip(-1, -2)]
            probs = torch.sigmoid(model(torch.cat(views))).float()
            prob = (probs[0] + probs[1].flip(-1) + probs[2].flip(-2) + probs[3].flip(-1, -2)) / 4
            prob = prob.cpu().numpy()[0]
        else:
            prob = torch.sigmoid(model(x)).float().cpu().numpy()[0, 0]
        crop_mask = (prob > fp.REFINE_THR).astype(np.uint8)
        full = cv2.resize(crop_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        out[y0:y1, x0:x1][full.astype(bool)] = i
    return out


def evaluate(model, eval_names, S, tag):
    model.eval()
    rows = {False: [], True: []}
    for k, name in enumerate(eval_names, 1):
        g = fp.read_gray(os.path.join(rf.TRAIN_IMG, name))
        h, w = g.shape
        cand = S["cand"](name)
        for tta in (False, True):
            c = fm.pq_counts(fp.label_map_to_rles(refine(g, cand, model, tta)), S["per_file_m"][name], h, w)
            rows[tta].append({"name": name, "counts": [float(c[0]), *map(int, c[1:])]})
        if k % 100 == 0:
            note(f"eval {tag}: {k}/{len(eval_names)}")
    return rows


def pooled(rows):
    t = np.sum([r["counts"] for r in rows], 0)
    return float(fm.pq(*t))


# ------------------------------------------------------------------ one recipe
def run_recipe(recipe, S):
    ck = os.path.join(OUT, f"ckpt_{recipe}_half{HALF}.pt")
    best_path = os.path.join(OUT, f"ref_{recipe}_half{HALF}.pt")
    prev_res = glob.glob(f"/kaggle/input/**/gen2_half{HALF}.json", recursive=True)
    if prev_res:
        d = json.load(open(prev_res[0]))
        if recipe in d.get("done", []):
            for arm in (recipe, recipe + "+tta"):
                log["arms"][arm] = d["arms"][arm]
            src = glob.glob(f"/kaggle/input/**/ref_{recipe}_half{HALF}.pt", recursive=True)
            if src:
                shutil.copy(src[0], best_path)
            log.setdefault("done", []).append(recipe)
            note(f"{recipe}: already finished in an attached source -- skipped")
            return

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = build(recipe).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=rf.REFINE_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rf.REFINE_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    train_dl = DataLoader(rf.RefineDataset(S["train_items"], rf.REFINE_CROP, augment=True), batch_size=rf.REFINE_BATCH,
                          shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(rf.RefineDataset(S["val_items"], rf.REFINE_CROP, augment=False), batch_size=rf.REFINE_BATCH * 2,
                        shuffle=False, num_workers=2, pin_memory=True)

    def step(x, y):
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            logits = model(x)
        loss = loss_fn(recipe, logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss.item()

    def save_ckpt(ep_done, best):
        torch.save({"epoch": ep_done, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best,
                    "recipe": recipe, "half": HALF}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    # fail fast: two real steps and a checkpoint round trip
    model.train()
    it = iter(train_dl)
    t0 = time.time()
    for _ in range(2):
        step(*next(it))
    torch.cuda.synchronize()
    del it
    save_ckpt(0, (-1.0, -1))
    rt = torch.load(ck, map_location="cpu", weights_only=False)
    assert rt["recipe"] == recipe and set(rt["model"]) == set(model.state_dict())
    note(f"{recipe}: 2 steps OK ({(time.time() - t0) / 2:.2f}s/step), checkpoint round trip OK")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = build(recipe).to(DEV)                    # restart cleanly after the smoke steps
    opt = torch.optim.AdamW(model.parameters(), lr=rf.REFINE_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rf.REFINE_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    start, best = 0, (-1.0, -1)
    prev = glob.glob(f"/kaggle/input/**/ckpt_{recipe}_half{HALF}.pt", recursive=True)
    if prev:
        c = torch.load(prev[0], map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"]); start, best = c["epoch"], tuple(c["best"])
        pb = glob.glob(f"/kaggle/input/**/ref_{recipe}_half{HALF}.pt", recursive=True)
        if pb:
            shutil.copy(pb[0], best_path)
        note(f"{recipe}: RESUMED at epoch {start}")
    t_rec = time.time()
    for ep in range(start, rf.REFINE_EPOCHS):
        model.train()
        tl = [step(x, y) for x, y in train_dl]
        sched.step()
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                dices.append(rf.batch_dice(model(x), y))
        vd = float(np.mean(dices))
        if vd > best[0]:
            best = (vd, ep + 1)
            torch.save({k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, best_path)
        save_ckpt(ep + 1, best)
        note(f"{recipe} ep{ep + 1:2d} loss={np.mean(tl):.4f} val_dice={vd:.4f} best={best[0]:.4f}@{best[1]}")
        per_ep = (time.time() - t_rec) / (ep + 1 - start)
        if time.time() - t_rec + per_ep > RECIPE_BUDGET_S and ep + 1 < rf.REFINE_EPOCHS:
            note(f"{recipe}: TIME BUDGET, stopping after epoch {ep + 1}")
            break
    del train_dl, val_dl
    model.load_state_dict(torch.load(best_path, map_location=DEV, weights_only=True))
    rows = evaluate(model, S["eval_names"], S, recipe)
    log["arms"][recipe] = {"pq": pooled(rows[False]), "best_epoch": best[1], "rows": rows[False]}
    log["arms"][recipe + "+tta"] = {"pq": pooled(rows[True]), "best_epoch": best[1], "rows": rows[True]}
    log.setdefault("done", []).append(recipe)
    note(f"{recipe}: DONE pq={log['arms'][recipe]['pq']:.4f} +tta={log['arms'][recipe + '+tta']['pq']:.4f}")
    if os.path.exists(ck):
        os.remove(ck)


def main():
    note(f"half={HALF} recipes={RECIPES} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    assert torch.cuda.is_available(), "no GPU -- refusing to train on CPU"
    names, per_file, dims = rf.load_annotations()
    per_file_m, _ = fm.load_annotations(rf.ANN, rf.TRAIN_IMG)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout, pool = [names[i] for i in idx[:rf.N_VAL]], [names[i] for i in idx[rf.N_VAL:]]
    assert holdout == fm.holdout_names(list(per_file_m)), "hold-out mismatch"
    perm = np.random.default_rng(2027).permutation(len(pool))
    halves = [sorted(pool[i] for i in perm[: len(pool) // 2]), sorted(pool[i] for i in perm[len(pool) // 2:])]
    assert not set(halves[0]) & set(halves[1]) and len(halves[0]) + len(halves[1]) == len(pool)
    train_half, eval_names = halves[HALF], halves[1 - HALF]
    rng = np.random.default_rng(7)
    tp = [train_half[i] for i in rng.permutation(len(train_half))]
    n_val = max(1, int(0.15 * len(tp)))
    S = {"per_file_m": per_file_m, "eval_names": eval_names,
         "train_items": rf.collect_instances(tp[n_val:], per_file, dims),
         "val_items": rf.collect_instances(tp[:n_val], per_file, dims)}
    store = {}
    for p in glob.glob("/kaggle/input/**/oof_preds_fold*.npz", recursive=True):
        for k in np.load(p).files:
            if k.endswith("__cand"):
                store[k[:-6]] = p
    S["cand"] = lambda name: np.load(store[name.rsplit(".", 1)[0]])[name.rsplit(".", 1)[0] + "__cand"].astype(np.int32)
    assert all(n.rsplit(".", 1)[0] in store for n in eval_names), "stored OOF candidates missing"
    note(f"train half {len(train_half)} images ({len(S['train_items'])} train / {len(S['val_items'])} val instances), "
         f"eval half {len(eval_names)} images")

    # FAIL FAST: released refiner + stored candidates must reproduce the stored OOF counts
    ref_counts = {}
    for p in glob.glob("/kaggle/input/**/oof_fold*.json", recursive=True):
        ref_counts.update({r["name"]: r["with_refiner"] for r in json.load(open(p))["results"]})
    released = fp.load_refiner(next(p for p in glob.glob("/kaggle/input/**/refiner_best.pt", recursive=True)), DEV)
    n0 = eval_names[0]
    g = fp.read_gray(os.path.join(rf.TRAIN_IMG, n0))
    got = fm.pq_counts(fp.label_map_to_rles(refine(g, S["cand"](n0), released, False)), per_file_m[n0], *g.shape)
    assert list(got[1:]) == ref_counts[n0][1:], f"FAIL FAST: {got[1:]} != stored {ref_counts[n0][1:]} on {n0}"
    note(f"fail-fast OK on {n0}: {list(got[1:])}")
    del released
    torch.cuda.empty_cache()

    for recipe in RECIPES:
        try:
            run_recipe(recipe, S)
        except Exception as e:     # keep the other recipes going; the error is logged
            note(f"{recipe}: ERROR {type(e).__name__}: {e}")
        torch.cuda.empty_cache()   # the recipe's model and optimiser are freed when run_recipe returns
    note("ALL DONE")


if __name__ == "__main__":
    main()
