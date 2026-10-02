"""
Solar Filament Segmentation 2026 -- multi-scale test-time augmentation, out of fold, Kaggle GPU, INFERENCE ONLY.

Stage-A currently averages 4 flips at one scale (512 px windows on the 2048 px image). Multi-scale TTA
(Tao et al. 2020; Zoph et al. 2020) adds scales 0.75 and 1.25: the 0.75 pass lets every window see more
context (it helps faint mid-filament sections, i.e. fragmentation), the 1.25 pass sees thin filaments at a
higher resolution. This script scores the released fold models on their own validation images (657 in all)
with six scale combinations and the production candidate rule, refiner and near-limb filter, so that every
combination is compared PAIRED against the single-scale baseline from the same run.

Combinations: s100 (baseline), s075, s125, s100_075, s100_125, s075_100_125 (the pre-registered primary).
Per image and combination it stores the official counts WITH and WITHOUT the near-limb filter.

Safety (the GPU is shared with other research):
  * fail-fast: CUDA is required; the first three images of the first fold must reproduce the stored
    out-of-fold counts of the CPU run (within +-1 per count) before anything else is computed;
  * every image is checked for non-finite probabilities; any error is written to the output JSON and
    re-raised (the kernel stops, nothing is retried);
  * results are written every 5 images, a fold is marked done only when all its images are scored,
    and a wall-clock budget (TIME_BUDGET_S, default 3 h) ends the run cleanly with status "partial".
Environment: FOLDS (e.g. "1,2,3"), TIME_BUDGET_S, REQUIRE_CUDA (default 1).
"""

import glob
import json
import os
import subprocess
import sys
import time
import traceback

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
import filament_metric as fm
import filament_pipeline as fp

FOLDS = [int(f) for f in os.environ.get("FOLDS", "1,2,3").split(",")]
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 3 * 3600))
INPUT, OUT = "/kaggle/input", "/kaggle/working"
SCALES = (0.75, 1.0, 1.25)
COMBOS = {"s100": (1.0,), "s075": (0.75,), "s125": (1.25,), "s100_075": (1.0, 0.75), "s100_125": (1.0, 1.25),
          "s075_100_125": (0.75, 1.0, 1.25)}
STORED_FIRST = {}          # fold -> [[name, sum_iou, tp, fp, fn], ...] of the first images of the CPU out-of-fold run
t0 = time.time()
state = {"folds": FOLDS, "status": "starting", "error": None, "combos": {k: list(v) for k, v in COMBOS.items()}}


def dump(name="msc_state.json", obj=None):
    json.dump(obj if obj is not None else state, open(f"{OUT}/{name}", "w"))


def scaled_prob(models, g, s, device):
    """Stage-A probability map at scale s, resampled back to the image size."""
    if s == 1.0:
        return fp.predict_ensemble(models, g, device)
    size = int(round(g.shape[0] * s))
    gs = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    p = fp.predict_ensemble(models, gs, device)
    return cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)


def score_image(g, passes, models, refiner, device):
    h, w = g.shape
    probs = {s: scaled_prob(models, g, s, device) for s in SCALES}
    for s, p in probs.items():
        assert np.isfinite(p).all(), f"non-finite probabilities at scale {s}"
    out = {}
    for name, ss in COMBOS.items():
        prob = np.mean([probs[s] for s in ss], 0)
        cand = fp.extract_candidates(prob)
        lab = fp.refine_candidates(g, cand, refiner, device)
        lim = fp.drop_near_limb(lab, g)
        c0 = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
        c1 = fm.pq_counts(fp.label_map_to_rles(lim), passes, h, w)
        out[name] = {"nolimb": [float(c0[0]), *map(int, c0[1:])], "limb": [float(c1[0]), *map(int, c1[1:])], "n_cand": int(cand.max())}
    return out


def main():
    if os.environ.get("REQUIRE_CUDA", "1") == "1":
        assert torch.cuda.is_available(), "STOP: no CUDA device"
    device = fp.default_device()
    ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
    train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
    per_file, _ = fm.load_annotations(ann_json, train_dir)
    names = sorted(per_file)
    assert len(names) == 707, len(names)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout, pool = [names[i] for i in idx[:50]], [names[i] for i in idx[50:]]
    assert holdout == fm.holdout_names(list(per_file)), "hold-out mismatch"
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), 5)
    assert len(set(np.concatenate(folds).tolist())) == len(pool) == 657
    refiner = fp.load_refiner(next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True)), device)
    print(f"device {device}; folds {FOLDS}", flush=True)

    checked = False
    for F in FOLDS:
        path = next(p for p in glob.glob(f"{INPUT}/**/unet_fold{F}.pt", recursive=True))
        models = fp.load_stage_a([path], device)
        val = [pool[i] for i in folds[F - 1]]
        assert not set(val) & set(holdout)
        res = {"fold": F, "status": "running", "n_images": len(val), "results": []}
        for k, name in enumerate(val, 1):
            if time.time() - t0 > TIME_BUDGET_S:
                res["status"] = "partial"
                dump(f"msc_fold{F}.json", res)
                state["status"] = "partial"
                print(f"time budget reached in fold {F} after {k - 1} images", flush=True)
                return
            g = fp.read_gray(os.path.join(train_dir, name))
            out = score_image(g, per_file[name], models, refiner, device)
            res["results"].append({"name": name, "n_pass": len(per_file[name]), "combos": out})
            if not checked and k == min(3, len(val)) and STORED_FIRST.get(str(F)):
                for r, ref in zip(res["results"], STORED_FIRST[str(F)]):
                    assert r["name"] == ref[0], (r["name"], ref[0])
                    got = r["combos"]["s100"]["nolimb"]
                    assert all(abs(a - b) <= 1 for a, b in zip(got[1:], ref[2:])), \
                        f"STOP: single-scale counts {got} do not reproduce the stored CPU run {ref} on {name}"
                checked = True
                print("integrity check passed: single-scale counts reproduce the stored out-of-fold run", flush=True)
            if k % 5 == 0 or k == len(val):
                print(f"fold {F}: {k}/{len(val)} {time.time() - t0:.0f}s", flush=True)
                dump(f"msc_fold{F}.json", res)
        res["status"] = "done"
        dump(f"msc_fold{F}.json", res)
    state["status"] = "done"


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:                       # noqa: BLE001 -- record any failure, then stop
        state.update(status="error", error="".join(traceback.format_exception_only(type(e), e)).strip())
        dump()
        raise
    finally:
        state["elapsed_s"] = round(time.time() - t0)
        dump()
    print("done", state["status"], state["elapsed_s"], "s", flush=True)
