"""
Solar Filament Segmentation 2026 -- stage-A maps of the retrained fold models on their own out-of-fold images,
CPU-ONLY, NO TRAINING. One kernel per (model set, fold): FOLD = 1..5, WEIGHT_PREFIX = "resample" (E1: per-epoch
patch re-sampling) or "sel" (the checkpoint-selection baseline), both the EMA weights.

Why: the E1 gate compared the two sets with the production candidate thresholds, refiner and scorer, all fitted to
the maps of models trained on repeated patches. E1 found more true positives (+232) but many more false positives
(+1,212), so the comparison may reflect calibration rather than quality. The fair test refits the instance scorer
on each set's own maps (analyze_recal.py). This script produces the maps and instances of one fold:

  * images: the fold's validation images minus the first 20 (the 20 the production rule selects on), as in E1;
  * production candidate thresholds (0.4 / area 400-40000 / mean p >= 0.7), 4-flip TTA, released crop refiner;
  * stored per image: official counts with and without the refiner, and (npz) the uint8 probability map,
    candidate map and refined label map, so features can be computed later without inference.

FAIL FAST: the first 3 images must reproduce the counts stored by the GPU evaluation (+-1 per count: CPU and GPU
differ by floating-point order); any error is written to the state file and re-raised (no retries).
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

import numpy as np

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

FOLD = int(os.environ.get("FOLD", "1"))
PREFIX = os.environ.get("WEIGHT_PREFIX", "resample")
TAG = {"resample": "e1", "sel": "base"}[PREFIX]
VAL_QUICK = 20
INPUT, OUT = "/kaggle/input", "/kaggle/working"
STORED = {}      # name -> [pq_sum, tp, fp, fn] with refiner, EMA weights, from the GPU evaluation (first 3 images)
t0 = time.time()
state = {"tag": TAG, "fold": FOLD, "status": "starting", "error": None}


def main():
    weights = next(p for p in glob.glob(f"{INPUT}/**/{PREFIX}_fold{FOLD}_ema.pt", recursive=True))
    refiner_path = next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True))
    ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
    train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
    device = fp.default_device()
    model = fp.load_stage_a([weights], device)
    refiner = fp.load_refiner(refiner_path, device)
    per_file, dims = fm.load_annotations(ann_json, train_dir)

    names = sorted(per_file)
    assert len(names) == 707, len(names)
    idx = np.random.default_rng(1234).permutation(len(names))
    holdout = [names[i] for i in idx[:50]]
    pool = [names[i] for i in idx[50:]]
    assert holdout == fm.holdout_names(list(per_file)), "hold-out mismatch"
    folds = np.array_split(np.random.default_rng(99).permutation(len(pool)), 5)
    fv = [pool[i] for i in folds[FOLD - 1]]
    val = fv[VAL_QUICK:]
    assert not set(val) & set(holdout)
    print(f"{TAG} fold {FOLD}: {len(val)} images; weights {weights}", flush=True)
    assert all(n in STORED for n in val[:3]), "integrity data missing for the first images"

    results, store = [], {}
    for k, name in enumerate(val, 1):
        t_img = time.time()
        g = fp.read_gray(os.path.join(train_dir, name))
        h, w = g.shape
        passes = per_file[name]
        prob = fp.predict_ensemble(model, g, device)
        cand = fp.extract_candidates(prob)
        lab = fp.refine_candidates(g, cand, refiner, device)
        with_ref = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
        no_ref = fm.pq_counts(fp.label_map_to_rles(cand), passes, h, w)
        got = [float(with_ref[0]), *map(int, with_ref[1:])]
        if k <= 3:
            want = STORED[name]
            assert all(abs(a - b) <= 1 for a, b in zip(got[1:], want[1:])), \
                f"STOP: counts {got[1:]} do not reproduce the stored evaluation {want[1:]} on {name}"
            if k == 3:
                print("integrity check passed: the first 3 images reproduce the stored GPU evaluation", flush=True)
        results.append({"name": name, "fold": FOLD, "n_pass": len(passes), "with_refiner": got,
                        "without_refiner": [float(no_ref[0]), *map(int, no_ref[1:])], "seconds": round(time.time() - t_img)})
        key = name.rsplit(".", 1)[0]
        store[f"{key}__prob"] = (prob * 255).round().astype(np.uint8)
        store[f"{key}__cand"] = cand.astype(np.int16)
        store[f"{key}__lab"] = lab.astype(np.int16)
        if k % 5 == 0 or k == len(val):
            print(f"{k}/{len(val)} {time.time() - t0:.0f}s last={results[-1]['with_refiner']}", flush=True)
            json.dump({"tag": TAG, "fold": FOLD, "status": "running" if k < len(val) else "done", "results": results},
                      open(f"{OUT}/{TAG}_fold{FOLD}.json", "w"))
        if k % 40 == 0:
            np.savez_compressed(f"{OUT}/{TAG}_preds_fold{FOLD}.npz", **store)
    json.dump({"tag": TAG, "fold": FOLD, "status": "done", "elapsed_s": round(time.time() - t0), "n_images": len(val),
               "results": results}, open(f"{OUT}/{TAG}_fold{FOLD}.json", "w"))
    np.savez_compressed(f"{OUT}/{TAG}_preds_fold{FOLD}.npz", **store)
    state["status"] = "done"


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:                       # noqa: BLE001 -- record any failure, then stop
        state.update(status="error", error="".join(traceback.format_exception_only(type(e), e)).strip())
        raise
    finally:
        state["elapsed_s"] = round(time.time() - t0)
        json.dump(state, open(f"{OUT}/{TAG}_fold{FOLD}_state.json", "w"))
    print("done", state["elapsed_s"], "s", flush=True)
