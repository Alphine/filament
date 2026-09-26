"""
Solar Filament Segmentation 2026 -- each stage-A fold model ALONE on the 50-image hold-out,
same protocol as the out-of-fold run, CPU-ONLY, NO TRAINING. One kernel per fold (FOLD = 1..5).

Question: why is the hold-out PQ (0.4325, 5-model ensemble) so far above the out-of-fold
single-model PQ (0.388) and the leaderboard (0.36)? Two explanations, separable here:
  (a) the hold-out is an easy / optimistic sample (thresholds were tuned on it and many
      variants were selected on it)  -> single models score well above 0.388 on it;
  (b) the 5-model ensemble is worth ~+0.04 over one model -> single models score about
      0.388 on the hold-out as well.
The comparison with the out-of-fold numbers is cleanest WITHOUT the refiner (the refiner was
trained on the out-of-fold images but never on the hold-out); both columns are stored.

Protocol identical to filament_oof_cpu.py: unet_fold{F}.pt alone with 4-flip TTA,
production candidate thresholds, crop refiner; official counts per image, pooled over passes.
"""

import glob
import json
import os
import subprocess
import sys
import time

try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

FOLD = int(os.environ.get("FOLD", "1"))
INPUT = "/kaggle/input"
fold_path = next(p for p in glob.glob(os.path.join(INPUT, "**", f"unet_fold{FOLD}.pt"), recursive=True))
refiner_path = next(p for p in glob.glob(os.path.join(INPUT, "**", "refiner_best.pt"), recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
device = fp.default_device()
model = fp.load_stage_a([fold_path], device)
refiner = fp.load_refiner(refiner_path, device)
per_file, dims = fm.load_annotations(ann_json, train_dir)
names = fm.holdout_names(list(per_file))
print(f"fold {FOLD} alone on the hold-out ({len(names)} images); model {fold_path}", flush=True)

results = []
t0 = time.time()
for k, name in enumerate(names, 1):
    t_img = time.time()
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    passes = per_file[name]
    prob = fp.predict_ensemble(model, g, device)
    cand = fp.extract_candidates(prob)
    lab = fp.refine_candidates(g, cand, refiner, device)
    with_ref = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
    no_ref = fm.pq_counts(fp.label_map_to_rles(cand), passes, h, w)
    results.append({"name": name, "fold_model": FOLD, "n_pass": len(passes),
                    "with_refiner": [float(with_ref[0]), *map(int, with_ref[1:])],
                    "without_refiner": [float(no_ref[0]), *map(int, no_ref[1:])],
                    "seconds": round(time.time() - t_img)})
    if k % 5 == 0 or k == len(names):
        print(f"{k}/{len(names)} {time.time() - t0:.0f}s", flush=True)
        json.dump({"fold_model": FOLD, "status": "running", "results": results},
                  open(f"/kaggle/working/holdout_single_fold{FOLD}.json", "w"))

json.dump({"fold_model": FOLD, "status": "done", "elapsed_s": round(time.time() - t0), "results": results},
          open(f"/kaggle/working/holdout_single_fold{FOLD}.json", "w"))
print("done", flush=True)
