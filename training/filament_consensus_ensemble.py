"""
Solar Filament Segmentation 2026 -- 5-fold CONSENSUS-target ensemble: hold-out scores and
test submission (Kaggle GPU, inference only).

Weights: best_consensus.pt (fold 1, from the pilot kernel) and best_consensus_fold{2..5}.pt
(from the two consensus training kernels), attached via kernel_sources. The pipeline is the
released one with only stage-A swapped: 5 models x 4-flip TTA, production candidate
thresholds, the released crop refiner.

Outputs:
  consensus_ensemble_holdout.json  per-image official counts on the 50-image hold-out
                                   (compared locally with the released ensemble's per-image
                                   counts, paired)
  submission.csv                   180 test images, validated with validate_submission_rows
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

import pandas as pd

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()
weights = {}
for f in range(1, 6):
    hits = glob.glob(f"{INPUT}/**/best_consensus_fold{f}.pt", recursive=True)
    if not hits and f == 1:
        hits = glob.glob(f"{INPUT}/**/best_consensus.pt", recursive=True)
    assert hits, f"consensus weights for fold {f} not found"
    weights[f] = hits[0]
print("weights:", weights, flush=True)
refiner_path = next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
device = fp.default_device()
pipe = fp.Pipeline(fp.load_stage_a([weights[f] for f in range(1, 6)], device), fp.load_refiner(refiner_path, device), device)

# ---- hold-out
per_file, dims = fm.load_annotations(ann_json, train_dir)
names = fm.holdout_names(list(per_file))
rows = []
for k, n in enumerate(names, 1):
    g = fp.read_gray(os.path.join(train_dir, n))
    h, w = g.shape
    c = fm.pq_counts(fp.label_map_to_rles(pipe.segment(g)), per_file[n], h, w)
    rows.append({"name": n, "n_pass": len(per_file[n]), "counts": [float(c[0]), *map(int, c[1:])]})
    if k % 10 == 0:
        print(f"hold-out {k}/{len(names)} {time.time() - t0:.0f}s", flush=True)
tot = [sum(r["counts"][i] for r in rows) for i in range(4)]
pq = fm.pq(*tot)
print(f"hold-out PQ consensus ensemble {pq:.4f} TP/FP/FN {tot[1:]}", flush=True)
json.dump({"weights": weights, "pq": pq, "tp_fp_fn": tot[1:], "per_image": rows},
          open(f"{OUT}/consensus_ensemble_holdout.json", "w"), indent=1)

# ---- test submission
test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg")))
sub = []
for k, fname in enumerate(test_files, 1):
    stem = fname.rsplit(".", 1)[0]
    rles = fp.label_map_to_rles(pipe.segment(fp.read_gray(os.path.join(test_dir, fname))))
    sub += [{"filament_id": f"{stem}_{i}", "segmentation_rle": r} for i, r in enumerate(rles, 1)]
    if k % 20 == 0:
        print(f"test {k}/{len(test_files)} {time.time() - t0:.0f}s rows={len(sub)}", flush=True)
check = fp.validate_submission_rows(sub, {f.rsplit(".", 1)[0] for f in test_files})
print("submission check:", check, flush=True)
pd.DataFrame(sub, columns=["filament_id", "segmentation_rle"]).to_csv(f"{OUT}/submission.csv", index=False)
json.dump({"n_test_images": len(test_files), "n_rows": len(sub), "check": check,
           "seconds": round(time.time() - t0)}, open(f"{OUT}/submission_meta.json", "w"), indent=1, default=str)
print("done", flush=True)
