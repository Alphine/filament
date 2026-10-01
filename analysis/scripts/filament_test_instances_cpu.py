"""
Solar Filament Segmentation 2026 -- released pipeline on the test images with per-instance scorer
features, CPU-ONLY, inference only, CHUNK_INDEX of CHUNK_TOTAL.

The learned instance scorer (analysis/scripts/analyze_instance_scorer.py) needs, for every predicted
test instance, the same features it was fitted on. Earlier test runs stored only the submission rows,
so this kernel re-runs the released pipeline (5-fold stage-A + 4-flip TTA, crop refiner, near-limb
filter) and writes test_instances_chunk{i}.json: per image the RLE and the features of every instance.
The scorer is applied locally; no test labels exist or are used. As in the stored out-of-fold
predictions, the features are computed from the probability map quantised to 1/255 steps.
"""

import csv
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

import numpy as np

import filament_pipeline as fp
import filament_scorer as feats

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()
weights = sorted(glob.glob(f"{INPUT}/**/unet_fold[1-5].pt", recursive=True))
assert len(weights) == 5, weights
refiner_path = next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True))
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
device = fp.default_device()
stage_a, refiner = fp.load_stage_a(weights, device), fp.load_refiner(refiner_path, device)

# rows of the current best submission (same pipeline on a T4), to report how closely this CPU run reproduces it
ref_rows = {}
for p in glob.glob(f"{INPUT}/**/submission_drop@0.88.csv", recursive=True)[:1]:
    with open(p) as f:
        for r in csv.DictReader(f):
            ref_rows.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append(r["segmentation_rle"])

test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg")))
mine = [f for k, f in enumerate(test_files) if k % CHUNK_TOTAL == CHUNK_INDEX]
print(f"chunk {CHUNK_INDEX}/{CHUNK_TOTAL}: {len(mine)} of {len(test_files)} test images on {device}", flush=True)
res = {"chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "n_test": len(test_files), "status": "running", "images": []}
for k, name in enumerate(mine, 1):
    stem = name.rsplit(".", 1)[0]
    g = fp.read_gray(os.path.join(test_dir, name))
    prob = fp.predict_ensemble(stage_a, g, device)
    cand = fp.extract_candidates(prob)
    lab = fp.drop_near_limb(fp.refine_candidates(g, cand, refiner, device), g)
    inst, disk = feats.image_instances(g, lab, cand, (prob * 255).round().astype(np.uint8).astype(np.float32) / 255.0)
    rles = fp.label_map_to_rles(lab)
    assert len(rles) == len(inst), (name, len(rles), len(inst))
    for r, c in zip(inst, rles):
        r["rle"] = c
    res["images"].append({"stem": stem, "shape": list(g.shape), "instances": inst,
                          "same_as_reference": rles == ref_rows.get(stem, []) if ref_rows else None,
                          "n_reference": len(ref_rows.get(stem, [])) if ref_rows else None})
    res["status"] = "running" if k < len(mine) else "done"
    json.dump(res, open(f"{OUT}/test_instances_chunk{CHUNK_INDEX}.json", "w"))
    print(f"{k}/{len(mine)} {stem} n={len(inst)} ref={res['images'][-1]['n_reference']} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
