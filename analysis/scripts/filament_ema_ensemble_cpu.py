"""
Solar Filament Segmentation 2026 -- 5-fold EMA-weights ensemble on the hold-out and the test set,
CPU-ONLY, inference only, CHUNK_INDEX of CHUNK_TOTAL.

The checkpoint-selection study (training/filament_ckpt_selection.py) found that EMA weights beat
the production checkpoint rule out of fold (+0.0144, CI95 +0.009..+0.021). This kernel runs the
released pipeline with only stage-A swapped for the five EMA models (sel_fold{1..5}_ema.pt):
5 models x 4-flip TTA, production candidate thresholds, released crop refiner.
MODE=holdout runs only the hold-out (the submit gate), MODE=test only the test set.
RULE selects the weights (default ema; prod = the production checkpoint rule of the same runs).
Per chunk it writes ema_<mode>_chunk{i}.json: per-image official counts on its hold-out images (compared
locally, paired, with the released ensemble's per-image counts) and the submission rows for its
test images. Submit criterion fixed before the run: hold-out ensemble PQ >= released (paired).
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

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "5"))
MODE = os.environ.get("MODE", "both")  # holdout | test | both
RULE = os.environ.get("RULE", "ema")   # which weights of the checkpoint-selection runs: ema | prod | last | swa
INPUT, OUT = "/kaggle/input", "/kaggle/working"
t0 = time.time()
weights = []
for f in range(1, 6):
    hits = glob.glob(f"{INPUT}/**/sel_fold{f}_{RULE}.pt", recursive=True)
    assert hits, f"{RULE} weights for fold {f} not found"
    weights.append(hits[0])
print("weights:", weights, flush=True)
refiner_path = next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
device = fp.default_device()
pipe = fp.Pipeline(fp.load_stage_a(weights, device), fp.load_refiner(refiner_path, device), device)

per_file, dims = fm.load_annotations(ann_json, train_dir)
hold = [n for k, n in enumerate(fm.holdout_names(list(per_file))) if k % CHUNK_TOTAL == CHUNK_INDEX] if MODE != "test" else []
test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg")))
tests = [f for k, f in enumerate(test_files) if k % CHUNK_TOTAL == CHUNK_INDEX] if MODE != "holdout" else []
print(f"chunk {CHUNK_INDEX}/{CHUNK_TOTAL}: {len(hold)} hold-out, {len(tests)} test images", flush=True)
res = {"rule": RULE, "mode": MODE, "chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "weights": weights, "status": "running",
       "holdout": [], "test_stems": [], "rows": []}


def dump():
    json.dump(res, open(f"{OUT}/{RULE}_{MODE}_chunk{CHUNK_INDEX}.json", "w"))


for n in hold:
    g = fp.read_gray(os.path.join(train_dir, n))
    h, w = g.shape
    c = fm.pq_counts(fp.label_map_to_rles(pipe.segment(g)), per_file[n], h, w)
    res["holdout"].append({"name": n, "counts": [float(c[0]), *map(int, c[1:])]})
    print(f"hold-out {n} {res['holdout'][-1]['counts']} {time.time() - t0:.0f}s", flush=True)
    dump()
for k, fname in enumerate(tests, 1):
    stem = fname.rsplit(".", 1)[0]
    rles = fp.label_map_to_rles(pipe.segment(fp.read_gray(os.path.join(test_dir, fname))))
    res["test_stems"].append(stem)
    res["rows"] += [{"filament_id": f"{stem}_{i}", "segmentation_rle": r} for i, r in enumerate(rles, 1)]
    print(f"test {k}/{len(tests)} {stem}: {len(rles)} instances {time.time() - t0:.0f}s", flush=True)
    dump()
res["status"] = "done"
res["seconds"] = round(time.time() - t0)
dump()
print("done", flush=True)
