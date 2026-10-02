"""
Solar Filament Segmentation 2026 -- end-to-end check of the released code path with the instance scorer,
CPU-ONLY, CHUNK_INDEX of CHUNK_TOTAL.

The scorer's submission was built from per-instance features of an earlier CPU run, not from
`Pipeline.segment` itself. This kernel closes that gap on the Kaggle image (torch 2.10, LightGBM as
installed there):
  0. runs the repository's test suite (tests/) -- including the check that the stored LightGBM model
     scores identically across library versions -- and stops if anything fails;
  1. runs the real `Pipeline(stage_a, refiner, device, scorer=InstanceScorer.load())` on this chunk's
     hold-out images and writes the official counts per image (compared locally with the counts the
     analysis predicted for the same scorer), and on one test image, writing its RLE rows (compared
     locally with the submitted file).
Output: e2e_chunk{i}.json.
"""

import glob
import json
import os
import subprocess
import sys
import time

t0 = time.time()
INPUT, OUT = "/kaggle/input", "/kaggle/working"
CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
res = {"chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "status": "tests", "holdout": [], "test": []}


def dump():
    json.dump(res, open(f"{OUT}/e2e_chunk{CHUNK_INDEX}.json", "w"))


try:
    import segmentation_models_pytorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)
import lightgbm
import torch

res["versions"] = {"lightgbm": lightgbm.__version__, "torch": torch.__version__}
print("versions", res["versions"], flush=True)
t = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True)
print(t.stdout[-3000:], t.stderr[-1000:], flush=True)
res["tests"] = {"returncode": t.returncode, "tail": t.stdout[-600:]}
dump()
assert t.returncode == 0, "the repository test suite failed on the Kaggle image"

sys.path.insert(0, ".")
import filament_metric as fm
import filament_pipeline as fp
import filament_scorer as fs

weights = sorted(glob.glob(f"{INPUT}/**/unet_fold[1-5].pt", recursive=True))
assert len(weights) == 5, weights
refiner_path = next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs_ in os.walk(INPUT) for f in fs_ if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
device = fp.default_device()
pipe = fp.Pipeline(fp.load_stage_a(weights, device), fp.load_refiner(refiner_path, device), device, scorer=fs.InstanceScorer.load())
per_file, dims = fm.load_annotations(ann_json, train_dir)
hold = [n for k, n in enumerate(fm.holdout_names(list(per_file))) if k % CHUNK_TOTAL == CHUNK_INDEX]
test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg")))
test_pick = [test_files[CHUNK_INDEX * (len(test_files) // CHUNK_TOTAL)]]
res["status"] = "running"
print(f"chunk {CHUNK_INDEX}/{CHUNK_TOTAL}: {len(hold)} hold-out images, test image {test_pick}", flush=True)

for n in hold:
    g = fp.read_gray(os.path.join(train_dir, n))
    h, w = g.shape
    c = fm.pq_counts(fp.label_map_to_rles(pipe.segment(g)), per_file[n], h, w)
    res["holdout"].append({"name": n, "counts": [float(c[0]), *map(int, c[1:])]})
    print(f"hold-out {n} {res['holdout'][-1]['counts']} {time.time() - t0:.0f}s", flush=True)
    dump()
for f in test_pick:
    rles = fp.label_map_to_rles(pipe.segment(fp.read_gray(os.path.join(test_dir, f))))
    res["test"].append({"stem": f.rsplit(".", 1)[0], "rles": rles})
    print(f"test {f}: {len(rles)} instances {time.time() - t0:.0f}s", flush=True)
res["status"] = "done"
dump()
print("done", round(time.time() - t0), "s", flush=True)
