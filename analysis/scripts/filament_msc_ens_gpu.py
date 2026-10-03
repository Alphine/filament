"""
Solar Filament Segmentation 2026 -- the released 5-model ensemble with multi-scale TTA, Kaggle GPU, INFERENCE ONLY.

Out of fold, averaging the stage-A maps of scales 0.75, 1.0 and 1.25 beat one scale by +0.0117 PQ (CI95
+0.0067..+0.0169; analysis/scripts/analyze_msc_tta.py), but that was measured with single fold models. This
script runs the released ensemble (5 models x 4 flips x 3 scales) and
  MODE=holdout  scores the 50 hold-out images with the one-scale and the three-scale ensemble in the same run
                (official counts with and without the near-limb filter), the gate for the test run;
  MODE=test     processes the 180 test images (no labels exist or are used).
In both modes it stores, per image, the three-scale probability map (uint8), the candidate map and the
refined label map (before the near-limb filter) in the format of the earlier prediction stores
(`{stem}__prob/__cand/__lab`), so the instance features and the scorer can be recomputed on CPU.

Safety (the GPU quota is shared with other research):
  * CUDA is required (REQUIRE_CUDA=1); five fold models and the refiner must be found;
  * fail-fast integrity: MODE=holdout, the one-scale counts of the first three images must reproduce the released
    ensemble's stored counts (+-1 per count); MODE=test, the one-scale pipeline on the first two images must give
    the released number of instances (+-1);
  * non-finite probabilities stop the run; any error is written to the state file and re-raised, no retries;
  * results are written every 5 images (counts) and 10 images (maps); a wall-clock budget (TIME_BUDGET_S, default
    3 h) ends the run cleanly with status "partial"; the outputs of a partial run, attached as a kernel source of a
    new kernel, are continued; after 10 images a projected total above twice the budget stops the run.
Environment: MODE, CHUNK_INDEX, CHUNK_TOTAL, TIME_BUDGET_S, REQUIRE_CUDA.
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
import torch

sys.path.insert(0, ".")
import filament_metric as fm
import filament_pipeline as fp

MODE = os.environ.get("MODE", "holdout")
CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "1"))
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 3 * 3600))
INPUT, OUT = "/kaggle/input", "/kaggle/working"
SCALES = fp.MSC_SCALES
STORED = {}      # holdout: name -> [tp, fp, fn] of the released ensemble (near-limb filter on); test: stem -> released instance count
t0 = time.time()
tag = f"ens_{MODE}_chunk{CHUNK_INDEX}"
state = {"mode": MODE, "chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "status": "starting", "error": None}


def counts_pair(g, passes, lab):
    h, w = g.shape
    c0 = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
    c1 = fm.pq_counts(fp.label_map_to_rles(fp.drop_near_limb(lab, g)), passes, h, w)
    return {"nolimb": [float(c0[0]), *map(int, c0[1:])], "limb": [float(c1[0]), *map(int, c1[1:])]}


def pipeline_labels(g, prob, refiner, device):
    cand = fp.extract_candidates(prob)
    return cand, fp.refine_candidates(g, cand, refiner, device)


def main():
    if os.environ.get("REQUIRE_CUDA", "1") == "1":
        assert torch.cuda.is_available(), "STOP: no CUDA device"
    device = fp.default_device()
    weights = sorted(glob.glob(f"{INPUT}/**/unet_fold[1-5].pt", recursive=True))
    assert len(weights) == 5, weights
    models = fp.load_stage_a(weights, device)
    refiner = fp.load_refiner(next(p for p in glob.glob(f"{INPUT}/**/refiner_best.pt", recursive=True)), device)
    ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
    train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
    test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
    per_file, _ = fm.load_annotations(ann_json, train_dir)
    if MODE == "holdout":
        names = [n for k, n in enumerate(fm.holdout_names(list(per_file))) if k % CHUNK_TOTAL == CHUNK_INDEX]
        folder = train_dir
    else:
        files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpeg", ".jpg")))
        names = [f for k, f in enumerate(files) if k % CHUNK_TOTAL == CHUNK_INDEX]
        folder = test_dir
    print(f"device {device}; mode {MODE}; chunk {CHUNK_INDEX}/{CHUNK_TOTAL}: {len(names)} images; scales {SCALES}", flush=True)

    res, store = {"images": []}, {}
    pj = glob.glob(f"{INPUT}/**/{tag}.json", recursive=True)
    pz = glob.glob(f"{INPUT}/**/{tag}.npz", recursive=True)
    if pj and pz:                                   # continue a cleanly ended partial run of this chunk
        prev = json.load(open(pj[0]))
        if all(r["name"] in names for r in prev["images"]):
            res["images"] = prev["images"]
            store = dict(np.load(pz[0]))
            print(f"resuming: {len(res['images'])} images already processed", flush=True)
    have = {r["name"] for r in res["images"]}
    todo = [n for n in names if n not in have]
    checked, n_new, t_start = bool(have), 0, None

    def save(final=False):
        json.dump({**res, "status": "done" if final else "running", "mode": MODE, "chunk_index": CHUNK_INDEX,
                   "chunk_total": CHUNK_TOTAL, "scales": SCALES}, open(f"{OUT}/{tag}.json", "w"))
        np.savez_compressed(f"{OUT}/{tag}.npz", **store)

    for name in todo:
        if time.time() - t0 > TIME_BUDGET_S:
            state["status"] = "partial"
            save()
            print(f"time budget reached after {len(res['images'])} images", flush=True)
            return
        t_start = t_start or time.time()
        g = fp.read_gray(os.path.join(folder, name))
        probs = {s: fp.predict_scaled(models, g, s, device) for s in SCALES}
        assert all(np.isfinite(p).all() for p in probs.values()), f"non-finite probabilities in {name}"
        prob3 = np.mean(list(probs.values()), axis=0)
        cand3, lab3 = pipeline_labels(g, prob3, refiner, device)
        stem = name.rsplit(".", 1)[0]
        store[f"{stem}__prob"] = (prob3 * 255).round().astype(np.uint8)
        store[f"{stem}__cand"] = cand3.astype(np.int16)
        store[f"{stem}__lab"] = lab3.astype(np.int16)
        rec = {"name": name, "n_cand3": int(cand3.max())}
        if MODE == "holdout":
            _, lab1 = pipeline_labels(g, probs[1.0], refiner, device)
            rec["s100"], rec["s075_100_125"] = counts_pair(g, per_file[name], lab1), counts_pair(g, per_file[name], lab3)
            if not checked and len(res["images"]) + 1 == min(3, len(todo)) and STORED:
                for r in res["images"] + [rec]:
                    got, ref = r["s100"]["limb"][1:], STORED[r["name"]]
                    assert all(abs(a - b) <= 1 for a, b in zip(got, ref)), \
                        f"STOP: one-scale counts {got} do not reproduce the released ensemble {ref} on {r['name']}"
                checked = True
                print("integrity check passed: one-scale ensemble counts reproduce the released ensemble", flush=True)
        else:
            if not checked and len(res["images"]) < 2 and STORED:
                n1 = len(fp.label_map_to_rles(fp.drop_near_limb(pipeline_labels(g, probs[1.0], refiner, device)[1], g)))
                assert abs(n1 - STORED[stem]) <= 1, f"STOP: one-scale pipeline gives {n1} instances on {stem}, released {STORED[stem]}"
                rec["n_s100"] = n1
                if len(res["images"]) == 1:
                    checked = True
                    print("integrity check passed: one-scale instance counts reproduce the released test predictions", flush=True)
        res["images"].append(rec)
        n_new += 1
        if n_new == 10:                               # do not burn the quota on a run that cannot finish
            rate = (time.time() - t_start) / n_new
            print(f"{rate:.1f} s per image; projected {rate * len(todo) / 3600:.2f} h for {len(todo)} images", flush=True)
            assert rate * len(todo) <= 2 * TIME_BUDGET_S, f"STOP: projected {rate * len(todo) / 3600:.1f} h exceeds twice the time budget"
        if len(res["images"]) % 5 == 0:
            print(f"{len(res['images'])}/{len(names)} {time.time() - t0:.0f}s", flush=True)
            json.dump({**res, "status": "running", "mode": MODE, "chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "scales": SCALES},
                      open(f"{OUT}/{tag}.json", "w"))
        if len(res["images"]) % 10 == 0:
            save()
    save(final=True)
    state["status"] = "done"


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:                       # noqa: BLE001 -- record any failure, then stop
        state.update(status="error", error="".join(traceback.format_exception_only(type(e), e)).strip())
        raise
    finally:
        state["elapsed_s"] = round(time.time() - t0)
        json.dump(state, open(f"{OUT}/{tag}_state.json", "w"))
    print("done", state["status"], state["elapsed_s"], "s", flush=True)
