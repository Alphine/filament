"""
Final submission from the public repository and the public weights, all 180 test images. CPU-ONLY Kaggle kernel
(one of N_CHUNKS; images stems[CHUNK::N_CHUNKS]). Inputs: the competition data and the weights dataset
abukayyisahbasyid/solar-filament-segmentation-2026-weights (which also holds the submitted CSV). Internet: on (git
clone of the public repository at a pinned commit, pip install of segmentation-models-pytorch).

What it does
  1. clones https://github.com/Alphine/filament at COMMIT and imports the released code from the clone;
  2. FAIL FAST: sha256 of the five weight files and of the submitted CSV against the repository's SHA256SUMS.txt,
     180 test images, the repository's test suite (chunk 0 only), first image within the time guard;
  3. runs filament_effv2s.Pipeline (the notebook's default path) on this chunk's images and compares every image
     with the submitted rows: number of rows, rows identical as RLE strings, IoU of each submitted mask with its best
     new mask. The submitted maps were computed on a GPU in float16, so a few boundary pixels may differ on CPU;
  4. chunk 0 only: executes the code cells of pipeline.ipynb itself (MAX_TEST_IMAGES = NB_IMAGES) and requires its
     submission.csv rows to equal the rows of step 3 for the same images.
STOP ON ERROR: any exception is written to the result file and re-raised. The result file is rewritten after every
image (partial results survive), and the kernel stops if it would exceed the time budget.
Outputs: effv2s_e2e_chunk<C>.json, submission_chunk<C>.csv.
"""

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

CHUNK, N_CHUNKS = 0, 4                                   # CHUNK is set per kernel by the build step
COMMIT = "1cd216e"                                       # repository commit under test (prefix)
INPUT = os.environ.get("E2E_INPUT", "/kaggle/input")
OUT = os.environ.get("E2E_OUT", "/kaggle/working")
REPO = os.environ.get("E2E_REPO")                        # local smoke test: an existing checkout instead of a clone
N_EXPECT = int(os.environ.get("E2E_N_EXPECT", "180"))
NB_IMAGES = int(os.environ.get("E2E_NB_IMAGES", "2"))
BUDGET_S = 11.0 * 3600
FIRST_IMAGE_MAX_S = 20 * 60
T0 = time.time()
RES_PATH = os.path.join(OUT, f"effv2s_e2e_chunk{CHUNK}.json")
res = {"chunk": CHUNK, "n_chunks": N_CHUNKS, "status": "running", "images": []}


def save():
    json.dump(res, open(RES_PATH + ".tmp", "w"), indent=1)
    os.replace(RES_PATH + ".tmp", RES_PATH)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find(name):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True))
    assert hits, f"{name} not found under {INPUT}"
    return hits[0]


def main():
    global REPO
    if REPO is None:
        REPO = os.path.join(OUT, "repo")
        subprocess.run(["git", "clone", "-q", "--branch", "final", "https://github.com/Alphine/filament.git", REPO], check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "-q", COMMIT], check=True)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    res["commit"] = head
    if os.environ.get("E2E_REPO") is None:
        assert head.startswith(COMMIT), head
    try:
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0", "timm==1.0.26"], check=True)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import numpy as np
    import pycocotools.mask as mu
    import torch

    import filament_effv2s as fe
    import filament_pipeline as fp
    import filament_scorer as fs
    res["versions"] = {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}

    # ---- FAIL FAST: files and checksums
    sums = dict(reversed(l.split()) for l in open("SHA256SUMS.txt").read().splitlines() if l.strip())
    paths = [find(f"effv2s_fold{k}.pt") for k in range(1, 6)]
    for p in paths:
        assert sha256(p) == sums[os.path.basename(p)], f"checksum mismatch: {p}"
    sub_csv = find("submission_effv2s_probe.csv")
    assert sha256(sub_csv) == sums["submission_effv2s_probe.csv"], "submitted CSV checksum mismatch"
    test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
    stems = sorted(f[:-5] for f in os.listdir(test_dir) if f.endswith(".jpeg"))
    assert len(stems) == N_EXPECT, len(stems)
    mine = stems[CHUNK::N_CHUNKS]
    sub = {}
    for r in csv.DictReader(open(sub_csv)):
        sub.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append(r["segmentation_rle"])
    assert N_EXPECT != 180 or set(sub) <= set(stems), "submitted CSV names images that are not in the test folder"
    print(f"fail-fast OK: checksums, {len(stems)} test images, chunk {CHUNK}: {len(mine)} images, commit {head[:7]}", flush=True)
    if CHUNK == 0:
        t = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True)
        res["pytest"] = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-300:]
        print("pytest:", res["pytest"], flush=True)
        assert t.returncode == 0, t.stdout[-2000:] + t.stderr[-2000:]
    save()

    device = fp.default_device()
    res["device"] = device.type
    pipe = fe.Pipeline(fe.load_stage_a(paths, device), device, scorer=fs.InstanceScorer.load(name="effv2s_scorer"))
    enc = lambda xs: [{"size": [2048, 2048], "counts": c.encode("ascii")} for c in xs]   # noqa: E731
    rows = []
    for k, stem in enumerate(mine, 1):
        t0 = time.time()
        new = fp.label_map_to_rles(pipe.segment(fp.read_gray(os.path.join(test_dir, stem + ".jpeg"))))
        old = sub.get(stem, [])
        iou = np.asarray(mu.iou(enc(old), enc(new), [0] * len(new))).reshape(len(old), len(new)) if old and new else np.zeros((len(old), len(new)))
        best = iou.max(1).tolist() if new else [0.0] * len(old)
        res["images"].append({"stem": stem, "rows_submitted": len(old), "rows_new": len(new),
                              "identical_rows": len(set(old) & set(new)), "same_order": old == new,
                              "min_best_iou": round(min(best), 4) if best else None,
                              "sum_best_iou": round(float(sum(best)), 4), "seconds": round(time.time() - t0)})
        rows += [{"filament_id": f"{stem}_{i}", "segmentation_rle": c} for i, c in enumerate(new, 1)]
        save()
        with open(os.path.join(OUT, f"submission_chunk{CHUNK}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
            w.writeheader()
            w.writerows(rows)
        el = time.time() - T0
        print(f"{k}/{len(mine)} {stem}: {res['images'][-1]} | {el:.0f}s", flush=True)
        if k == 1:
            assert time.time() - t0 < FIRST_IMAGE_MAX_S, "first image too slow: stopping"
        if el + (el / k) * 1.5 > BUDGET_S and k < len(mine):
            raise RuntimeError(f"time budget: stopping after {k}/{len(mine)} images (partial results kept)")
    res["rows_valid"] = fp.validate_submission_rows(rows, set(mine))

    if CHUNK == 0:        # the notebook itself, on its first NB_IMAGES test images
        os.environ["FILAMENT_INPUT"] = INPUT
        os.environ.pop("FILAMENT_WEIGHTS", None)
        nb = json.load(open("pipeline.ipynb", encoding="utf-8"))
        ns = {"__name__": "__notebook__"}
        knob = "MAX_TEST_IMAGES = None"
        n_knob = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            n_knob += src.count(knob)
            src = src.replace(knob, f"MAX_TEST_IMAGES = {NB_IMAGES}")
            src = src.replace('OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."', f"OUT_DIR = {os.path.join(OUT, 'nb')!r}")
            os.makedirs(os.path.join(OUT, "nb"), exist_ok=True)
            exec(compile(src, "pipeline.ipynb", "exec"), ns)
        assert n_knob == 1, n_knob
        nb_rows = list(csv.DictReader(open(os.path.join(OUT, "nb", "submission.csv"))))
        nb_stems = stems[:NB_IMAGES]
        ref = {s: None for s in nb_stems}
        direct = {}
        for r in rows:
            direct.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append((r["filament_id"], r["segmentation_rle"]))
        got = {}
        for r in nb_rows:
            got.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append((r["filament_id"], r["segmentation_rle"]))
        common = [s for s in ref if s in mine]
        res["notebook"] = {"images": nb_stems, "rows": len(nb_rows), "compared_with_direct": common,
                           "equal_to_direct": all(got.get(s, []) == direct.get(s, []) for s in common)}
        print("notebook:", res["notebook"], flush=True)
        assert common and res["notebook"]["equal_to_direct"], "the notebook's rows differ from the direct code path"

    tot = {k: sum(r[k] for r in res["images"]) for k in ("rows_submitted", "rows_new", "identical_rows")}
    res["total"], res["status"], res["seconds"] = tot, "done", round(time.time() - T0)
    save()
    print("DONE", tot, res["seconds"], "s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                # STOP ON ERROR: record and end the kernel
        res["status"], res["error"] = "error", f"{type(e).__name__}: {e}"
        save()
        raise
