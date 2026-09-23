"""
Solar Filament Segmentation 2026 -- fragment-merging study, CPU-ONLY, NO TRAINING.

Motivation: the distribution/fragmentation analysis (filament_distributions_cpu.py)
found 4.4% of ground-truth filaments on the hold-out split across >= 2 predictions.
Each split filament costs roughly 1 FN + 2 FP although the filament was found, so
merging fragments could be worth up to ~+0.036 PQ (crude bound). This kernel measures
it properly, in two steps:

 1. ORACLE upper bound (uses ground truth, NOT a usable method): greedy merges of
    nearby instance pairs (within 60 px), each merge accepted only if it improves the
    image's contribution to the pooled PQ (linearised as S - Q*(TP + FP/2 + FN/2),
    Q = 0.4325). Two variants: (a) union of the refined masks; (b) merge the stage-A
    candidates and let the crop refiner redraw the merged instance.
 2. REALISTIC rules (no ground truth): merge every pair of instances whose masks
    come within d pixels, d in D_GRID, three variants:
      union        -- relabel refined masks of a group to one id
      union_close  -- union + morphological closing (radius d) to fill the gap
      cand_refine  -- merge stage-A candidates, then run the crop refiner on each group
    Per-image counts are stored so the choice of d can be cross-validated locally.

EXACT final pipeline (stage-A 5-fold + TTA, candidates thr 0.4 / area 400-40000 /
mean p >= 0.7, crop refiner), checkpoints loaded via kernel_sources (the original
training kernels), not retrained. Hold-out: seed 1234, 50 images, split into
CHUNK_TOTAL chunks. Per chunk it writes merge_chunk{i}.json (counts) and
preds_chunk{i}.npz (uint8 probability map, candidate and final label maps) so later
merge rules can be tested without re-running inference.

Integrity reference: pooled base TP/FP/FN over all chunks = 400/239/211 (PQ 0.4325).
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

import cv2
import numpy as np

sys.path.insert(0, ".")
import filament_pipeline as fp
import filament_metric as fm

CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.environ.get("CHUNK_TOTAL", "4"))
D_GRID = (5, 10, 20, 30, 45, 60)
ORACLE_D = 60
Q = 0.4325

INPUT = "/kaggle/input"
folds = sorted(glob.glob(os.path.join(INPUT, "**", "unet_fold*.pt"), recursive=True))
refiner_path = next(p for p in glob.glob(os.path.join(INPUT, "**", "refiner_best.pt"), recursive=True))
ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
device = fp.default_device()
stage_a = fp.load_stage_a(folds, device)
refiner = fp.load_refiner(refiner_path, device)
per_file, dims = fm.load_annotations(ann_json, train_dir)
names = fm.holdout_names(list(per_file))
names = [n for k, n in enumerate(names) if k % CHUNK_TOTAL == CHUNK_INDEX]
print(f"chunk {CHUNK_INDEX}/{CHUNK_TOTAL}: {len(names)} images", flush=True)


def counts(lab, passes, h, w):
    s, tp, fpn, fnn = fm.pq_counts(fp.label_map_to_rles(lab), passes, h, w)
    return [float(s), int(tp), int(fpn), int(fnn)]


def objective(c):
    return c[0] - Q * (c[1] + 0.5 * c[2] + 0.5 * c[3])


def adjacency(lab, d):
    """Pairs (i, j), i < j, of instances whose masks come within d pixels."""
    h, w = lab.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    edges = set()
    for i in range(1, int(lab.max()) + 1):
        box = fp._bbox(lab == i)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        y0, y1, x0, x1 = max(0, y0 - d), min(h, y1 + d), max(0, x0 - d), min(w, x1 + d)
        crop = lab[y0:y1, x0:x1]
        dil = cv2.dilate((crop == i).astype(np.uint8), k).astype(bool)
        for j in np.unique(crop[dil]):
            j = int(j)
            if j and j != i:
                edges.add((min(i, j), max(i, j)))
    return edges


def groups_from_edges(n, edges):
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in range(1, n + 1):
        groups.setdefault(find(i), []).append(i)
    return groups


def relabel(lab, groups):
    lut = np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    for root, members in groups.items():
        for m in members:
            lut[m] = root
    return lut[lab]


def close_groups(lab, groups, d):
    out = lab.copy()
    h, w = lab.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    for root, members in groups.items():
        if len(members) < 2:
            continue
        box = fp._bbox(out == root)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        mg = 2 * d + 2  # margin > dilation radius so erosion never sees the crop border
        y0, y1, x0, x1 = max(0, y0 - mg), min(h, y1 + mg), max(0, x0 - mg), min(w, x1 + mg)
        crop = out[y0:y1, x0:x1]
        m = np.pad((crop == root).astype(np.uint8), mg)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)[mg:-mg, mg:-mg].astype(bool)
        crop[closed & (crop == 0)] = root
    return out


def refine_group(gray, cand, members):
    """Refiner output (bool mask) for the union of the given candidate ids."""
    single = np.isin(cand, members).astype(np.int32)
    return fp.refine_candidates(gray, single, refiner, device) == 1


results = []
store = {}
t0 = time.time()
for k, name in enumerate(names, 1):
    t_img = time.time()
    g = fp.read_gray(os.path.join(train_dir, name))
    h, w = g.shape
    passes = per_file[name]
    prob = fp.predict_ensemble(stage_a, g, device)
    cand = fp.extract_candidates(prob)
    lab = fp.refine_candidates(g, cand, refiner, device)
    n = int(cand.max())
    base = counts(lab, passes, h, w)
    rec = {"name": name, "n_pass": len(passes), "n_inst": n, "base": base,
           "union": {}, "union_close": {}, "cand_refine": {}, "n_groups_merged": {}}

    # ---- realistic distance rules
    for d in D_GRID:
        gl = groups_from_edges(n, adjacency(lab, d))
        merged = {r: m for r, m in gl.items() if len(m) > 1}
        rec["n_groups_merged"][str(d)] = len(merged)
        if not merged:
            rec["union"][str(d)] = rec["union_close"][str(d)] = base
        else:
            u = relabel(lab, gl)
            rec["union"][str(d)] = counts(u, passes, h, w)
            rec["union_close"][str(d)] = counts(close_groups(u, gl, d), passes, h, w)
        gc = groups_from_edges(n, adjacency(cand, d))
        if all(len(m) == 1 for m in gc.values()):
            rec["cand_refine"][str(d)] = base
        else:
            rec["cand_refine"][str(d)] = counts(fp.refine_candidates(g, relabel(cand, gc), refiner, device), passes, h, w)

    # ---- oracle A: greedy union of refined masks (uses ground truth)
    cur, cur_c = lab.copy(), base
    pairs = adjacency(lab, ORACLE_D)
    merges_a = 0
    while True:
        best = None
        for a, b in pairs:
            if not (cur == a).any() or not (cur == b).any():
                continue
            trial = cur.copy()
            trial[trial == b] = a
            c = counts(trial, passes, h, w)
            if objective(c) > objective(cur_c) + 1e-9 and (best is None or objective(c) > objective(best[1])):
                best = (trial, c)
        if best is None:
            break
        cur, cur_c = best
        merges_a += 1
    rec["oracle_union"] = {"counts": cur_c, "merges": merges_a}

    # ---- oracle B: greedy candidate merge + refiner redraw (uses ground truth)
    cur_lab, cur_cand, cur_c = lab.copy(), cand.copy(), base
    pairs = adjacency(cand, ORACLE_D)
    merges_b = 0
    while True:
        best = None
        for a, b in pairs:
            if not (cur_cand == a).any() or not (cur_cand == b).any():
                continue
            mask = refine_group(g, cur_cand, [a, b])
            tl = cur_lab.copy()
            tl[(tl == a) | (tl == b)] = 0
            tl[mask & (tl == 0)] = a
            c = counts(tl, passes, h, w)
            if objective(c) > objective(cur_c) + 1e-9 and (best is None or objective(c) > objective(best[2])):
                tc = cur_cand.copy()
                tc[tc == b] = a
                best = (tl, tc, c)
        if best is None:
            break
        cur_lab, cur_cand, cur_c = best
        merges_b += 1
    rec["oracle_refine"] = {"counts": cur_c, "merges": merges_b}

    rec["seconds"] = round(time.time() - t_img)
    results.append(rec)
    key = name.rsplit(".", 1)[0]
    store[f"{key}__prob"] = (prob * 255).round().astype(np.uint8)
    store[f"{key}__cand"] = cand.astype(np.int16)
    store[f"{key}__lab"] = lab.astype(np.int16)
    print(f"{k}/{len(names)} {name} base {base} | oracle_union {rec['oracle_union']['counts']} ({merges_a} merges)"
          f" | oracle_refine {rec['oracle_refine']['counts']} ({merges_b} merges) | {rec['seconds']}s", flush=True)
    json.dump({"chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "status": "running",
               "d_grid": D_GRID, "results": results},
              open(f"/kaggle/working/merge_chunk{CHUNK_INDEX}.json", "w"))

json.dump({"chunk_index": CHUNK_INDEX, "chunk_total": CHUNK_TOTAL, "status": "done",
           "elapsed_s": round(time.time() - t0), "d_grid": D_GRID, "results": results},
          open(f"/kaggle/working/merge_chunk{CHUNK_INDEX}.json", "w"))
np.savez_compressed(f"/kaggle/working/preds_chunk{CHUNK_INDEX}.npz", **store)
print("done", flush=True)
