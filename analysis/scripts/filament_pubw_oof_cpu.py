"""
Public third-party stage-A weights, scored out of fold with OUR post-processing. CPU-ONLY, no network.

Weights: Kaggle dataset dariushafshar/solar-filament-unet-weights (CC BY 4.0; 5 x U-Net ResNet34 + 5 x U-Net
EfficientNetV2-S, trained on the competition data only; folds.csv gives the fold each image was held out in).
Their card reports OOF PQ 0.4502 (707 images) and public 0.39 with a plain keep rule; our released pipeline
scores 0.4089 OOF and 0.37. This kernel measures whether our near-limb filter and a refit instance scorer add
to their maps. Nothing here is a submission; using these weights at all is the repository owner's decision.

One kernel per fold k (env FOLD): the two fold-k models predict the fold-k training images at full
resolution (inference exactly as in their notebook, TTA views from env TTA), the two uint8 maps are averaged.
Instances = 8-connected regions of p > 0.35 under a LOOSE rule (peak > 0.5, area >= 100, mean p >= 0.5), a
superset of their keep rule (peak > 0.99, area >= 200, mean p >= 0.85; flag `strict`). Per instance: the 36
scorer features (filament_scorer.image_instances, cand = lab since there is no refiner) and its IoU per
annotator pass; per image: official counts of the loose set (asserted against filament_metric.pq_counts).
Then, time permitting, the same two models predict the 180 test images (sum of the two uint8 maps stored).
Env PILOT=1: two images only, prints seconds per forward pass.
"""

import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pycocotools.mask as mu
from scipy import ndimage as ndi

import filament_metric as fm
import filament_pipeline as fp
import filament_scorer as feats

INPUT, OUT = "/kaggle/input", "/kaggle/working"
FOLD, TTA, PILOT = int(os.environ["FOLD"]), int(os.environ.get("TTA", "4")), os.environ.get("PILOT") == "1"
BUDGET_S = 11.0 * 3600
T0 = time.time()
H = W = 2048
DISK_R = 950
T_LO = 0.35
LOOSE = dict(t_hi=0.5, min_area=100, min_meanp=0.5)
STRICT = dict(t_hi=0.99, min_area=200, min_meanp=0.85)


def find_file(name):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True))
    assert len(hits) == 1, (name, hits[:3])
    return hits[0]


RECEIPT_PATH = find_file("solar_unet_receipt.json")
WDIR = os.path.dirname(RECEIPT_PATH)
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                    find_file("segmentation_models_pytorch-0.5.0-py3-none-any.whl")], check=True)
    import importlib
    importlib.invalidate_caches()
    import segmentation_models_pytorch as smp
import torch

torch.set_num_threads(os.cpu_count())
print("torch", torch.__version__, "smp", smp.__version__, "cpus", os.cpu_count(), "fold", FOLD, "tta", TTA, flush=True)
RECEIPT = json.load(open(RECEIPT_PATH))
MODELS = [m for m in RECEIPT["ensemble"]["models"] if m["fold"] == FOLD]
assert len(MODELS) == 2 and {m["family"] for m in MODELS} == {"resnet34", "efficientnetv2_s"}, MODELS

YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2
D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(m):
    path = os.path.join(WDIR, m["file"])
    assert sha256(path) == m["sha256"], m["file"]
    model = smp.Unet(encoder_name=m["encoder"], encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(memory_format=torch.channels_last).eval()


@torch.inference_mode()
def predict(model, img_u8):
    """TTA-mean probability rounded to uint8 (p x 255): their notebook's `predict`, one view per forward pass."""
    v = img_u8[DISK].astype(np.float32) / 255.0
    mean, std = float(v.mean()), max(float(v.std()), 1e-3)
    z = ((torch.from_numpy(img_u8.copy()).float() / 255.0 - mean) / std)[None, None]
    acc = torch.zeros((H, W))
    for k, flip in D4[:TTA]:
        x = torch.flip(z, dims=[3]) if flip else z
        x = torch.rot90(x, k, dims=[2, 3]) if k else x
        p = torch.sigmoid(model(x.contiguous(memory_format=torch.channels_last)).float())
        p = torch.rot90(p, -k, dims=[2, 3]) if k else p
        acc += (torch.flip(p, dims=[3]) if flip else p)[0, 0]
    return (acc / TTA * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()


def u8(t):
    return int(round(float(t) * 255))


def label_instances(p_u8):
    """Loose-rule instances of the averaged map, relabelled 1..n; -> (label map, strict flags)."""
    lab, n = ndi.label(p_u8 > u8(T_LO), structure=np.ones((3, 3), bool))
    lv, pv = lab.ravel(), p_u8.ravel()
    nz = np.flatnonzero(lv)
    area = np.bincount(lv[nz], minlength=n + 1)
    psum = np.bincount(lv[nz], weights=pv[nz], minlength=n + 1)
    pmax = np.zeros(n + 1, np.uint8)
    np.maximum.at(pmax, lv[nz], pv[nz])
    meanp = psum / np.maximum(area, 1) / 255.0

    def ok(rule):
        return (pmax > u8(rule["t_hi"])) & (area >= rule["min_area"]) & (meanp >= rule["min_meanp"])

    loose, strict = ok(LOOSE), ok(STRICT)
    loose[0] = False
    assert not (strict[1:] & ~loose[1:]).any()
    ids = np.flatnonzero(loose)
    remap = np.zeros(n + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[lab], [bool(strict[i]) for i in ids]


def gt_rles(anns, h, w):
    out = []
    for a in anns:
        if a.get("segmentation"):
            r = mu.frPyObjects(a["segmentation"], h, w)
            out.append(mu.merge(r) if isinstance(r, list) else r)
    return out


ann_json = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith(".json") and "nnotation" in f)
train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "train_images")
test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
per_file, dims = fm.load_annotations(ann_json, train_dir)
folds = pd.read_csv(find_file("folds.csv"))
assert len(folds) == 707 and set(folds.stem + ".jpeg") == set(per_file), (len(folds), len(per_file))
stems = sorted(folds.stem[folds.fold == FOLD])
if PILOT:
    stems = stems[:2]
models = [load_model(m) for m in MODELS]
print(f"{len(stems)} out-of-fold images, models {[m['file'] for m in MODELS]}", flush=True)

results, maps = [], {}
for k, stem in enumerate(stems, 1):
    name = stem + ".jpeg"
    g = fp.read_gray(os.path.join(train_dir, name))
    assert g.shape == (H, W), (name, g.shape)
    t = time.time()
    ps = [predict(m, g) for m in models]
    if k == 1:
        print(f"first image: {(time.time() - t) / (2 * TTA):.1f} s per forward pass", flush=True)
    avg = np.clip(np.round((ps[0].astype(np.float32) + ps[1]) / 2), 0, 255).astype(np.uint8)
    maps[stem] = avg
    lab, strict = label_instances(avg)
    inst, disk = feats.image_instances(g, lab, lab, avg.astype(np.float32) / 255.0)
    pred = [{"size": [H, W], "counts": c.encode("ascii")} for c in fp.label_map_to_rles(lab)]
    assert len(pred) == len(inst) == len(strict)
    for r, s in zip(inst, strict):
        r["iou"], r["strict"] = [], s
    counts = np.zeros(4)
    for pid, anns in sorted(per_file[name].items(), key=lambda kv: str(kv[0])):
        gt = gt_rles(anns, H, W)
        if gt and pred:
            iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)), np.float64)
            hit = iou > 0.5
            counts += [iou[hit].sum(), hit.sum(), (hit.sum(1) == 0).sum(), (hit.sum(0) == 0).sum()]
            for r, row in zip(inst, iou):
                r["iou"].append(float(row.max()))
        else:
            counts += [0, 0, len(pred) if not gt else 0, len(gt) if not pred else 0]
            for r in inst:
                r["iou"].append(0.0)
    ref = fm.pq_counts([p["counts"].decode("ascii") for p in pred], per_file[name], H, W)
    assert abs(ref[0] - counts[0]) < 1e-6 and tuple(ref[1:]) == tuple(int(c) for c in counts[1:]), (name, ref, counts)
    results.append({"name": name, "set": "oof", "fold": FOLD, "n_pass": len(per_file[name]),
                    "counts": [float(counts[0]), *map(int, counts[1:])], "disk": [round(v, 2) for v in disk], "instances": inst})
    if k % 10 == 0 or k == len(stems):
        print(f"{k}/{len(stems)} {time.time() - T0:.0f}s instances {sum(len(r['instances']) for r in results)}", flush=True)
        json.dump({"status": "running" if k < len(stems) else "done", "tta": TTA, "results": results},
                  open(f"{OUT}/pubw_instances_fold{FOLD}.json", "w"))
np.savez_compressed(f"{OUT}/pubw_oofmaps_fold{FOLD}.npz", **maps)
del maps

# the same two models on the test images (sum of the two uint8 maps; the five folds are summed later)
tstems = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(test_dir, "*.jpeg")))
if PILOT:
    tstems = tstems[:1]
tmaps, status = {}, "done"
for k, stem in enumerate(tstems, 1):
    if time.time() - T0 > BUDGET_S:
        status = f"stopped at {k - 1}/{len(tstems)} (time budget)"
        break
    g = fp.read_gray(os.path.join(test_dir, stem + ".jpeg"))
    tmaps[stem] = predict(models[0], g).astype(np.uint16) + predict(models[1], g)
    if k % 20 == 0:
        print(f"test {k}/{len(tstems)} {time.time() - T0:.0f}s", flush=True)
np.savez_compressed(f"{OUT}/pubw_testsum_fold{FOLD}.npz", **tmaps)
json.dump({"status": status, "n_test": len(tmaps), "tta": TTA, "fold": FOLD, "seconds": round(time.time() - T0)},
          open(f"{OUT}/pubw_status_fold{FOLD}.json", "w"))
print("test:", status, "| total", round(time.time() - T0), "s", flush=True)
