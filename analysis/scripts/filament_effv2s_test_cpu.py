"""
Test-image instances of OUR EfficientNetV2-S stage-A (5-fold ensemble), for the probe submission. CPU-ONLY, inference-free.

Input: the five training kernels filament-effv2s-f1..f5 attached as kernel sources (effv2s_maps_fold<F>_test.npz, uint8
probability maps of the 180 test images), and the competition test images.
The five maps of an image are averaged (rounded to uint8), then the LOOSE instance rule of filament_effv2s_instances_cpu.py
(label_instances, copied verbatim into this kernel by the build script) gives the candidates. Per candidate: the 36 scorer
features (filament_scorer.image_instances) and the COCO RLE. The near-limb filter, the scorer and tau are applied locally
(make_effv2s_submission.py) with the same procedure as gate 2 of analyze_effv2s.py.
No test labels exist or are used.
"""

import glob
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi  # noqa: F401  (used by label_instances)

import filament_pipeline as fp
import filament_scorer as feats

INPUT, OUT = "/kaggle/input", "/kaggle/working"
T0 = time.time()
H = W = 2048
T_LO = 0.35
LOOSE = dict(t_hi=0.5, min_area=100, min_meanp=0.5)
STRICT = dict(t_hi=0.99, min_area=200, min_meanp=0.85)
BUDGET_S = 40 * 60


def u8(t):
    return int(round(float(t) * 255))


# @@LABEL_INSTANCES@@ (replaced by the build script with the function of filament_effv2s_instances_cpu.py)

test_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == "test_images")
stems = sorted(f[:-5] for f in os.listdir(test_dir) if f.endswith(".jpeg"))
assert len(stems) == 180, len(stems)

# FAIL FAST: exactly five fold files, one per fold, every test image present, expected format
files = {}
for p in sorted(glob.glob(os.path.join(INPUT, "**", "effv2s_maps_fold*_test.npz"), recursive=True)):
    f = int(os.path.basename(p).split("fold")[1].split("_")[0])
    assert f not in files, ("duplicate fold", p)
    files[f] = p
assert sorted(files) == [1, 2, 3, 4, 5], sorted(files)
zs = [np.load(files[f]) for f in sorted(files)]
for z in zs:
    assert sorted(z.files) == stems, (len(z.files), len(stems))
    m = z[z.files[0]]
    assert m.shape == (H, W) and m.dtype == np.uint8 and m.max() > 128, (m.shape, m.dtype, int(m.max()))
print("fail-fast OK: 5 folds x 180 test maps", flush=True)

images = []
for k, stem in enumerate(stems, 1):
    avg = np.clip(np.round(np.mean([z[stem].astype(np.float32) for z in zs], axis=0)), 0, 255).astype(np.uint8)
    g = fp.read_gray(os.path.join(test_dir, stem + ".jpeg"))
    assert g.shape == (H, W), (stem, g.shape)
    lab, strict = label_instances(avg)
    inst, disk = feats.image_instances(g, lab, lab, avg.astype(np.float32) / 255.0)
    rles = fp.label_map_to_rles(lab)
    assert len(rles) == len(inst) == len(strict), (stem, len(rles), len(inst), len(strict))
    images.append({"stem": stem, "instances": [{"rle": r, "feats": it["feats"], "strict": s}
                                               for r, it, s in zip(rles, inst, strict)]})
    if k % 30 == 0 or k == len(stems):
        print(f"{k}/{len(stems)} {time.time() - T0:.0f}s", flush=True)
    if time.time() - T0 > BUDGET_S:
        raise RuntimeError(f"time budget exceeded at {k}/{len(stems)}")

n_inst = sum(len(im["instances"]) for im in images)
json.dump({"status": "done", "n_test": len(stems), "images": images}, open(f"{OUT}/effv2s_test_instances.json", "w"))
print("done", len(images), "images,", n_inst, "instances,", round(time.time() - T0), "s", flush=True)
