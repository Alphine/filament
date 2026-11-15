"""End-to-end check of filament_effv2s.Pipeline against the submitted CSV (the 0.38 submission), from the weights.

  python analysis/scripts/check_effv2s_e2e.py <weights_dir> <test_images_dir> <submitted_csv> [out_json]

Runs the released code path (five effv2s_fold*.pt, four views, loose instances, near-limb filter, stored scorer) on
every image found in <test_images_dir> and compares its rows with the submitted rows of the same image: number of
rows, rows identical as RLE strings, and the IoU of every submitted mask with its best new mask. The submission's maps
were computed on a GPU with float16 autocast; on another device a few boundary pixels can differ.
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import pycocotools.mask as mu

sys.path.insert(0, '.')
import filament_effv2s as fe          # noqa: E402
import filament_pipeline as fp        # noqa: E402
import filament_scorer as fs          # noqa: E402

wdir, idir, sub_csv = sys.argv[1:4]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
paths = sorted(glob.glob(os.path.join(wdir, 'effv2s_fold*.pt')))
assert len(paths) == 5, paths
device = fp.default_device()
pipe = fe.Pipeline(fe.load_stage_a(paths, device), device, scorer=fs.InstanceScorer.load(name='effv2s_scorer'))
sub = {}
for r in csv.DictReader(open(sub_csv)):
    sub.setdefault(r['filament_id'].rsplit('_', 1)[0], []).append(r['segmentation_rle'])
res = []
for f in sorted(glob.glob(os.path.join(idir, '*.jpeg'))):
    stem, t0 = os.path.basename(f)[:-5], time.time()
    new = fp.label_map_to_rles(pipe.segment(fp.read_gray(f)))
    old = sub.get(stem, [])
    enc = lambda xs: [{'size': [2048, 2048], 'counts': c.encode('ascii')} for c in xs]
    iou = np.asarray(mu.iou(enc(old), enc(new), [0] * len(new))).reshape(len(old), len(new)) if old and new else np.zeros((len(old), len(new)))
    best = iou.max(1).tolist() if new else [0.0] * len(old)
    r = {'stem': stem, 'rows_submitted': len(old), 'rows_new': len(new), 'identical_rows': len(set(old) & set(new)),
         'min_best_iou': round(min(best), 4) if best else None, 'mean_best_iou': round(float(np.mean(best)), 4) if best else None,
         'seconds': round(time.time() - t0)}
    res.append(r)
    print(r, flush=True)
tot = {k: sum(r[k] for r in res) for k in ('rows_submitted', 'rows_new', 'identical_rows')}
print('device', device.type, '| total', tot, flush=True)
if out_json:
    json.dump({'device': device.type, 'total': tot, 'images': res}, open(out_json, 'w'), indent=1)
