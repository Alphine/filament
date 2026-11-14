"""Probe submission of OUR EfficientNetV2-S stage-A (5-fold ensemble), built with exactly the procedure of gate 2 in
analyze_effv2s.py (fixed before the test maps were scored): the scorer is refit on the near-limb out-of-fold instances of
the 657 pool images (released LightGBM parameters, 36 features), tau = PQ(near-limb out-of-fold) / 2, applied to the
near-limb test instances. Input: analysis/results/effv2s/effv2s_instances.json (out-of-fold) and
effv2s_test_instances.json (filament_effv2s_test_cpu.py). Writes the CSV; nothing is submitted here.
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, 'analysis/scripts')
sys.path.insert(0, '.')
from scorer_common import PQ, counts_after, feature_names, fit, table   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
out_csv = sys.argv[1]
new = json.load(open('analysis/results/effv2s/effv2s_instances.json'))
assert new['status'] == 'done' and new['folds'] == [1, 2, 3, 4, 5]
oof = sorted((r for r in new['results'] if r['set'] == 'oof'), key=lambda r: r['name'])
assert len(oof) == 657
feats = feature_names(oof)
assert len(feats) == 36
Dall = table(oof, feats)
inner = np.array([it['feats']['dist_r'] <= 0.88 for r in oof for it in r['instances']])
base = counts_after(Dall, inner)
tau = PQ(base.sum(0)) / 2
print(f"out-of-fold near-limb PQ {PQ(base.sum(0)):.4f}, tau {tau:.4f}")


def near_limb(rs):
    return [{**r, 'instances': [it for it in r['instances'] if it['feats']['dist_r'] <= 0.88]} for r in rs]


Dn = table(near_limb(oof), feats)
model = fit(Dn)

test = json.load(open('analysis/results/effv2s/effv2s_test_instances.json'))
assert test['status'] == 'done' and test['n_test'] == 180 and len(test['images']) == 180
rows, n_all, n_nl = [], 0, 0
for im in test['images']:
    inst = [it for it in im['instances']]
    n_all += len(inst)
    keep_nl = [it for it in inst if it['feats']['dist_r'] <= 0.88]
    n_nl += len(keep_nl)
    if not keep_nl:
        continue
    X = np.array([[it['feats'][k] for k in feats] for it in keep_nl], float)
    s = model.predict(X)
    kept = [it['rle'] for it, v in zip(keep_nl, s) if v >= tau]
    rows += [{'filament_id': f"{im['stem']}_{i}", 'segmentation_rle': c} for i, c in enumerate(kept, 1)]
print(f"test: {len(test['images'])} images, {n_all} loose instances, {n_nl} after near-limb, kept {len(rows)} "
      f"(share {len(rows) / max(n_nl, 1):.1%})")

stems = {im['stem'] for im in test['images']}
ids = [r['filament_id'] for r in rows]
assert len(set(ids)) == len(ids) and {i.rsplit('_', 1)[0] for i in ids} <= stems
try:
    import filament_pipeline as fp
    print('validate:', fp.validate_submission_rows(rows, stems))
except ImportError as e:
    print('validate skipped (no torch/cv2 locally):', e)
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['filament_id', 'segmentation_rle'])
    w.writeheader()
    w.writerows(rows)
print('written', out_csv)
