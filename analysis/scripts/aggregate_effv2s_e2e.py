"""Aggregate the four Kaggle CPU kernels of filament_effv2s_e2e_cpu.py and compare their rows with the submitted CSV.

  python analysis/scripts/aggregate_effv2s_e2e.py <dir with effv2s_e2e_chunk*.json and submission_chunk*.csv>

Writes analysis/results/effv2s/e2e_kaggle/: the chunk result files, the combined CSV of the CPU re-run
(submission_cpu_rerun.csv, rows in the submission order) and summary.json. Agreement is reported in the metric's own
terms: a submitted instance is "matched" if a re-run instance of the same image has IoU > 0.5 with it (one-to-one by
construction, as in PQ).
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pycocotools.mask as mu

src = sys.argv[1]
SUB = 'analysis/results/effv2s/submission_effv2s_probe.csv'
OUT = 'analysis/results/effv2s/e2e_kaggle'
os.makedirs(OUT, exist_ok=True)


def read(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault(r['filament_id'].rsplit('_', 1)[0], []).append(r['segmentation_rle'])
    return d


chunks = []
for p in sorted(glob.glob(os.path.join(src, '**', 'effv2s_e2e_chunk*.json'), recursive=True)):
    chunks.append(json.load(open(p)))
    shutil.copyfile(p, os.path.join(OUT, os.path.basename(p)))
assert [c['chunk'] for c in chunks] == [0, 1, 2, 3], [c['chunk'] for c in chunks]
assert all(c['status'] == 'done' for c in chunks), [c['status'] for c in chunks]
images = sorted((i for c in chunks for i in c['images']), key=lambda i: i['stem'])
assert len(images) == 180 and len({i['stem'] for i in images}) == 180

new = {}
for p in glob.glob(os.path.join(src, '**', 'submission_chunk*.csv'), recursive=True):
    new.update(read(p))
old = read(SUB)
rows = [(f'{s}_{k}', c) for s in sorted(new) for k, c in enumerate(new[s], 1)]
with open(os.path.join(OUT, 'submission_cpu_rerun.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['filament_id', 'segmentation_rle'])
    w.writerows(rows)

enc = lambda xs: [{'size': [2048, 2048], 'counts': c.encode('ascii')} for c in xs]   # noqa: E731
matched, ious, only_old, only_new, img_same_set, img_same_count = 0, [], 0, 0, 0, 0
for i in images:
    a, b = old.get(i['stem'], []), new.get(i['stem'], [])
    assert len(a) == i['rows_submitted'] and len(b) == i['rows_new'], i['stem']
    m = np.asarray(mu.iou(enc(a), enc(b), [0] * len(b))).reshape(len(a), len(b)) if a and b else np.zeros((len(a), len(b)))
    hit = m > 0.5
    matched += int(hit.sum())
    ious += m[hit].tolist()
    only_old += len(a) - int(hit.any(1).sum())
    only_new += len(b) - int(hit.any(0).sum())
    img_same_count += len(a) == len(b)
    img_same_set += len(a) == len(b) == int(hit.sum())
n_old, n_new = sum(map(len, old.values())), len(rows)
summary = {
    'commit': chunks[0]['commit'], 'versions': chunks[0]['versions'], 'device': chunks[0]['device'],
    'pytest': chunks[0].get('pytest'), 'notebook': chunks[0].get('notebook'),
    'images': 180, 'rows_submitted': n_old, 'rows_rerun': n_new,
    'identical_rle_rows': sum(i['identical_rows'] for i in images),
    'matched_iou_gt_0.5': matched, 'submitted_without_match': only_old, 'rerun_without_match': only_new,
    'matched_iou_mean': round(float(np.mean(ious)), 4), 'matched_iou_min': round(float(np.min(ious)), 4),
    'matched_iou_ge_0.99': int((np.asarray(ious) >= 0.99).sum()),
    'images_same_instance_set': img_same_set, 'images_same_row_count': img_same_count,
    'images_same_rows_and_order': sum(i['same_order'] for i in images),
    'pq_like_agreement': round(float(np.sum(ious)) / (matched + 0.5 * only_old + 0.5 * only_new), 4),
    'seconds_per_image_mean': round(float(np.mean([i['seconds'] for i in images])), 1),
    'kernel_seconds': [c['seconds'] for c in chunks],
    'rerun_csv_sha256': hashlib.sha256(open(os.path.join(OUT, 'submission_cpu_rerun.csv'), 'rb').read()).hexdigest(),
}
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
