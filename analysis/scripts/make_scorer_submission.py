"""Apply the learned instance scorer to the test predictions (filament_test_instances_cpu.py).

Loads the released scorer (models/, fitted on all out-of-fold instances by analyze_instance_scorer.py;
tau = out-of-fold PQ / 2), drops every test instance whose predicted matched IoU is below tau and
writes the submission CSV to the directory given as the first argument. Nothing is submitted here.

An optional second argument overrides tau (e.g. 0.15, the balanced Dice/PQ variant of the report); the file is then
named submission_scorer_tau<value>.csv and the base file is not rewritten.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
import filament_pipeline as fp
import filament_scorer as fs

sys.stdout.reconfigure(encoding='utf-8')
out_dir = sys.argv[1]
summ = json.load(open('analysis/results/instance_scorer_summary.json'))
assert summ['verdict'].startswith('ADOPT'), summ['verdict']
tau_arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
scorer = fs.InstanceScorer.load(tau=tau_arg)
tau = scorer.tau
images = []
for p in sorted(glob.glob('analysis/results/test_instances/test_instances_chunk*.json')):
    d = json.load(open(p))
    assert d['status'] == 'done', p
    images += d['images']
n_test = d['n_test']
assert len(images) == n_test == len({im['stem'] for im in images}), (len(images), n_test)
rows, base_rows, scores = [], [], []
for im in images:
    inst = im['instances']
    s = scorer.scores(inst)
    scores += s.tolist()
    base_rows += [{'filament_id': f"{im['stem']}_{i}", 'segmentation_rle': r['rle']} for i, r in enumerate(inst, 1)]
    kept = [r['rle'] for r, v in zip(inst, s) if v >= tau]
    rows += [{'filament_id': f"{im['stem']}_{i}", 'segmentation_rle': c} for i, c in enumerate(kept, 1)]
stems = {im['stem'] for im in images}
same = [im['same_as_reference'] for im in images if im['same_as_reference'] is not None]
print(f'{n_test} test images, {len(base_rows)} instances; identical to the T4 reference on {sum(same)}/{len(same)} images')
print(f'tau {tau}: kept {len(rows)}, dropped {len(base_rows) - len(rows)} ({1 - len(rows) / len(base_rows):.1%}); '
      f'score quantiles 10/50/90%: {np.percentile(scores, [10, 50, 90]).round(3).tolist()}')
suffix = '' if tau_arg is None else f'_tau{tau_arg:g}'
outputs = [(f'submission_scorer{suffix}.csv', rows)] + ([('submission_cpu_base.csv', base_rows)] if tau_arg is None else [])
for name, rr in outputs:
    print(name, fp.validate_submission_rows(rr, stems))
    with open(os.path.join(out_dir, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filament_id', 'segmentation_rle'])
        w.writeheader()
        w.writerows(rr)
