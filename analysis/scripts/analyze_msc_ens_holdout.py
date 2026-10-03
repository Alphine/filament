"""Hold-out gate of the multi-scale ensemble (analysis/scripts/filament_msc_ens_gpu.py, MODE=holdout).

The released 5-model ensemble with one scale and with scales 0.75/1.0/1.25, on the 50 hold-out images,
paired in the same run (official counts, near-limb filter on and off).

Rule fixed BEFORE the run (stated when the plan was proposed): the three-scale ensemble must not fall below the
one-scale ensemble on the hold-out (diff >= 0, near-limb filter on); only then is the test set run. With 50 images
the paired SE is about 0.01, so this gate can only catch a clear loss; it cannot confirm the out-of-fold gain.
"""
import json
import sys

import numpy as np

from scorer_common import PQ, paired_ci

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
j = json.load(open('analysis/results/msc_ens/ens_holdout_chunk0.json'))
assert j['status'] == 'done' and len(j['images']) == 50, (j['status'], len(j['images']))
oi = json.load(open('analysis/results/oof_instances/oof_instances.json'))
released = {r['name']: r['counts'] for r in oi['results'] if r['set'] == 'holdout'}
out = {}
for k in ('limb', 'nolimb'):
    a = np.array([r['s075_100_125'][k] for r in j['images']], float)
    b = np.array([r['s100'][k] for r in j['images']], float)
    d, (lo, hi) = paired_ci(a, b, rng)
    print(f'{k:6s}: one scale {PQ(b.sum(0)):.4f} TP/FP/FN {b.sum(0)[1:].astype(int).tolist()} -> three scales {PQ(a.sum(0)):.4f} '
          f'{a.sum(0)[1:].astype(int).tolist()}  diff {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]')
    out[k] = {'one_scale': round(PQ(b.sum(0)), 4), 'three_scales': round(PQ(a.sum(0)), 4), 'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
rel = np.array([released[r['name']] for r in j['images']], float)
mine = np.array([r['s100']['limb'] for r in j['images']], float)
same = int((np.abs(rel[:, 1:] - mine[:, 1:]) < 0.5).all(1).sum())
print(f'integrity: one-scale counts equal the released ensemble on {same}/50 images (PQ {PQ(mine.sum(0)):.4f} vs {PQ(rel.sum(0)):.4f})')
ok = out['limb']['diff'] >= 0
out.update({'integrity_identical': same, 'gate': 'PASS -> run the test set' if ok else 'FAIL -> stop, do not run the test set'})
print('GATE:', out['gate'])
json.dump(out, open('analysis/results/msc_ens_holdout_summary.json', 'w'), indent=1)
