"""EMA versus production-rule weights at the ENSEMBLE level, on the hold-out (informative, not a gate).

Both ensembles come from the same new training runs (training/filament_ckpt_selection.py), so
neither was tuned on the hold-out -- unlike the released ensemble, which the submit gate compared
against. Interpretation fixed before the run:
  * EMA - prod (new runs): the fair method comparison at the ensemble level;
  * prod (new runs) - released: if also negative, the hold-out favours the released models
    (thresholds and earlier choices were tuned on it), which would explain part of the gate failure.
EMA hold-out counts: the GPU run (cross-checked: identical to the CPU run on 25 images).
"""
import glob
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
R = 'analysis/results/ema_ensemble'


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


def load(pattern):
    out = {}
    for p in glob.glob(pattern):
        d = json.load(open(p))
        assert d['status'] == 'done', p
        out.update({r['name']: r['counts'] for r in d['holdout']})
    return out


ema = load(f'{R}/ema_both_chunk0.json')
prod = load(f'{R}/prod_holdout_chunk*.json')
rel = {}
for p in glob.glob('analysis/results/merge/merge_chunk*.json'):
    rel.update({r['name']: r['base'] for r in json.load(open(p))['results']})
names = sorted(rel)
assert sorted(ema) == sorted(prod) == names and len(names) == 50
E, P, Rl = (np.array([x[n] for n in names], float) for x in (ema, prod, rel))
out = {}
for label, a, b in (('ema_minus_prod', E, P), ('prod_minus_released', P, Rl), ('ema_minus_released', E, Rl)):
    bs = [PQ(a[k].sum(0)) - PQ(b[k].sum(0)) for k in (rng.integers(0, 50, 50) for _ in range(4000))]
    d = PQ(a.sum(0)) - PQ(b.sum(0))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'{label:22s} {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]')
    out[label] = {'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
for label, a in (('released', Rl), ('prod_new', P), ('ema_new', E)):
    print(f'  {label:9s} PQ {PQ(a.sum(0)):.4f} TP/FP/FN {a.sum(0)[1:].astype(int).tolist()}')
    out[f'pq_{label}'] = round(PQ(a.sum(0)), 4)
json.dump(out, open(f'{R}/ema_vs_prod_summary.json', 'w'), indent=1)
