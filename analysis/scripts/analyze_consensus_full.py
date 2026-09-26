"""5-fold consensus target vs the released (union) models. Submit criteria fixed BEFORE the run:
  1. out-of-fold, 657 images, production pipeline (with refiner), single model per image:
     consensus - released >= +0.005 with a paired bootstrap CI95 that excludes 0;
  2. hold-out, 5-model ensembles: consensus - released >= 0 (paired, same 50 images).
Both -> SUBMIT the consensus ensemble. Criterion 2 is skipped (reported as pending) if the
ensemble hold-out file is not there yet.
"""
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
R = 'analysis/results'


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


def paired(a, b, n=4000):
    k = len(a)
    bs = [PQ(a[i].sum(0)) - PQ(b[i].sum(0)) for i in (rng.integers(0, k, k) for _ in range(n))]
    return PQ(a.sum(0)) - PQ(b.sum(0)), np.percentile(bs, [2.5, 97.5])


# ---- criterion 1: out-of-fold
cons = {}
for f in range(1, 6):
    p = f'{R}/pilot/pilot_consensus_fold{f}.json'
    if f == 1 and not os.path.exists(p):
        p = f'{R}/pilot/pilot_consensus.json'
    d = json.load(open(p))
    assert 'oof' in d, (f, d.get('error'))
    for r in d['oof']:
        cons[r['name']] = r
rel = {}
for f in range(1, 6):
    for r in json.load(open(f'{R}/oof/oof_fold{f}.json'))['results']:
        rel[r['name']] = r
names = sorted(rel)
assert sorted(cons) == names and len(names) == 657
out = {}
for key in ('with_refiner', 'without_refiner'):
    c = np.array([cons[n][key] for n in names], float)
    r = np.array([rel[n][key] for n in names], float)
    d, ci = paired(c, r)
    print(f'OOF 657 {key:16s} released {PQ(r.sum(0)):.4f} consensus {PQ(c.sum(0)):.4f} diff {d:+.4f} '
          f'CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]  TP/FP/FN released {r.sum(0)[1:].astype(int).tolist()} '
          f'consensus {c.sum(0)[1:].astype(int).tolist()}')
    for p_ in (1, 2, 3):
        idx = [i for i, n in enumerate(names) if rel[n]['n_pass'] == p_]
        print(f'        {p_}-pass (n={len(idx)}): diff {PQ(c[idx].sum(0)) - PQ(r[idx].sum(0)):+.4f}')
    out[f'oof_{key}'] = {'released': round(PQ(r.sum(0)), 4), 'consensus': round(PQ(c.sum(0)), 4),
                         'diff': round(d, 4), 'ci95': [round(ci[0], 4), round(ci[1], 4)]}
c1 = bool(out['oof_with_refiner']['diff'] >= 0.005 and out['oof_with_refiner']['ci95'][0] > 0)

# ---- criterion 2: hold-out ensembles
c2 = None
hp = f'{R}/consensus_ensemble/consensus_ensemble_holdout.json'
if os.path.exists(hp):
    ce = {r['name']: r['counts'] for r in json.load(open(hp))['per_image']}
    re_ = {}
    for p in glob.glob(f'{R}/merge/merge_chunk*.json'):
        for r in json.load(open(p))['results']:
            re_[r['name']] = r['base']
    hn = sorted(ce)
    assert sorted(re_) == hn
    a, b = np.array([ce[n] for n in hn], float), np.array([re_[n] for n in hn], float)
    d, ci = paired(a, b)
    print(f'hold-out ensembles: released {PQ(b.sum(0)):.4f} consensus {PQ(a.sum(0)):.4f} diff {d:+.4f} '
          f'CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]  TP/FP/FN released {b.sum(0)[1:].astype(int).tolist()} '
          f'consensus {a.sum(0)[1:].astype(int).tolist()}')
    out['holdout_ensemble'] = {'released': round(PQ(b.sum(0)), 4), 'consensus': round(PQ(a.sum(0)), 4),
                               'diff': round(d, 4), 'ci95': [round(ci[0], 4), round(ci[1], 4)]}
    c2 = bool(d >= 0)
verdict = ('SUBMIT' if c1 and c2 else 'DO NOT SUBMIT' if (c1 is False or c2 is False) else 'PENDING')
print(f'\ncriterion 1 (OOF >= +0.005, CI > 0): {c1} | criterion 2 (hold-out ensemble >= released): {c2} -> {verdict}')
out.update({'criterion_oof': bool(c1), 'criterion_holdout': c2, 'verdict': verdict})
json.dump(out, open(f'{R}/consensus_full_summary.json', 'w'), indent=1)
