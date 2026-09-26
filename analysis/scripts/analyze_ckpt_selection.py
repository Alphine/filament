"""Checkpoint-selection study (training/filament_ckpt_selection.py): last / SWA / EMA versus the
production rule on the same training trajectories, five folds, out-of-fold images minus the 20
the production rule selects on.
Decision rule fixed before the run: a rule replaces prod only if its paired gain pooled over the
five folds (with refiner) is >= +0.005 with a bootstrap CI95 that excludes 0.
Also reported: fold-to-fold spread of each rule (the variance the study was motivated by).
"""
import glob
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
RULES = ('prod', 'last', 'swa', 'ema')


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


folds = {}
for p in sorted(glob.glob('analysis/results/ckpt_selection/sel_fold*.json')):
    d = json.load(open(p))
    assert 'eval' in d and not d.get('error'), (p, d.get('error'))
    folds[d['fold']] = d
assert sorted(folds) == [1, 2, 3, 4, 5], sorted(folds)
out = {'per_fold': {}, 'pooled': {}}
for f, d in folds.items():
    out['per_fold'][f] = {r: d['eval_pq'][r] for r in RULES} | {'prod_epoch': d['train']['prod_best_epoch']}
    print(f"fold {f}: prod epoch {d['train']['prod_best_epoch']:2d} | " +
          ' '.join(f"{r} {d['eval_pq'][r]['with_refiner']:.4f}" for r in RULES))
for key in ('with_refiner', 'without_refiner'):
    A = {r: np.array([row[key] for f in sorted(folds) for row in folds[f]['eval'][r]], float) for r in RULES}
    names = [row['name'] for f in sorted(folds) for row in folds[f]['eval']['prod']]
    for r in RULES:
        assert [row['name'] for f in sorted(folds) for row in folds[f]['eval'][r]] == names
    n = len(names)
    print(f'\n== {key} ({n} images) ==')
    out['pooled'][key] = {}
    for r in RULES:
        spread = np.std([PQ(np.array([row[key] for row in folds[f]['eval'][r]]).sum(0)) for f in sorted(folds)])
        line = f'  {r:5s} PQ {PQ(A[r].sum(0)):.4f}  fold spread (SD) {spread:.4f}'
        res = {'pq': round(PQ(A[r].sum(0)), 4), 'fold_sd': round(float(spread), 4)}
        if r != 'prod':
            bs = [PQ(A[r][k].sum(0)) - PQ(A['prod'][k].sum(0)) for k in (rng.integers(0, n, n) for _ in range(4000))]
            d_ = PQ(A[r].sum(0)) - PQ(A['prod'].sum(0))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line += f'  vs prod {d_:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]'
            res.update({'diff_vs_prod': round(d_, 4), 'ci95': [round(lo, 4), round(hi, 4)],
                        'passes': bool(key == 'with_refiner' and d_ >= 0.005 and lo > 0)})
        print(line)
        out['pooled'][key][r] = res
winners = [r for r, v in out['pooled']['with_refiner'].items() if v.get('passes')]
verdict = (f'REPLACE prod with {max(winners, key=lambda r: out["pooled"]["with_refiner"][r]["diff_vs_prod"])}'
           if winners else 'KEEP prod: no rule gains >= +0.005 with CI above 0')
print('\nVERDICT:', verdict)
out['verdict'] = verdict
json.dump(out, open('analysis/results/ckpt_selection_summary.json', 'w'), indent=1)
