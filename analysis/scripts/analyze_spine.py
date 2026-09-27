"""Spine multi-task stage-A (training/filament_spine_stagea.py) versus the baseline EMA stage-A.

Baseline: the EMA weights of the checkpoint-selection runs (training/filament_ckpt_selection.py):
same recipe, same seed and data order, union target, no spine head. Both are scored on the same
out-of-fold images (each fold's validation images minus the first 20), same protocol.
Decision rule fixed beforehand: spine-EMA replaces baseline-EMA only if the paired gain pooled over
the five folds (with refiner) is >= +0.005 with a bootstrap CI95 above 0.
"""
import glob
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


spine, base = {}, {}
for p in sorted(glob.glob('analysis/results/spine/spine_fold*.json')):
    d = json.load(open(p))
    assert 'eval' in d and not d.get('error'), (p, d.get('error'))
    spine[d['fold']] = d
for p in sorted(glob.glob('analysis/results/ckpt_selection/sel_fold*.json')):
    d = json.load(open(p))
    base[d['fold']] = d
folds = sorted(spine)
print('folds available:', folds)
out = {'folds': folds}
for key in ('with_refiner', 'without_refiner'):
    S, B = [], []
    for f in folds:
        s = {r['name']: r[key] for r in spine[f]['eval']['ema']}
        b = {r['name']: r[key] for r in base[f]['eval']['ema']}
        assert sorted(s) == sorted(b), f'fold {f}: image sets differ'
        names = sorted(s)
        S += [s[n] for n in names]
        B += [b[n] for n in names]
        print(f'  fold {f} {key:15s}: baseline EMA {PQ(np.sum([b[n] for n in names], 0)):.4f} | '
              f'spine EMA {PQ(np.sum([s[n] for n in names], 0)):.4f}')
    S, B = np.array(S, float), np.array(B, float)
    n = len(S)
    bs = [PQ(S[k].sum(0)) - PQ(B[k].sum(0)) for k in (rng.integers(0, n, n) for _ in range(4000))]
    d = PQ(S.sum(0)) - PQ(B.sum(0))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'== {key} ({n} images): baseline {PQ(B.sum(0)):.4f} spine {PQ(S.sum(0)):.4f} diff {d:+.4f} '
          f'CI95 [{lo:+.4f}, {hi:+.4f}]  TP/FP/FN baseline {B.sum(0)[1:].astype(int).tolist()} spine {S.sum(0)[1:].astype(int).tolist()}')
    out[key] = {'baseline': round(PQ(B.sum(0)), 4), 'spine': round(PQ(S.sum(0)), 4), 'diff': round(d, 4),
                'ci95': [round(lo, 4), round(hi, 4)], 'n': n}
w = out['with_refiner']
ok = len(folds) == 5 and w['diff'] >= 0.005 and w['ci95'][0] > 0
out['verdict'] = ('PASS -> hold-out gate' if ok else
                  'NOT ACTIONABLE' if len(folds) == 5 else f'PARTIAL ({len(folds)} of 5 folds) -- no decision yet')
print('\nVERDICT:', out['verdict'])
json.dump(out, open('analysis/results/spine_summary.json', 'w'), indent=1)
