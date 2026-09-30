"""Per-epoch patch re-sampling (training/filament_resample_stagea.py) versus the baseline.

Baseline: the EMA / last-epoch weights of the checkpoint-selection runs (training/filament_ckpt_selection.py):
same recipe, seed, initialisation and folds, but the same 1,760 patches in every epoch. Both are scored on
the same out-of-fold images (each fold's validation images minus the first 20), same protocol.

Rules fixed BEFORE the run:
  primary   resampled-EMA vs baseline-EMA, paired over the pooled images, with refiner:
            adopt only if the gain is >= +0.005 with a bootstrap CI95 above 0 (needs all five folds);
  next step if adopted: ensemble-level hold-out check of the five resampled-EMA models against the
            five baseline-EMA models (0.4223 on the hold-out; both untuned on it), then one leaderboard probe;
  also reported: last-epoch weights, without refiner, per fold, and the fold-to-fold standard deviation.
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


new, base = {}, {}
for p in sorted(glob.glob('analysis/results/resample/resample_fold*.json')):
    d = json.load(open(p))
    assert 'eval' in d and not d.get('error'), (p, d.get('error'))
    new[d['fold']] = d
for p in sorted(glob.glob('analysis/results/ckpt_selection/sel_fold*.json')):
    d = json.load(open(p))
    base[d['fold']] = d
folds = sorted(new)
print('folds available:', folds)
out = {'folds': folds, 'rules': {}}
for rule in ('ema', 'last'):
    for key in ('with_refiner', 'without_refiner'):
        N, B, per_fold = [], [], []
        for f in folds:
            a = {r['name']: r[key] for r in new[f]['eval'][rule]}
            b = {r['name']: r[key] for r in base[f]['eval'][rule]}
            assert sorted(a) == sorted(b), f'fold {f}: image sets differ'
            names = sorted(a)
            fa, fb = np.array([a[n] for n in names], float), np.array([b[n] for n in names], float)
            N += list(fa)
            B += list(fb)
            per_fold.append((PQ(fb.sum(0)), PQ(fa.sum(0))))
        N, B = np.array(N), np.array(B)
        n = len(N)
        bs = [PQ(N[k].sum(0)) - PQ(B[k].sum(0)) for k in (rng.integers(0, n, n) for _ in range(4000))]
        d = PQ(N.sum(0)) - PQ(B.sum(0))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sd_b, sd_n = np.std([x for x, _ in per_fold]), np.std([y for _, y in per_fold])
        print(f'== {rule:4s} {key:15s} ({n} images): baseline {PQ(B.sum(0)):.4f} resampled {PQ(N.sum(0)):.4f} '
              f'diff {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  fold SD {sd_b:.4f} -> {sd_n:.4f}\n'
              f'      per fold (baseline -> resampled): ' + ', '.join(f'{x:.3f}->{y:.3f}' for x, y in per_fold) +
              f'\n      TP/FP/FN baseline {B.sum(0)[1:].astype(int).tolist()} resampled {N.sum(0)[1:].astype(int).tolist()}')
        out['rules'][f'{rule}_{key}'] = {'baseline': round(PQ(B.sum(0)), 4), 'resampled': round(PQ(N.sum(0)), 4),
                                         'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)], 'n': n,
                                         'fold_sd': [round(float(sd_b), 4), round(float(sd_n), 4)]}
p = out['rules']['ema_with_refiner']
ok = len(folds) == 5 and p['diff'] >= 0.005 and p['ci95'][0] > 0
out['verdict'] = ('ADOPT -> ensemble-level hold-out check, then one leaderboard probe' if ok else
                  'NOT ADOPTED (gain < +0.005 or CI includes 0)' if len(folds) == 5 else
                  f'PARTIAL ({len(folds)} of 5 folds) -- no decision yet')
print('\nVERDICT:', out['verdict'])
json.dump(out, open('analysis/results/resample_summary.json', 'w'), indent=1)
