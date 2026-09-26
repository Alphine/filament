"""Split the hold-out vs out-of-fold gap into a hold-out effect and an ensemble effect.

Inputs: results/holdout_single/holdout_single_fold{1..5}.json (each fold model alone on the
50 hold-out images) and results/oof/oof_fold{1..5}.json (each fold model on its own 131-132
validation images). Same model, same protocol, different images:
  hold-out effect  = PQ(model on hold-out) - PQ(model on its OOF images), per model, averaged
                     (cleanest WITHOUT the refiner, which was trained on the OOF images);
  ensemble effect  = 0.4325 (5-model ensemble on the hold-out) - mean single-model hold-out PQ
                     (with refiner, as in the released pipeline).
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
ENSEMBLE_HOLDOUT = 0.4325


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


hs = {f: json.load(open(f'analysis/results/holdout_single/holdout_single_fold{f}.json')) for f in range(1, 6)}
oof = {f: json.load(open(f'analysis/results/oof/oof_fold{f}.json')) for f in range(1, 6)}
assert all(d['status'] == 'done' for d in list(hs.values()) + list(oof.values()))
names = [r['name'] for r in hs[1]['results']]
assert all([r['name'] for r in hs[f]['results']] == names for f in hs)

out = {}
for key in ('without_refiner', 'with_refiner'):
    H = {f: np.array([r[key] for r in hs[f]['results']], float) for f in hs}
    O = {f: np.array([r[key] for r in oof[f]['results']], float) for f in oof}
    rows = {f: (PQ(H[f].sum(0)), PQ(O[f].sum(0))) for f in hs}
    diffs = np.array([h - o for h, o in rows.values()])
    # uncertainty: bootstrap hold-out images (shared across models) and each fold's OOF images
    bs = []
    for _ in range(2000):
        hi = rng.integers(0, len(names), len(names))
        d = []
        for f in hs:
            oi = rng.integers(0, len(O[f]), len(O[f]))
            d.append(PQ(H[f][hi].sum(0)) - PQ(O[f][oi].sum(0)))
        bs.append(np.mean(d))
    mean_h = float(np.mean([h for h, _ in rows.values()]))
    print(f'== {key} ==')
    for f, (h, o) in rows.items():
        print(f'  fold {f}: hold-out {h:.4f} | own OOF images {o:.4f} | difference {h - o:+.4f}')
    print(f'  mean single-model hold-out PQ {mean_h:.4f}; hold-out effect {diffs.mean():+.4f} '
          f'CI95 [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]')
    out[key] = {'per_fold': {f: {'holdout': round(h, 4), 'oof': round(o, 4)} for f, (h, o) in rows.items()},
                'mean_single_holdout': round(mean_h, 4), 'holdout_effect': round(float(diffs.mean()), 4),
                'holdout_effect_ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
ens = ENSEMBLE_HOLDOUT - out['with_refiner']['mean_single_holdout']
print(f'\nensemble effect on the hold-out (0.4325 - mean single model, with refiner): {ens:+.4f}')
out['ensemble_effect_holdout'] = round(ens, 4)
json.dump(out, open('analysis/results/holdout_single_summary.json', 'w'), indent=1)
print('saved analysis/results/holdout_single_summary.json')
