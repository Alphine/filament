"""Merge the out-of-fold runs (filament_oof_cpu.py) and summarise.

Reports, over the 657 pool images scored by the stage-A model that did not train on them:
  * pooled PQ with and without the refiner, bootstrap SE over images (the resolution
    future stage-A comparisons will have), per fold, and by number of annotator passes;
  * the single-pass vs multi-pass gap (the leaderboard-gap hypothesis) on 13x more images
    than the hold-out.
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
B = 'analysis/results/oof'

res = []
for f in range(1, 6):
    d = json.load(open(f'{B}/oof_fold{f}.json'))
    assert d['status'] == 'done', (f, d['status'])
    res += d['results']
names = [r['name'] for r in res]
assert len(names) == len(set(names)) == 657, len(set(names))
N = len(res)


def PQ(t):
    t = np.asarray(t, float)
    den = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / den if den else 0.0


def boot(a, idx=None, n=4000):
    idx = np.arange(N) if idx is None else np.asarray(idx)
    return np.array([PQ(a[rng.choice(idx, len(idx))].sum(0)) for _ in range(n)])


out = {'n_images': N}
for key in ('with_refiner', 'without_refiner'):
    a = np.array([r[key] for r in res], float)
    bs = boot(a)
    t = a.sum(0)
    print(f'{key:16s} PQ {PQ(t):.4f}  SE {bs.std():.4f}  CI95 [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]  '
          f'TP/FP/FN {t[1:].astype(int).tolist()}')
    folds = {f: round(PQ(a[[i for i, r in enumerate(res) if r['fold'] == f]].sum(0)), 4) for f in range(1, 6)}
    by_pass = {}
    for p in (1, 2, 3):
        idx = [i for i, r in enumerate(res) if r['n_pass'] == p]
        b = boot(a, idx, 2000)
        by_pass[p] = {'n': len(idx), 'pq': round(PQ(a[idx].sum(0)), 4), 'se': round(float(b.std()), 4)}
    single = [i for i, r in enumerate(res) if r['n_pass'] == 1]
    multi = [i for i, r in enumerate(res) if r['n_pass'] > 1]
    diffs = []
    for _ in range(4000):
        s, m = rng.choice(single, len(single)), rng.choice(multi, len(multi))
        diffs.append(PQ(a[s].sum(0)) - PQ(a[m].sum(0)))
    diffs = np.array(diffs)
    print(f'   per fold {folds}')
    print(f'   by passes {by_pass}')
    print(f'   single-pass minus multi-pass {np.mean(diffs):+.4f}  CI95 [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]')
    out[key] = {'pq': round(PQ(t), 4), 'se': round(float(bs.std()), 4),
                'ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                'tp_fp_fn': t[1:].astype(int).tolist(), 'per_fold': folds, 'by_passes': by_pass,
                'single_minus_multi': {'mean': round(float(diffs.mean()), 4),
                                       'ci95': [round(float(np.percentile(diffs, 2.5)), 4),
                                                round(float(np.percentile(diffs, 97.5)), 4)]}}
json.dump(out, open('analysis/results/oof_summary.json', 'w'), indent=1)
print('saved analysis/results/oof_summary.json')
