"""Consensus-target pilot (fold 1): paired comparison of the two arms, decision rule fixed before the run:
go to 5 folds only if consensus - union >= +0.005 on fold 1's out-of-fold images (production pipeline,
i.e. with refiner) with a paired bootstrap CI95 that excludes 0."""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
U = json.load(open('analysis/results/pilot/pilot_union.json'))
C = json.load(open('analysis/results/pilot/pilot_consensus.json'))
REL = {r['name']: r for r in json.load(open('analysis/results/oof/oof_fold1.json'))['results']}


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


out = {}
for part in ('oof', 'holdout'):
    names = [r['name'] for r in U[part]]
    assert names == [r['name'] for r in C[part]]
    for key in ('with_refiner', 'without_refiner'):
        u = np.array([r[key] for r in U[part]], float)
        c = np.array([r[key] for r in C[part]], float)
        n = len(names)
        bs = []
        for _ in range(4000):
            k = rng.integers(0, n, n)
            bs.append(PQ(c[k].sum(0)) - PQ(u[k].sum(0)))
        d = PQ(c.sum(0)) - PQ(u.sum(0))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{part:8s} {key:16s} union {PQ(u.sum(0)):.4f} consensus {PQ(c.sum(0)):.4f}  diff {d:+.4f}  '
              f'paired CI95 [{lo:+.4f}, {hi:+.4f}]  TP/FP/FN union {u.sum(0)[1:].astype(int).tolist()} '
              f'consensus {c.sum(0)[1:].astype(int).tolist()}')
        out[f'{part}_{key}'] = {'union': round(PQ(u.sum(0)), 4), 'consensus': round(PQ(c.sum(0)), 4),
                                'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
        if part == 'oof':
            r = np.array([REL[nm][key] for nm in names], float)
            print(f'{"":8s} {"":16s} released unet_fold1.pt on the same images: {PQ(r.sum(0)):.4f} '
                  f'(union re-run minus released {PQ(u.sum(0)) - PQ(r.sum(0)):+.4f} = run-to-run noise)')
            out[f'oof_{key}']['released_fold1'] = round(PQ(r.sum(0)), 4)
p = out['oof_with_refiner']
go = p['diff'] >= 0.005 and p['ci95'][0] > 0
verdict = ('GO: consensus beats union on the OOF images by the pre-set margin' if go else
           'NO-GO under the pre-set rule (diff < +0.005 or paired CI includes 0)')
print('\nVERDICT:', verdict)
out['verdict'] = verdict
json.dump(out, open('analysis/results/pilot_summary.json', 'w'), indent=1)
