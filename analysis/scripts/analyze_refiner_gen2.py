"""Refiner generation 2 (training/filament_refiner_gen2.py): pool both cross-fitted halves and decide.

Each half's JSON holds per-image official counts on the OTHER half's images, for every arm
(gen1, lovasz, effb3, each with and without 4-flip TTA). Pooled, every one of the 657 pool
images is refined by a refiner that never saw it (stage-A candidates are out of fold too).
Decision rule fixed before the run: an arm replaces gen1 only if its paired gain over the 657
images is >= +0.005 with a bootstrap CI95 above 0.
"""
import glob
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
halves = [json.load(open(p)) for p in sorted(glob.glob('analysis/results/refiner_gen2/gen2_half*.json'))]
assert len(halves) == 2, len(halves)
arms = sorted(set(halves[0]['arms']) & set(halves[1]['arms']))
print('arms present in both halves:', arms)
for h in halves:
    errs = [m for _, m in h['events'] if 'ERROR' in m]
    if errs:
        print(f"half {h['half']} errors:", errs)
rows = {a: {r['name']: r['counts'] for h in halves for r in h['arms'][a]['rows']} for a in arms}
names = sorted(rows['gen1'])
assert len(names) == 657 and all(sorted(rows[a]) == names for a in arms)


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


M = {a: np.array([rows[a][n] for n in names], float) for a in arms}
ref = {}
for p in glob.glob('analysis/results/oof/oof_fold*.json'):
    ref.update({r['name']: r['with_refiner'] for r in json.load(open(p))['results']})
R = np.array([ref[n] for n in names], float)
print(f'context: released refiner (trained on all 657, in-sample) on the same candidates: PQ {PQ(R.sum(0)):.4f}')
out = {'n_images': len(names), 'arms': {}}
base = M['gen1']
for a in arms:
    bs = [PQ(M[a][k].sum(0)) - PQ(base[k].sum(0)) for k in (rng.integers(0, len(names), len(names)) for _ in range(4000))]
    d = PQ(M[a].sum(0)) - PQ(base.sum(0))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    passes = a != 'gen1' and d >= 0.005 and lo > 0
    print(f'  {a:12s} PQ {PQ(M[a].sum(0)):.4f}  vs gen1 {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  '
          f'TP/FP/FN {M[a].sum(0)[1:].astype(int).tolist()}' + ('  PASSES' if passes else ''))
    out['arms'][a] = {'pq': round(PQ(M[a].sum(0)), 4), 'diff_vs_gen1': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)],
                      'passes': bool(passes)}
winners = [a for a, v in out['arms'].items() if v['passes']]
out['verdict'] = (f"REPLACE gen1 with {max(winners, key=lambda a: out['arms'][a]['diff_vs_gen1'])}" if winners
                  else 'KEEP gen1: no arm gains >= +0.005 with CI above 0')
print('\nVERDICT:', out['verdict'])
json.dump(out, open('analysis/results/refiner_gen2_summary.json', 'w'), indent=1)
