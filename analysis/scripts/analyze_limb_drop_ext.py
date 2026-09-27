"""Choose the near-limb drop threshold t on the extended grid (filament_limb_drop_ext_cpu.py).

Rule fixed beforehand: t = argmax out-of-fold PQ over the grid; report the selection-aware
split-half CV gain; the chosen t is submitted only if its hold-out (released ensemble) effect is >= 0
and its out-of-fold paired CI95 excludes 0.
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
d = json.load(open('analysis/results/limb_mask/limb_drop_ext.json'))
base = {r['name'] + r['set']: r['base'] for r in json.load(open('analysis/results/limb_mask/limb_mask.json'))['results']}
res = d['results']


def PQ(t):
    t = np.asarray(t, float)
    den = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / den if den else 0.0


out = {}
for part in ('oof', 'holdout'):
    rs = [r for r in res if r['set'] == part]
    B = np.array([base[r['name'] + part] for r in rs], float)
    R = {k: np.array([r['rules'][k] for r in rs], float) for k in rs[0]['rules']}
    out[part] = {'n': len(rs), 'base': round(PQ(B.sum(0)), 4)}
    for k, a in R.items():
        n = len(rs)
        bs = [PQ(a[i].sum(0)) - PQ(B[i].sum(0)) for i in (rng.integers(0, n, n) for _ in range(2000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[part][k] = {'diff': round(PQ(a.sum(0)) - PQ(B.sum(0)), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                        'tp_fp_fn': a.sum(0)[1:].astype(int).tolist()}
    if part == 'oof':
        gains = []
        for _ in range(500):
            p = rng.permutation(n)
            A, Bi = p[:n // 2], p[n // 2:]
            k = max(R, key=lambda k: PQ(R[k][A].sum(0)) - PQ(B[A].sum(0)))
            gains.append(PQ(R[k][Bi].sum(0)) - PQ(B[Bi].sum(0)))
        out['cv_gain'] = round(float(np.mean(gains)), 4)
        out['chosen'] = max(R, key=lambda k: PQ(R[k].sum(0)))
for part in ('oof', 'holdout'):
    print(f"== {part} (n={out[part]['n']}, base {out[part]['base']})")
    for k, v in out[part].items():
        if k.startswith('drop'):
            print(f"  {k:10s} {v['diff']:+.4f} CI95 [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] TP/FP/FN {v['tp_fp_fn']}")
c = out['chosen']
ok = out['holdout'][c]['diff'] >= 0 and out['oof'][c]['ci95'][0] > 0
out['submit'] = bool(ok)
print(f"\nchosen {c}: OOF {out['oof'][c]['diff']:+.4f}, hold-out {out['holdout'][c]['diff']:+.4f}, "
      f"selection-aware CV gain {out['cv_gain']:+.4f} -> {'SUBMIT' if ok else 'DO NOT SUBMIT'}")
json.dump(out, open('analysis/results/limb_drop_ext_summary.json', 'w'), indent=1)
