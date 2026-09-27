"""Off-limb / near-limb masking (filament_limb_mask_cpu.py): decide with a rule fixed beforehand.

Out-of-fold (657 images, production pipeline with refiner, single model per image): the rule is
chosen on a random half of the images among all drop@t / clip@t and scored on the other half
(500 repeats). Actionable only if the mean out-of-sample gain is >= +0.005 and the paired CI95 of
the chosen rule on all images excludes 0. The hold-out (released 5-model ensemble) effect of every
rule is reported alongside.
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
d = json.load(open('analysis/results/limb_mask/limb_mask.json'))
assert d['status'] == 'done'
oof = [r for r in d['results'] if r['set'] == 'oof']
hold = [r for r in d['results'] if r['set'] == 'holdout']
print(f"images: OOF {len(oof)}, hold-out {len(hold)}; OOF base == stored counts on "
      f"{sum(r['base_matches_stored'] for r in oof)}/{len(oof)}")
assert len(oof) == 657 and len(hold) == 50


def PQ(t):
    t = np.asarray(t, float)
    den = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / den if den else 0.0


rules = list(oof[0]['rules'])
B = np.array([r['base'] for r in oof], float)
R = {k: np.array([r['rules'][k] for r in oof], float) for k in rules}
Bh = np.array([r['base'] for r in hold], float)
Rh = {k: np.array([r['rules'][k] for r in hold], float) for k in rules}
dist = np.concatenate([r['inst_dist'] for r in oof])
print(f'instance centroid distance / R: median {np.median(dist):.3f}, share > 0.90: {np.mean(dist > .90):.1%}, '
      f'> 0.95: {np.mean(dist > .95):.1%}, > 1.00: {np.mean(dist > 1.0):.1%}')
print(f'\nbase PQ: OOF {PQ(B.sum(0)):.4f} | hold-out {PQ(Bh.sum(0)):.4f}')
n = len(oof)
out = {'rules': {}}
for k in rules:
    bs = [PQ(R[k][i].sum(0)) - PQ(B[i].sum(0)) for i in (rng.integers(0, n, n) for _ in range(2000))]
    dd = PQ(R[k].sum(0)) - PQ(B.sum(0))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    dh = PQ(Rh[k].sum(0)) - PQ(Bh.sum(0))
    print(f'  {k:9s} OOF {dd:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  TP/FP/FN {R[k].sum(0)[1:].astype(int).tolist()} | hold-out {dh:+.4f}')
    out['rules'][k] = {'oof_diff': round(dd, 4), 'ci95': [round(lo, 4), round(hi, 4)], 'holdout_diff': round(dh, 4)}
gains = []
for _ in range(500):
    p = rng.permutation(n)
    A, Bi = p[:n // 2], p[n // 2:]
    k = max(rules, key=lambda k: PQ(R[k][A].sum(0)) - PQ(B[A].sum(0)))
    gains.append(PQ(R[k][Bi].sum(0)) - PQ(B[Bi].sum(0)))
gains = np.array(gains)
best = max(rules, key=lambda k: out['rules'][k]['oof_diff'])
ok = gains.mean() >= 0.005 and out['rules'][best]['ci95'][0] > 0
verdict = f'ACTIONABLE: {best}' if ok else 'NOT ACTIONABLE (out-of-sample gain < +0.005 or CI includes 0)'
print(f'\nsplit-half CV: mean {gains.mean():+.4f}, P(gain>0) {np.mean(gains > 0):.0%}; best in-sample {best}')
print('VERDICT:', verdict)
out.update({'base_oof': round(PQ(B.sum(0)), 4), 'base_holdout': round(PQ(Bh.sum(0)), 4),
            'cv_mean_gain': round(float(gains.mean()), 4), 'best_rule': best, 'verdict': verdict})
json.dump(out, open('analysis/results/limb_mask_summary.json', 'w'), indent=1)
