"""Merge the fragment-merging chunks (filament_merge_cpu.py) and decide.

Stop rules, fixed BEFORE looking at the results:
  * oracle gain (best of the two oracles) < +0.01  -> STOP, fragment merging is not worth pursuing.
  * best realistic rule, out-of-sample (split-half CV over images, rule chosen among all
    variant x d on one half and scored on the other) < +0.005 -> NOT ACTIONABLE.
Also reports the 31 single-pass images separately (closest to the leaderboard setting).
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
B = 'analysis/results/merge'

res = []
for i in range(4):
    d = json.load(open(f'{B}/merge_chunk{i}.json'))
    assert d['status'] == 'done', (i, d['status'])
    res += d['results']
N = len(res)
D = [str(x) for x in json.load(open(f'{B}/merge_chunk0.json'))['d_grid']]


def PQ(t):
    t = np.asarray(t, float)
    den = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / den if den else 0.0


base = np.array([r['base'] for r in res], float)
tot = base.sum(0)
print(f'images {N}; base PQ {PQ(tot):.4f}  TP/FP/FN {tot[1:].astype(int).tolist()}  (integrity: 400/239/211)')
assert tot[1:].astype(int).tolist() == [400, 239, 211], 'integrity check failed'

rules = {}
for v in ('union', 'union_close', 'cand_refine'):
    for d in D:
        rules[f'{v}@{d}'] = np.array([r[v][d] for r in res], float)
oracles = {k: np.array([r[k]['counts'] for r in res], float) for k in ('oracle_union', 'oracle_refine')}


def boot_diff(a, b, idx=None, n=2000):
    idx = np.arange(N) if idx is None else np.asarray(idx)
    out = []
    for _ in range(n):
        k = rng.choice(idx, len(idx))
        out.append(PQ(a[k].sum(0)) - PQ(b[k].sum(0)))
    return np.percentile(out, [2.5, 97.5])


single = [i for i, r in enumerate(res) if r['n_pass'] == 1]
print(f'single-pass images: {len(single)}; base PQ there {PQ(base[single].sum(0)):.4f}')

print('\n== ORACLE upper bounds (use ground truth; not a method) ==')
summary = {'n_images': N, 'base_pq': round(PQ(tot), 4), 'oracle': {}, 'rules': {}}
for k, v in oracles.items():
    g = PQ(v.sum(0)) - PQ(tot)
    merges = sum(r[k]['merges'] for r in res)
    gs = PQ(v[single].sum(0)) - PQ(base[single].sum(0))
    print(f'  {k:14s} PQ {PQ(v.sum(0)):.4f}  gain {g:+.4f}  ({merges} merges; single-pass gain {gs:+.4f})  '
          f'TP/FP/FN {v.sum(0)[1:].astype(int).tolist()}')
    summary['oracle'][k] = {'pq': round(PQ(v.sum(0)), 4), 'gain': round(g, 4), 'merges': merges,
                            'gain_single_pass': round(gs, 4)}
best_oracle = max(s['gain'] for s in summary['oracle'].values())

print('\n== realistic distance rules (no ground truth), in-sample ==')
for k, v in rules.items():
    g = PQ(v.sum(0)) - PQ(tot)
    ci = boot_diff(v, base)
    gs = PQ(v[single].sum(0)) - PQ(base[single].sum(0))
    nm = sum(r['n_groups_merged'][k.split('@')[1]] for r in res) if not k.startswith('cand') else None
    print(f'  {k:16s} PQ {PQ(v.sum(0)):.4f}  d={g:+.4f}  paired CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  '
          f'single-pass d={gs:+.4f}' + (f'  groups merged {nm}' if nm is not None else ''))
    summary['rules'][k] = {'pq': round(PQ(v.sum(0)), 4), 'gain': round(g, 4),
                           'ci95': [round(ci[0], 4), round(ci[1], 4)], 'gain_single_pass': round(gs, 4)}

gains = []
for _ in range(500):
    perm = rng.permutation(N)
    A, Bh = perm[:N // 2], perm[N // 2:]
    kbest = max(rules, key=lambda k: PQ(rules[k][A].sum(0)) - PQ(base[A].sum(0)))
    gains.append(PQ(rules[kbest][Bh].sum(0)) - PQ(base[Bh].sum(0)))
gains = np.array(gains)
best_in = max(rules, key=lambda k: PQ(rules[k].sum(0)))
print(f'\nsplit-half CV over all {len(rules)} rules (500 repeats): mean {gains.mean():+.4f}, median {np.median(gains):+.4f}, '
      f'P(gain>0) {np.mean(gains > 0):.0%}, 5th/95th {np.percentile(gains, 5):+.4f}/{np.percentile(gains, 95):+.4f}')
print(f'best rule in-sample: {best_in}  gain {PQ(rules[best_in].sum(0)) - PQ(tot):+.4f}')

if best_oracle < 0.01:
    verdict = f'STOP: best oracle gain {best_oracle:+.4f} < +0.01, fragment merging is not worth pursuing'
elif gains.mean() < 0.005:
    verdict = (f'ORACLE PASSES ({best_oracle:+.4f}) but the realistic distance rules are NOT ACTIONABLE '
               f'(out-of-sample {gains.mean():+.4f} < +0.005): a better merge decision rule would be needed')
else:
    verdict = f'PASS: oracle {best_oracle:+.4f}, realistic out-of-sample {gains.mean():+.4f}'
print('\nVERDICT:', verdict)
summary.update({'best_rule_in_sample': best_in,
                'split_half_cv_gain': {'mean': round(float(gains.mean()), 4), 'median': round(float(np.median(gains)), 4),
                                       'p_gain_positive': round(float(np.mean(gains > 0)), 3)},
                'verdict': verdict})
json.dump(summary, open('analysis/results/merge_summary.json', 'w'), indent=1)
print('saved analysis/results/merge_summary.json')
