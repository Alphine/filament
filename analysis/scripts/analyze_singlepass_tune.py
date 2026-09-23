"""Single-pass threshold re-tuning: merge the tune chunks (filament_singlepass_tune_cpu.py) and decide.

Rules fixed BEFORE looking at the results:
  * integrity: the production configuration (0.4|0.7|400|0.5) must reproduce pooled
    TP/FP/FN 400/239/211, and the rebuilt candidate maps must equal the stored ones.
  * the configuration is chosen on a random half of the 31 single-pass images and scored
    on the other half (500 repeats); mean out-of-sample gain over production < +0.005
    -> NOT ACTIONABLE.
Also reported: the same protocol with the configuration chosen on all images (the way the
production thresholds were originally tuned), and what a single-pass-tuned configuration
does to the 19 multi-pass images.
"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
B = 'analysis/results/singlepass_tune'
PROD = '0.4|0.7|400|0.5'

res = []
for i in range(4):
    d = json.load(open(f'{B}/tune_chunk{i}.json'))
    assert d['status'] == 'done', (i, d['status'])
    res += d['results']
N = len(res)
cfgs = sorted(res[0]['counts'])
C = {c: np.array([r['counts'][c] for r in res], float) for c in cfgs}
single = np.array([i for i, r in enumerate(res) if r['n_pass'] == 1])
multi = np.array([i for i, r in enumerate(res) if r['n_pass'] > 1])


def PQ(t):
    t = np.asarray(t, float)
    den = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / den if den else 0.0


def pq_on(c, idx):
    return PQ(C[c][idx].sum(0))


tot = C[PROD].sum(0)
print(f'images {N} (single-pass {len(single)}, multi-pass {len(multi)}), configurations {len(cfgs)}')
print(f'production {PROD}: PQ {PQ(tot):.4f} TP/FP/FN {tot[1:].astype(int).tolist()} (integrity 400/239/211); '
      f'candidate maps reproduced on {sum(r["cand_matches_stored"] for r in res)}/{N} images')
integrity = tot[1:].astype(int).tolist() == [400, 239, 211]
print('INTEGRITY', 'OK' if integrity else 'FAILED (uint8 probability quantisation?) -- compare configurations relative to the recomputed production row')

print(f'\nproduction PQ: all {pq_on(PROD, np.arange(N)):.4f} | single-pass {pq_on(PROD, single):.4f} | multi-pass {pq_on(PROD, multi):.4f}')


def top(idx, k=8):
    ranked = sorted(cfgs, key=lambda c: -pq_on(c, idx))
    return [(c, round(pq_on(c, idx), 4)) for c in ranked[:k]]


print('\nbest configurations IN-SAMPLE on single-pass images (optimistic, 300 tried):')
for c, v in top(single):
    print(f'  {c:22s} single {v:.4f} ({v - pq_on(PROD, single):+.4f})  multi {pq_on(c, multi):.4f} '
          f'({pq_on(c, multi) - pq_on(PROD, multi):+.4f})  all {pq_on(c, np.arange(N)):.4f}')
print('best configurations IN-SAMPLE on all images:')
for c, v in top(np.arange(N), 5):
    print(f'  {c:22s} all {v:.4f} ({v - pq_on(PROD, np.arange(N)):+.4f})')


def cv(select_pool, eval_pool, reps=500):
    """Choose the config on a random half of eval_pool (plus all of select_pool extra), score on the other half."""
    g = []
    for _ in range(reps):
        perm = rng.permutation(eval_pool)
        A, Bh = perm[:len(perm) // 2], perm[len(perm) // 2:]
        pool = np.concatenate([A, select_pool]) if len(select_pool) else A
        best = max(cfgs, key=lambda c: pq_on(c, pool))
        g.append(pq_on(best, Bh) - pq_on(PROD, Bh))
    return np.array(g)


g_single = cv(np.array([], int), single)
g_all = cv(multi, single)  # choose on (half of single + all multi), score on the other single half
for name, g in (('tuned on single-pass only', g_single), ('tuned on all images', g_all)):
    print(f'\nsplit-half CV, scored on held-out single-pass images -- {name}: mean {g.mean():+.4f}, '
          f'median {np.median(g):+.4f}, P(gain>0) {np.mean(g > 0):.0%}, 5th/95th {np.percentile(g, 5):+.4f}/{np.percentile(g, 95):+.4f}')

verdict = ('NOT ACTIONABLE: out-of-sample gain on single-pass images below +0.005'
           if g_single.mean() < 0.005 else 'PASS: single-pass tuning gains out of sample')
print('\nVERDICT:', verdict)
best_s = top(single, 1)[0][0]
json.dump({'n_images': N, 'n_single': int(len(single)), 'n_configs': len(cfgs), 'integrity_ok': bool(integrity),
           'production': {'all': round(pq_on(PROD, np.arange(N)), 4), 'single': round(pq_on(PROD, single), 4),
                          'multi': round(pq_on(PROD, multi), 4)},
           'best_single_in_sample': {'config': best_s, 'single': round(pq_on(best_s, single), 4),
                                     'multi': round(pq_on(best_s, multi), 4), 'all': round(pq_on(best_s, np.arange(N)), 4)},
           'cv_single_tuned': {'mean': round(float(g_single.mean()), 4), 'p_gain_positive': round(float(np.mean(g_single > 0)), 3)},
           'cv_all_tuned': {'mean': round(float(g_all.mean()), 4), 'p_gain_positive': round(float(np.mean(g_all > 0)), 3)},
           'verdict': verdict}, open('analysis/results/singlepass_tune_summary.json', 'w'), indent=1)
print('saved analysis/results/singlepass_tune_summary.json')
