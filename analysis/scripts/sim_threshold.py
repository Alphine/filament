"""Simulation of TIGHTENING the candidate thresholds (min_mean_prob, min_area, ...)
using the per-candidate data in analysis/results/fpdiag (final pipeline). The
simulation is exact for tightening (the surviving candidates are a subset of the
existing ones); loosening would need the model to be re-run. Includes repeated
split-half cross-validation so that the best thresholds are not scored on the
same data they were selected on."""
import json, itertools, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
imgs = []
for i in range(4):
    imgs += json.load(open(f'analysis/results/fpdiag/holdout_chunk{i}.json'))['results']
N = len(imgs)

def stats(r, keep):
    ids = r['pred_cand_ids']
    kp = {j for j, c in enumerate(ids) if c in keep}
    S = tp = fp = fn = 0
    for p in r['passes']:
        pr = [x for x in p['pairs'] if x[0] in kp]
        S += sum(x[2] for x in pr); tp += len(pr)
        fp += len(kp) - len({x[0] for x in pr}); fn += p['n_gt'] - len({x[1] for x in pr})
    return S, tp, fp, fn
def PQ(t):
    d = t[1] + .5 * t[2] + .5 * t[3]; return t[0] / d if d else 0.0

feat = [{c['id']: c['feats'] for c in r['cands']} for r in imgs]
def keep_set(gi, mp, ma, mx):
    return {cid for cid, f in feat[gi].items() if f['mean_p'] >= mp and f['area'] >= ma and f['max_p'] >= mx}
def per_image(mp, ma, mx):
    return np.array([stats(imgs[g], keep_set(g, mp, ma, mx)) for g in range(N)])

base = per_image(0.7, 400, 0.0)
print('base PQ', round(PQ(base.sum(0)), 4), 'TP/FP/FN', base.sum(0)[1:].astype(int).tolist())
rng = np.random.default_rng(0)
grid = [(mp, ma, mx) for mp in (0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90)
        for ma in (400, 600, 800, 1000, 1500) for mx in (0.0, 0.9, 0.95)]
cache = {g: per_image(*g) for g in grid}
def boot_diff(a, b, n=1500):
    d = []
    for _ in range(n):
        k = rng.integers(0, N, N); d.append(PQ(a[k].sum(0)) - PQ(b[k].sum(0)))
    return np.mean(d), np.percentile(d, [2.5, 97.5])

print('\n--- one dimension at a time ---')
for name, sel in (('min_mean_prob', lambda g: g[1] == 400 and g[2] == 0.0),
                  ('min_area',     lambda g: g[0] == 0.70 and g[2] == 0.0),
                  ('min_max_prob', lambda g: g[0] == 0.70 and g[1] == 400)):
    for g in grid:
        if sel(g) and g != (0.7, 400, 0.0):
            t = cache[g].sum(0); m, ci = boot_diff(cache[g], base)
            print(f'{name:14s} {g}: PQ {PQ(t):.4f}  d={PQ(t)-PQ(base.sum(0)):+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]  '
                  f'TP {int(t[1])} FP {int(t[2])} FN {int(t[3])}')
best = max(grid, key=lambda g: PQ(cache[g].sum(0)))
m, ci = boot_diff(cache[best], base)
print(f'\nBEST over the whole grid ({len(grid)} points, chosen on the same data = optimistic): {best} '
      f'PQ {PQ(cache[best].sum(0)):.4f}  d={PQ(cache[best].sum(0))-PQ(base.sum(0)):+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]')

# repeated split-half cross-validation: pick the best thresholds on half A, score them on half B
gains = []
for _ in range(300):
    perm = rng.permutation(N); A, B = perm[:N // 2], perm[N // 2:]
    g = max(grid, key=lambda g: PQ(cache[g][A].sum(0)))
    gains.append(PQ(cache[g][B].sum(0)) - PQ(base[B].sum(0)))
gains = np.array(gains)
print(f'\nSPLIT-HALF CROSS-VALIDATION (300 repeats): out-of-sample gain mean {gains.mean():+.4f}, '
      f'median {np.median(gains):+.4f}, P(gain>0) {np.mean(gains > 0):.0%}, 5th/95th percentile '
      f'{np.percentile(gains,5):+.4f}/{np.percentile(gains,95):+.4f}')
