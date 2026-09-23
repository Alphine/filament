"""CPU falsification test: does `std_views` (TTA/fold disagreement) or a calibrated
`mean_p` make a better FP filter than the already-tested full-feature classifier
(AUC 0.70, found NOT actionable as a filter)?

Motivation. Re-reading the existing fpdiag data (no new run) found `std_views` alone
has AUC 0.715 for TP-vs-not, cleaner and more monotonic than `mean_p` alone (AUC 0.710).
That is close to the full-feature classifier's 0.70, so the prior expectation is that
this ALSO fails as a practical filter -- but it has not been tested with the same
PQ-simulation protocol, so we test it directly instead of assuming.

Protocol (identical discipline to support_classifier.py):
  * label = TP if the candidate is IoU>0.5 in at least one annotator pass of its image.
  * scores: (a) std_views alone (b) mean_p alone (c) logistic(std_views, mean_p) --
    image-grouped repeated CV, so a candidate is scored by folds that never saw its image.
  * PQ simulation is exact (dropping a candidate changes TP/FP/FN of every pass it's in).
  * threshold chosen inside split-half CV so the reported gain is out-of-sample.
Stop rule (same as support_classifier.py): AUC < 0.70 or best simulated gain < +0.005 -> stop.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)

imgs = []
for i in range(4):
    imgs += json.load(open(f'analysis/results/fpdiag/holdout_chunk{i}.json'))['results']
N = len(imgs)


def stats(r, keep_ids):
    ids = r['pred_cand_ids']
    kp = {j for j, c in enumerate(ids) if c in keep_ids}
    S = tp = fp = fn = 0
    for p in r['passes']:
        pr = [x for x in p['pairs'] if x[0] in kp]
        S += sum(x[2] for x in pr)
        tp += len(pr)
        fp += len(kp) - len({x[0] for x in pr})
        fn += p['n_gt'] - len({x[1] for x in pr})
    return S, tp, fp, fn


def PQ(t):
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


# ------------------------------------------------------------------ candidate table
rows = []
for gi, r in enumerate(imgs):
    ids = r['pred_cand_ids']
    feats = {c['id']: c['feats'] for c in r['cands']}
    for j, cid in enumerate(ids):
        nh = sum(any(x[0] == j for x in p['pairs']) for p in r['passes'])
        rows.append(dict(img=gi, cid=cid, nh=nh, f=feats[cid]))

img_of = np.array([r['img'] for r in rows])
y = (np.array([r['nh'] for r in rows]) >= 1).astype(int)  # TP in >=1 pass vs never matched
std_views = np.array([r['f']['std_views'] for r in rows])
mean_p = np.array([r['f']['mean_p'] for r in rows])
groups = np.unique(img_of)
print(f'images {N}, candidates {len(rows)}, TP-labelled {int(y.sum())} ({y.mean():.1%})')


def oof_logistic(cols, reps=20, k=5):
    X = np.column_stack(cols)
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for _ in range(reps):
        perm = rng.permutation(groups)
        for f in np.array_split(perm, k):
            te = np.isin(img_of, f)
            if y[~te].min() == y[~te].max():
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[~te], y[~te])
            acc[te] += m.predict_proba(X[te])[:, 1]
            cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def auc_ci(score, n=2000):
    a = roc_auc_score(y, score)
    bs = []
    for _ in range(n):
        pick = rng.choice(groups, len(groups))
        idx = np.concatenate([np.flatnonzero(img_of == g) for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        bs.append(roc_auc_score(y[idx], score[idx]))
    return a, np.percentile(bs, [2.5, 97.5])


scores = {
    'std_views_alone': -std_views,           # lower std_views = more likely TP
    'mean_p_alone': mean_p,
    'logistic(std_views, mean_p)': oof_logistic([std_views, mean_p]),
}
print('\n== AUC for TP-vs-not (image-grouped CV where applicable) ==')
res = {}
for name, sc in scores.items():
    a, ci = auc_ci(sc)
    res[name] = (a, ci)
    print(f'  {name:28s} AUC {a:.3f}  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]')

# ------------------------------------------------------------------ PQ simulation
base_img = np.array([stats(imgs[g], {r['cid'] for r in rows if r['img'] == g}) for g in range(N)])
print(f'\nbase PQ over 50 images {PQ(base_img.sum(0)):.4f}  (integrity check vs known 0.4325)')


def pq_with_drop(score, thr):
    per = []
    for g in range(N):
        keep = {r['cid'] for k, r in enumerate(rows) if r['img'] == g and not (score[k] < thr)}
        per.append(stats(imgs[g], keep))
    return np.array(per)


def boot_diff(a, b, n=2000):
    d = []
    for _ in range(n):
        k = rng.integers(0, N, N)
        d.append(PQ(a[k].sum(0)) - PQ(b[k].sum(0)))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


qgrid = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
summary = {}
for name, sc in scores.items():
    print(f'\n== drop candidates scored below the quantile -- score = {name} ==')
    qs = np.quantile(sc, qgrid)
    sim = {}
    for q, thr in zip(qgrid, qs):
        per = pq_with_drop(sc, thr)
        d, ci = boot_diff(per, base_img)
        dropped = int((sc < thr).sum())
        sim[q] = per
        print(f'  drop lowest {int(q*100):2d}% (thr {thr:.3f}, {dropped:3d} cands): '
              f'PQ {PQ(per.sum(0)):.4f}  d={PQ(per.sum(0)) - PQ(base_img.sum(0)):+.4f}  '
              f'paired CI [{ci[0]:+.4f}, {ci[1]:+.4f}]')

    grid = list(zip(qgrid, qs))
    gains = []
    for _ in range(300):
        perm = rng.permutation(N)
        A, B = perm[:N // 2], perm[N // 2:]
        q_best = max(grid, key=lambda t: PQ(sim[t[0]][A].sum(0)))[0]
        gains.append(PQ(sim[q_best][B].sum(0)) - PQ(base_img[B].sum(0)))
    gains = np.array(gains)
    best_gain = max(PQ(v.sum(0)) - PQ(base_img.sum(0)) for v in sim.values())
    print(f'  split-half CV (300 repeats): mean {gains.mean():+.4f}, median {np.median(gains):+.4f}, '
          f'P(gain>0) {np.mean(gains > 0):.0%}, 5th/95th pct {np.percentile(gains,5):+.4f}/{np.percentile(gains,95):+.4f}')
    auc_ = res[name][0]
    letter_pass = auc_ >= 0.70 and best_gain >= 0.005
    actionable = letter_pass and gains.mean() >= 0.005
    verdict = 'PASS' if actionable else ('MARGINAL PASS but NOT ACTIONABLE (out-of-sample gain < +0.005)' if letter_pass else 'STOP (falsified)')
    print(f'  VERDICT ({name}): {verdict}')
    summary[name] = {
        'auc': round(float(auc_), 4), 'auc_ci95': [round(float(res[name][1][0]), 4), round(float(res[name][1][1]), 4)],
        'best_in_sample_gain': round(float(best_gain), 4),
        'split_half_cv_gain_mean': round(float(gains.mean()), 4),
        'p_gain_positive': round(float(np.mean(gains > 0)), 3),
        'letter_of_rule_passed': bool(letter_pass), 'actionable': bool(actionable), 'verdict': verdict,
    }

json.dump({'n_images': N, 'n_candidates': len(rows), 'base_pq': round(PQ(base_img.sum(0)), 4), 'results': summary},
          open('analysis/results/stdviews_filter_summary.json', 'w'), indent=1)
print('\nsaved analysis/results/stdviews_filter_summary.json')
