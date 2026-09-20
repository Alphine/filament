"""CPU falsification test: can a light classifier predict which candidates are
supported by >= 2 annotator passes, and would dropping the unsupported ones raise PQ?

Motivation. A leave-one-pass-out simulation on the annotation files showed that
predicting only objects marked by at least two passes beats the pixel-union
target by +0.0145 PQ, and an oracle that drops candidates hit by fewer than two
passes gains about +0.03 on the multi-pass hold-out images. That gain is only
usable if a model can predict "supported by >= 2 passes" from the image itself.

Data: analysis/results/fpdiag (final pipeline, 50 hold-out images, per-candidate
features and exact per-pass hit pairs). No model is run.

Protocol
  * label = number of annotator passes in which the candidate's refined mask is a
    true positive (IoU > 0.5); "supported" = at least 2 (multi-pass images only).
  * scores for multi-pass images: repeated image-grouped cross-validation (a
    candidate is scored by models that never saw its image).
  * scores for single-pass images: a model fitted on all multi-pass candidates.
  * PQ simulation is exact (dropping a candidate changes TP/FP/FN of every pass).
  * The threshold is chosen inside a split-half cross-validation so the reported
    gain is out-of-sample.
Stop rule set beforehand: AUC < 0.70 or best simulated gain < +0.005 -> stop.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
    P = len(r['passes'])
    feats = {c['id']: c['feats'] for c in r['cands']}
    for j, cid in enumerate(ids):
        nh = sum(any(x[0] == j for x in p['pairs']) for p in r['passes'])
        rows.append(dict(img=gi, cid=cid, P=P, nh=nh, f=feats[cid]))
fn_ = sorted(rows[0]['f'].keys())
X = np.array([[r['f'][k] for k in fn_] for r in rows])
img_of = np.array([r['img'] for r in rows])
P_of = np.array([r['P'] for r in rows])
nh = np.array([r['nh'] for r in rows])
multi = P_of >= 2
y = (nh >= 2).astype(int)
print(f'images {N} (multi-pass {len({r["img"] for r in rows if r["P"] >= 2})}), candidates {len(rows)}, '
      f'on multi-pass images {int(multi.sum())}')
print('supported (>=2 passes) among multi-pass candidates:', int(y[multi].sum()), '/', int(multi.sum()))
print('by (passes, hits):', {k: int(((P_of == k[0]) & (nh == k[1])).sum()) for k in sorted({(a, b) for a, b in zip(P_of, nh)}) if k[0] >= 2})

Xm, ym, gm = X[multi], y[multi], img_of[multi]
groups = np.unique(gm)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def make_gb():
    return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


def oof_scores(make, reps=20, k=5):
    acc = np.zeros(len(ym))
    cnt = np.zeros(len(ym))
    for _ in range(reps):
        perm = rng.permutation(groups)
        folds = np.array_split(perm, k)
        for f in folds:
            te = np.isin(gm, f)
            if ym[~te].min() == ym[~te].max():
                continue
            m = make().fit(Xm[~te], ym[~te])
            acc[te] += m.predict_proba(Xm[te])[:, 1]
            cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def auc_ci(score, n=2000):
    a = roc_auc_score(ym, score)
    bs = []
    for _ in range(n):
        pick = rng.choice(groups, len(groups))
        idx = np.concatenate([np.flatnonzero(gm == g) for g in pick])
        if ym[idx].min() == ym[idx].max():
            continue
        bs.append(roc_auc_score(ym[idx], score[idx]))
    return a, np.percentile(bs, [2.5, 97.5])


print('\n== AUC for "supported by >= 2 passes" (multi-pass images, image-grouped CV) ==')
res = {}
sc = {}
for name, mk in (('logistic', make_lr), ('GBM', make_gb)):
    sc[name] = oof_scores(mk)
    a, ci = auc_ci(sc[name])
    res[name] = (a, ci)
    print(f'  {name:9s} AUC {a:.3f}  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]')
best_single = None
for k, col in zip(fn_, Xm.T):
    a = roc_auc_score(ym, col)
    a = max(a, 1 - a)
    if best_single is None or a > best_single[1]:
        best_single = (k, a)
a_ms, ci_ms = auc_ci(Xm[:, fn_.index('mean_p')])
print(f'  single feature mean_p (already used by the min_mean_prob filter): AUC {a_ms:.3f}  CI95 [{ci_ms[0]:.3f}, {ci_ms[1]:.3f}]')
print(f'  best single feature overall: {best_single[0]} AUC {best_single[1]:.3f} (chosen post hoc, optimistic)')

# ------------------------------------------------------------------ PQ simulation
base_img = np.array([stats(imgs[g], {r['cid'] for r in rows if r['img'] == g}) for g in range(N)])
print(f'\nbase PQ over 50 images {PQ(base_img.sum(0)):.4f}')

lr_all = make_lr().fit(Xm, ym)
score_all = np.zeros(len(rows))
score_all[multi] = sc['logistic']
score_all[~multi] = lr_all.predict_proba(X[~multi])[:, 1]


def pq_with_drop(score, thr, apply_to='all'):
    per = []
    for g in range(N):
        keep = {r['cid'] for k, r in enumerate(rows) if r['img'] == g and not (score[k] < thr and (apply_to == 'all' or P_of[k] >= 2))}
        per.append(stats(imgs[g], keep))
    return np.array(per)


def boot_diff(a, b, n=2000):
    d = []
    for _ in range(n):
        k = rng.integers(0, N, N)
        d.append(PQ(a[k].sum(0)) - PQ(b[k].sum(0)))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


qs = np.quantile(score_all, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
print('\n== drop candidates scored below the quantile (score = logistic, exact PQ over all 50 images) ==')
sim = {}
for q, thr in zip((0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40), qs):
    per = pq_with_drop(score_all, thr)
    d, ci = boot_diff(per, base_img)
    dropped = int((score_all < thr).sum())
    sim[q] = per
    print(f'  drop lowest {int(q*100):2d}% (thr {thr:.3f}, {dropped:3d} cands): PQ {PQ(per.sum(0)):.4f}  d={PQ(per.sum(0)) - PQ(base_img.sum(0)):+.4f}  paired CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  TP {int(per[:, 1].sum())}')

# oracle: known support, multi-pass images only (upper bound, uses ground truth)
orc = []
for g in range(N):
    keep = {r['cid'] for r in rows if r['img'] == g and (r['P'] < 2 or r['nh'] >= 2)}
    orc.append(stats(imgs[g], keep))
orc = np.array(orc)
print(f'  ORACLE (drop candidates hit by < 2 passes on multi-pass images): PQ {PQ(orc.sum(0)):.4f}  d={PQ(orc.sum(0)) - PQ(base_img.sum(0)):+.4f}')

# split-half cross-validated threshold choice
grid = list(zip((0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40), qs))
gains = []
for _ in range(300):
    perm = rng.permutation(N)
    A, B = perm[:N // 2], perm[N // 2:]
    q_best = max(grid, key=lambda t: PQ(sim[t[0]][A].sum(0)))[0]
    gains.append(PQ(sim[q_best][B].sum(0)) - PQ(base_img[B].sum(0)))
gains = np.array(gains)
print(f'\nsplit-half CV (threshold chosen on one half, scored on the other, 300 repeats): '
      f'mean {gains.mean():+.4f}, median {np.median(gains):+.4f}, P(gain>0) {np.mean(gains > 0):.0%}, '
      f'5th/95th percentile {np.percentile(gains, 5):+.4f}/{np.percentile(gains, 95):+.4f}')

best_gain = max(PQ(v.sum(0)) - PQ(base_img.sum(0)) for v in sim.values())
auc_best = max(res['logistic'][0], res['GBM'][0])
print(f'\nSTOP RULE: best AUC {auc_best:.3f} (need >= 0.70), best in-sample simulated gain {best_gain:+.4f} '
      f'(need >= +0.005), out-of-sample mean gain {gains.mean():+.4f}')
letter_pass = auc_best >= 0.70 and best_gain >= 0.005
actionable = letter_pass and gains.mean() >= 0.005
if actionable:
    verdict = 'PASS'
elif letter_pass:
    verdict = ('MARGINAL PASS of the pre-set rule but NOT ACTIONABLE: the best threshold was picked from 7, its paired CI '
               'includes 0, and the out-of-sample gain is below +0.005')
else:
    verdict = 'STOP (falsified)'
print('VERDICT:', verdict)
json.dump({
    'n_images': N, 'n_candidates': len(rows), 'n_multipass_candidates': int(multi.sum()),
    'n_supported': int(y[multi].sum()),
    'auc': {k: {'auc': round(float(v[0]), 4), 'ci95': [round(float(v[1][0]), 4), round(float(v[1][1]), 4)]} for k, v in res.items()},
    'auc_mean_p': {'auc': round(float(a_ms), 4), 'ci95': [round(float(ci_ms[0]), 4), round(float(ci_ms[1]), 4)]},
    'base_pq': round(PQ(base_img.sum(0)), 4),
    'oracle_pq': round(PQ(orc.sum(0)), 4),
    'simulated_pq': {str(q): round(PQ(v.sum(0)), 4) for q, v in sim.items()},
    'split_half_cv_gain': {'mean': round(float(gains.mean()), 4), 'median': round(float(np.median(gains)), 4),
                           'p_gain_positive': round(float(np.mean(gains > 0)), 3)},
    'best_in_sample_gain': round(float(best_gain), 4),
    'letter_of_rule_passed': bool(letter_pass),
    'actionable': bool(actionable),
    'verdict': verdict,
}, open('analysis/results/support_classifier_summary.json', 'w'), indent=1)
