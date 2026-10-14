"""A1 of HANDOFF_TOP3_STRATEGY.md: is the 0.04 gap between out-of-fold PQ (0.409) and the public leaderboard (0.37)
explained by a measurable difference between the test images and the training images?

Reads stored numbers only (analysis/results/oof_instances, analysis/results/test_instances, models/); no images, no GPU.

Confound known before the run: the out-of-fold predictions come from ONE fold model per image, while hold-out and test
predictions come from the 5-model ensemble. Probability-derived features therefore differ for a reason unrelated to the
images. Hence three comparisons, fixed here before the run:

  (i)   test vs out-of-fold, GEOMETRY/IMAGE features only (probability features removed): the main comparison;
  (ii)  test vs out-of-fold, all features (expected to separate on the single-model/ensemble difference alone);
  (iii) test vs hold-out, all features (same stage-A type, 50 images: low power, a control).
Each is a LightGBM classifier cross-validated by image (GroupKFold 5); AUC is compared with 20 label permutations
(shuffled at image level).

Decision rules (fixed beforehand):
  * Shift visible to the pipeline: AUC (i) >= 0.65 and above every permutation. No shift: AUC (i) <= 0.55.
    Between the two: inconclusive.
  * Importance-weighted out-of-fold PQ (released pipeline with scorer, 0.4088 unweighted): weights from a logistic
    regression test-vs-train on image-level aggregates (year, site, mean distance to the limb centre, mean contrast,
    mean ring/inside brightness, mean instance area; no counts and no probabilities), clipped at 10.
    Gap "explained" if the weighted PQ <= 0.390 (at least half of the 0.039 gap), "not explained" if >= 0.400,
    otherwise "partly".
  * Scorer view (descriptive): mean predicted matched IoU and the share of instances kept at the scorer threshold,
    test vs hold-out vs out-of-fold (cross-validated predictions).
"""
import glob
import json
import re
import sys

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'analysis/scripts')
from scorer_common import PQ, counts_after, cv_predict, feature_names, table   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
PROB = {'mean_p', 'max_p', 'p10_p', 'p50_p', 'frac_p40', 'cand_mean_p', 'cand_p10_p', 'ring_p', 'img_mean_p', 'cand_iou'}
oi = json.load(open('analysis/results/oof_instances/oof_instances.json'))['results']
oof = [r for r in oi if r['set'] != 'holdout']
hold = [r for r in oi if r['set'] == 'holdout']
test = []
for p in sorted(glob.glob('analysis/results/test_instances/test_instances_chunk*.json')):
    for im in json.load(open(p))['images']:
        test.append({'name': im['stem'], 'instances': im['instances']})
print(f'images: out-of-fold {len(oof)}, hold-out {len(hold)}, test {len(test)}')
assert len(oof) == 657 and len(hold) == 50 and len(test) == 180
feats = feature_names(oi)
geo = [f for f in feats if f not in PROB]
print(f'{len(feats)} features, {len(geo)} without probability features')


def stem(n):
    return n.rsplit('.', 1)[0]


def mat(rs, names):
    X, g = [], []
    for i, r in enumerate(rs):
        for it in r['instances']:
            X.append([it['feats'][k] for k in names]); g.append(i)
    return np.array(X, float), np.array(g)


def adv_auc(A, B, names, label):
    XA, gA = mat(A, names); XB, gB = mat(B, names)
    X = np.vstack([XA, XB]); y = np.r_[np.zeros(len(XA)), np.ones(len(XB))]; g = np.r_[gA, gB + len(A)]
    n_img = len(A) + len(B)
    img_label = np.r_[np.zeros(len(A)), np.ones(len(B))]

    def run(yy):
        pred = np.zeros(len(yy))
        w = np.where(yy == 1, 0.5 * len(yy) / yy.sum(), 0.5 * len(yy) / (len(yy) - yy.sum()))
        for tr, va in GroupKFold(5).split(X, yy, g):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
            m.fit(X[tr], yy[tr], sample_weight=w[tr]); pred[va] = m.predict_proba(X[va])[:, 1]
        return roc_auc_score(yy, pred, sample_weight=w), pred

    auc, pred = run(y)
    nulls = []
    for _ in range(20):
        perm = rng.permutation(img_label)
        nulls.append(run(perm[g])[0])
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1).fit(X, y)
    imp = sorted(zip(m.booster_.feature_importance('gain'), names), reverse=True)[:5]
    print(f'{label}: AUC {auc:.3f} | permutations {np.min(nulls):.3f}..{np.max(nulls):.3f} | top gain: '
          + ', '.join(f'{n}' for _, n in imp))
    return {'auc': round(auc, 4), 'perm_min': round(min(nulls), 4), 'perm_max': round(max(nulls), 4), 'top': [n for _, n in imp]}


out = {'n_images': {'oof': len(oof), 'hold': len(hold), 'test': len(test)}}
out['adv_geo_test_vs_oof'] = adv_auc(oof, test, geo, '(i)   test vs out-of-fold, geometry/image features')
out['adv_all_test_vs_oof'] = adv_auc(oof, test, feats, '(ii)  test vs out-of-fold, all features')
out['adv_all_test_vs_hold'] = adv_auc(hold, test, feats, '(iii) test vs hold-out, all features')
a = out['adv_geo_test_vs_oof']['auc']
out['shift_verdict'] = ('SHIFT' if a >= 0.65 and a > out['adv_geo_test_vs_oof']['perm_max'] else
                        'NO SHIFT' if a <= 0.55 else 'INCONCLUSIVE')
print('shift verdict (i):', out['shift_verdict'])

# image-level aggregates and the importance-weighted out-of-fold PQ
SITES = sorted({stem(r['name'])[-2:] for r in oof + hold + test})


def agg(r):
    n = stem(r['name'])
    F = {k: np.mean([it['feats'][k] for it in r['instances']]) if r['instances'] else np.nan
         for k in ('dist_r', 'contrast', 'ring_mean', 'inside_mean', 'area')}
    return [int(n[:4])] + [float(n[-2:] == s) for s in SITES] + [F[k] for k in ('dist_r', 'contrast', 'ring_mean', 'inside_mean', 'area')]


A = np.array([agg(r) for r in oof + test], float)
yA = np.r_[np.zeros(len(oof)), np.ones(len(test))]
ok = ~np.isnan(A).any(1)
Z = StandardScaler().fit_transform(A[ok])
lr = LogisticRegression(C=0.5, max_iter=2000).fit(Z, yA[ok])
auc_img = roc_auc_score(yA[ok], lr.predict_proba(Z)[:, 1])
p = np.full(len(A), np.nan); p[ok] = lr.predict_proba(Z)[:, 1]
w_all = np.clip((p / (1 - p)) * (len(oof) / len(test)), 0, 10)
w = np.nan_to_num(w_all[:len(oof)], nan=1.0)
print(f'image-level test-vs-train logistic AUC (in-sample) {auc_img:.3f}; weights: mean {w.mean():.2f}, '
      f'effective n {w.sum() ** 2 / (w ** 2).sum():.0f} of {len(w)}')

Doof = table(oof, feats)
pred = cv_predict(Doof)
keep = pred >= 0.197
base = Doof['base']; kept = counts_after(Doof, keep)
print(f'integrity: out-of-fold PQ no scorer {PQ(base.sum(0)):.4f} (expected 0.3944), with scorer {PQ(kept.sum(0)):.4f} (expected ~0.4088)')


def wpq(c, ww):
    return PQ((c * ww[:, None]).sum(0))


boots = []
for _ in range(2000):
    k = rng.integers(0, len(oof), len(oof))
    boots.append(wpq(kept[k], w[k]))
lo, hi = np.percentile(boots, [2.5, 97.5])
wp = wpq(kept, w)
verdict = 'EXPLAINED' if wp <= 0.390 else 'NOT EXPLAINED' if wp >= 0.400 else 'PARTLY'
print(f'importance-weighted out-of-fold PQ (with scorer): {wp:.4f} CI95 [{lo:.4f}, {hi:.4f}] vs unweighted {PQ(kept.sum(0)):.4f} -> {verdict}')
out['weighted_oof'] = {'unweighted': round(PQ(kept.sum(0)), 4), 'weighted': round(wp, 4), 'ci95': [round(lo, 4), round(hi, 4)],
                       'effective_n': round(float(w.sum() ** 2 / (w ** 2).sum()), 1), 'verdict': verdict,
                       'image_level_auc': round(auc_img, 3)}

# descriptive: test year/site mixture and per-image aggregates against training
yrs = lambda rs: np.array([int(stem(r['name'])[:4]) for r in rs])
for nm, rs in (('out-of-fold', oof), ('hold-out', hold), ('test', test)):
    n_inst = np.array([len(r['instances']) for r in rs])
    ar = np.array([it['feats']['area'] for r in rs for it in r['instances']])
    print(f'{nm:11s} instances/image {n_inst.mean():.2f}, median area {np.median(ar):.0f}, years {yrs(rs).min()}-{yrs(rs).max()}, '
          f'share >=2019 {np.mean(yrs(rs) >= 2019):.2f}')

# scorer view
bst = lgb.Booster(model_file='models/instance_scorer.txt')
names = feats                      # the booster stores generic names (Column_N); its order is the table order
assert bst.num_feature() == len(names), (bst.num_feature(), len(names))
Xt, _ = mat(test, names); Xh, _ = mat(hold, names)
pt, ph = bst.predict(Xt), bst.predict(Xh)
print(f'scorer: mean predicted matched IoU / kept share at 0.197: test {pt.mean():.3f} / {np.mean(pt >= 0.197):.2f}; '
      f'hold-out {ph.mean():.3f} / {np.mean(ph >= 0.197):.2f}; out-of-fold (cv) {pred.mean():.3f} / {keep.mean():.2f}')
out['scorer_view'] = {'test': [round(float(pt.mean()), 3), round(float(np.mean(pt >= 0.197)), 3)],
                      'hold': [round(float(ph.mean()), 3), round(float(np.mean(ph >= 0.197)), 3)],
                      'oof_cv': [round(float(pred.mean()), 3), round(float(keep.mean()), 3)]}
json.dump(out, open('analysis/results/shift_a1_summary.json', 'w'), indent=1)
