"""Shared helpers of the instance-scorer studies (analyze_instance_scorer.py, analyze_loose_candidates.py)."""
import numpy as np

DROP = ('cx', 'cy')                      # absolute position: not used as a feature
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)


def PQ(t):
    t = np.asarray(t, float)
    d = t[1] + .5 * t[2] + .5 * t[3]
    return t[0] / d if d else 0.0


def feature_names(results):
    return [k for k in next(r['instances'][0]['feats'] for r in results if r['instances']) if k not in DROP]


def table(results, feats):
    """Instance table of a list of per-image records ({'instances': [...], 'counts': [...]}).
    y = mean over annotator passes of IoU * [IoU > 0.5]: the instance's expected matched IoU."""
    X, y, w, grp, iou = [], [], [], [], []
    for gi, r in enumerate(results):
        for it in r['instances']:
            v = np.array(it['iou'])
            X.append([it['feats'][k] for k in feats])
            y.append(float((v * (v > 0.5)).mean()))
            w.append(len(v))
            grp.append(gi)
            iou.append(v)
    return dict(res=results, X=np.array(X, float).reshape(-1, len(feats)), y=np.array(y), w=np.array(w, float),
                grp=np.array(grp, int), iou=iou, base=np.array([r['counts'] for r in results], float))


def counts_after(D, keep):
    """Per-image official counts when only the instances with keep[j] are submitted
    (a dropped TP becomes an FN, a dropped FP disappears)."""
    c = D['base'].copy()
    for j in np.flatnonzero(~keep):
        v, g = D['iou'][j], D['grp'][j]
        hit = v > 0.5
        c[g] += [-v[hit].sum(), -hit.sum(), -(~hit).sum(), hit.sum()]
    return c


def fit(D):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**PARAMS).fit(D['X'], D['y'], sample_weight=D['w'])


def cv_predict(D, folds=5):
    """Predictions for every instance from a model that saw no instance of the same image."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    pred = np.zeros(len(D['y']))
    for tr, va in GroupKFold(folds).split(D['X'], D['y'], D['grp']):
        m = lgb.LGBMRegressor(**PARAMS).fit(D['X'][tr], D['y'][tr], sample_weight=D['w'][tr])
        pred[va] = m.predict(D['X'][va])
    return pred


def paired_ci(a, b, rng, n_boot=4000):
    """PQ(a) - PQ(b) and its bootstrap CI95 over images; a, b: per-image counts in the same image order."""
    n = len(a)
    bs = [PQ(a[k].sum(0)) - PQ(b[k].sum(0)) for k in (rng.integers(0, n, n) for _ in range(n_boot))]
    return PQ(a.sum(0)) - PQ(b.sum(0)), np.percentile(bs, [2.5, 97.5])
