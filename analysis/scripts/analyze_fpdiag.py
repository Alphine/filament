import json, numpy as np, collections, sys
sys.stdout.reconfigure(encoding='utf-8')
B = 'analysis/results/fpdiag'
imgs = []
for i in range(4):
    d = json.load(open(f'{B}/holdout_chunk{i}.json'))
    assert d['status'] == 'done', (i, d['status'])
    imgs += d['results']
print('images', len(imgs), 'candidates', sum(len(r['cands']) for r in imgs))

def img_stats(r, keep=None):
    """Official (sumIoU, tp, fp, fn) of one image, summed over all passes."""
    ids = r['pred_cand_ids']
    kp = set(range(len(ids))) if keep is None else {j for j, c in enumerate(ids) if c in keep}
    S = tp = fp = fn = 0
    for p in r['passes']:
        pairs = [x for x in p['pairs'] if x[0] in kp]
        hit_p = {x[0] for x in pairs}; hit_g = {x[1] for x in pairs}
        S += sum(x[2] for x in pairs); tp += len(pairs)
        fp += len(kp) - len(hit_p); fn += p['n_gt'] - len(hit_g)
    return S, tp, fp, fn
def PQ(t): 
    S, tp, fp, fn = t; d = tp + .5*fp + .5*fn
    return S/d if d else 0
tot = np.sum([img_stats(r) for r in imgs], 0)
print('official PQ (from pairs)', round(PQ(tot),4), 'TP/FP/FN', tot[1:].astype(int).tolist(), '(reference 0.4325; TP=400 FP=239 FN=211 in the earlier run)')
px = np.sum([r['proxy'] for r in imgs], 0)
print('proxy PQ', round(px[0]/(px[1]+.5*px[2]+.5*px[3]),4), '(reference 0.42367 +-0.0016)')

# ---- 2. error categories
cat = collections.Counter()
for r in imgs:
    for p in r['passes']:
        for m in p['pred_max_iou']:
            if m > 0.5: cat['TP']+=1
            elif m > 0: cat['FP_localization(0<IoU<=0.5)']+=1
            else: cat['FP_pure(touches no GT)']+=1
        for m in p['gt_max_iou']:
            if m > 0.5: pass
            elif m > 0: cat['FN_localization']+=1
            else: cat['FN_pure(touched by no prediction)']+=1
print('\nBREAKDOWN:', dict(cat))
ious = [x[2] for r in imgs for p in r['passes'] for x in p['pairs']]
print('mean TP IoU', round(np.mean(ious),3), 'median', round(np.median(ious),3))
den = tot[1] + .5*tot[2] + .5*tot[3]
print('PQ ceiling: zero FP ->', round(tot[0]/(tot[1]+.5*tot[3]),4), '| zero FN ->', round(tot[0]/(tot[1]+.5*tot[2]),4),
      '| TP IoU=1 ->', round(tot[1]/den,4), '| FP halved ->', round(tot[0]/(tot[1]+.25*tot[2]+.5*tot[3]),4))

# ---- 3. bootstrap over images
rng = np.random.default_rng(0)
per = np.array([img_stats(r) for r in imgs])
bs = [PQ(per[rng.integers(0, len(per), len(per))].sum(0)) for _ in range(4000)]
print('\nBOOTSTRAP over images: PQ', round(PQ(tot),4), 'SE', round(np.std(bs),4), 'CI95', np.round(np.percentile(bs,[2.5,97.5]),4))

# ---- 4. candidate features
rows = []; grp = []; ylab = []; keys = []
for gi, r in enumerate(imgs):
    ids = r['pred_cand_ids']
    for c in r['cands']:
        if c['id'] not in ids: continue
        j = ids.index(c['id'])
        hits = np.mean([any(x[0]==j for x in p['pairs']) for p in r['passes']]) if r['passes'] else 0
        rows.append(c['feats']); grp.append(gi); ylab.append(int(hits >= 0.5)); keys.append((gi, c['id']))
fn_ = sorted(rows[0].keys()); X = np.array([[f[k] for k in fn_] for f in rows]); y = np.array(ylab); grp = np.array(grp)
print('\nlabelled candidates', len(y), 'TP(>=50% pass)', int(y.sum()), 'FP', int((1-y).sum()))
from sklearn.metrics import roc_auc_score
print('AUC per feature (0.5 = chance):')
for k, col in zip(fn_, X.T):
    a = roc_auc_score(y, col); print(f'  {k:14s} {a:.3f}' + ('  <- inverted' if a < .5 else ''))
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def oof(model_fn, seeds=1):
    o = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = model_fn(); m.fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:,1]
    return o
o_lr = oof(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)))
o_gb = oof(lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0))
print('OOF AUC (5-fold grouped by image): logistic', round(roc_auc_score(y,o_lr),3), '| GBM', round(roc_auc_score(y,o_gb),3))
# best single feature
best_single = max(fn_, key=lambda k: roc_auc_score(y, X[:, fn_.index(k)]) )
print('best single feature', best_single, round(roc_auc_score(y, X[:, fn_.index(best_single)]),3))

# ---- exact simulation of dropping candidates
def sim(score, thr):
    drop = {k for k, s in zip(keys, score) if s < thr}
    t = np.sum([img_stats(r, {c for (g,c) in [(gi,cid) for gi,cid in keys if gi==gi_] if (gi_,c) not in drop}) for gi_, r in enumerate(imgs)], 0)
    return t
print('\nSIMULATION: drop candidates with score < threshold (OOF score; exact official PQ):')
print('  base', round(PQ(tot),4))
for name, sc in (('logistic', o_lr), ('GBM', o_gb)):
    out = []
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
        t = sim(sc, thr); out.append(f'thr{thr}: {PQ(t):.4f} (dropped {int((sc<thr).sum())}, TP {int(t[1])})')
    print(' ', name, '|', ' | '.join(out))
# oracle: drop every candidate that is not a TP
t = sim(y.astype(float), 0.5); print('  ORACLE drop all non-TP ->', round(PQ(t),4))
