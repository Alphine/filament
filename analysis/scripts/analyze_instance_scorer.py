"""Learned instance scorer on the out-of-fold instances (analysis/scripts/filament_oof_instances_cpu.py).

Target of an instance: y = mean over annotator passes of IoU * [IoU > 0.5] (its expected matched IoU).
Keeping an instance adds y to the PQ numerator and 0.5 to the denominator per pass, so it pays iff
y > PQ / 2. A gradient-boosted regressor predicts y from inference-time features; instances with a
prediction below tau are dropped and the official counts are recomputed exactly from the stored
per-pass IoUs (a dropped TP becomes an FN, a dropped FP disappears).

Rules fixed BEFORE the run:
  tau       = baseline out-of-fold PQ / 2 (no tuning; a tau grid is printed for information only);
  primary   out-of-fold instances (657 images), predictions from 5-fold CV grouped by image:
            adopt only if the gain is >= +0.005 with a paired bootstrap CI95 above 0;
  gate      a scorer fitted on all out-of-fold instances, applied to the hold-out ensemble's
            instances (single-model -> ensemble feature shift), must not lower the hold-out PQ.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from scorer_common import PARAMS, PQ, counts_after, cv_predict, feature_names, fit, paired_ci, table

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)

d = json.load(open('analysis/results/oof_instances/oof_instances.json'))
assert d['status'] == 'done'
feats = feature_names(d['results'])
data = {tag: table([r for r in d['results'] if r['set'] == tag], feats) for tag in ('oof', 'holdout')}


def report(D, pred, tau, label):
    a = counts_after(D, pred >= tau)
    diff, (lo, hi) = paired_ci(a, D['base'], rng)
    print(f'{label}: baseline {PQ(D["base"].sum(0)):.4f} -> {PQ(a.sum(0)):.4f} diff {diff:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  '
          f'dropped {int((pred < tau).sum())}/{len(pred)}  TP/FP/FN {D["base"].sum(0)[1:].astype(int).tolist()} -> {a.sum(0)[1:].astype(int).tolist()}')
    return {'baseline': round(PQ(D['base'].sum(0)), 4), 'scored': round(PQ(a.sum(0)), 4), 'diff': round(diff, 4),
            'ci95': [round(lo, 4), round(hi, 4)], 'dropped': int((pred < tau).sum()), 'n': len(pred)}


O, H = data['oof'], data['holdout']
tau = PQ(O['base'].sum(0)) / 2
print(f'{len(O["y"])} out-of-fold instances on {len(O["res"])} images, {len(H["y"])} hold-out instances; {len(feats)} features; tau = {tau:.4f}')
print(f'instances that are a TP in no pass: OOF {(O["y"] == 0).mean():.1%}; oracle (drop exactly those with y < tau): '
      f'{PQ(counts_after(O, O["y"] >= tau).sum(0)) - PQ(O["base"].sum(0)):+.4f}')
oof_pred = cv_predict(O)
auc = roc_auc_score(O['y'] > 0, oof_pred)
print(f'AUC (TP in any pass vs none), grouped CV: {auc:.3f}; correlation with y: {np.corrcoef(oof_pred, O["y"])[0, 1]:.3f}')
out = {'tau': round(tau, 4), 'auc': round(auc, 4), 'oof': report(O, oof_pred, tau, 'OOF (primary)  ')}
for t in (0.10, 0.15, 0.25, 0.30, 0.35):
    a = counts_after(O, oof_pred >= t)
    print(f'      tau {t:.2f}: diff {PQ(a.sum(0)) - PQ(O["base"].sum(0)):+.4f}  dropped {int((oof_pred < t).sum())}')
full = fit(O)
imp = sorted(zip(full.booster_.feature_importance('gain'), feats), reverse=True)
print('top features by gain:', ', '.join(f'{n} {g / sum(i for i, _ in imp):.0%}' for g, n in imp[:10]))
hp = full.predict(H['X'])
print(f'hold-out AUC {roc_auc_score(H["y"] > 0, hp):.3f}')
out['holdout'] = report(H, hp, tau, 'hold-out (gate)')
ok = out['oof']['diff'] >= 0.005 and out['oof']['ci95'][0] > 0 and out['holdout']['diff'] >= 0
out['verdict'] = 'ADOPT -> build the scorer into the pipeline, then one leaderboard probe' if ok else 'NOT ADOPTED'
print('\nVERDICT:', out['verdict'])
full.booster_.save_model('models/instance_scorer.txt')                # loaded by filament_scorer.InstanceScorer
json.dump({'features': feats, 'tau': out['tau'], 'fitted_on': f'{len(O["y"])} out-of-fold instances, {len(O["res"])} images',
           'lightgbm': lgb.__version__, 'params': PARAMS}, open('models/instance_scorer.json', 'w'), indent=1)
json.dump(out, open('analysis/results/instance_scorer_summary.json', 'w'), indent=1)
