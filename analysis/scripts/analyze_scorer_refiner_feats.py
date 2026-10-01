"""Instance scorer with refiner-confidence and mask-stability features
(analysis/scripts/filament_refiner_feats_cpu.py), against the released scorer.

The refiner features are joined to the out-of-fold instance table (oof_instances.json) by image name
and instance id. Both scorers are fitted the same way (scorer_common: 5-fold CV grouped by image out of
fold; a fit on all out-of-fold instances applied to the hold-out), with tau fixed at the released 0.1972.

Rules fixed BEFORE the run:
  adopt the extended scorer only if, out of fold (657 images), it beats the released feature set by
  >= +0.005 PQ with a paired bootstrap CI95 above 0, and is not below it on the hold-out.
  Adopting it means re-running the test set (the refiner features were not stored there).
"""
import glob
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from scorer_common import PQ, counts_after, cv_predict, feature_names, fit, paired_ci, table

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
TAU = 0.1972

d = json.load(open('analysis/results/oof_instances/oof_instances.json'))
ref, n_inst, n_same = {}, 0, 0
for p in sorted(glob.glob('analysis/results/refiner_feats/refiner_feats_chunk*.json')):
    c = json.load(open(p))
    assert c['status'] == 'done', p
    ref.update(c['features'])
    n_inst, n_same = n_inst + c['n_instances'], n_same + c['reproduced']
print(f'refiner masks reproduced on {n_same}/{n_inst} instances')
for r in d['results']:
    for it in r['instances']:
        it['feats'].update(ref[r['name']][str(it['id'])])
new_names = sorted(next(iter(next(iter(ref.values())).values())))
all_feats = feature_names(d['results'])
base_feats = [f for f in all_feats if f not in new_names]
print(f'{len(base_feats)} released features + {len(new_names)} refiner features: {", ".join(new_names)}')

out, scored = {'tau': TAU}, {}
for label, feats in (('released', base_feats), ('extended', all_feats)):
    O, H = (table([r for r in d['results'] if r['set'] == tag], feats) for tag in ('oof', 'holdout'))
    pred = cv_predict(O)
    model = fit(O)
    hp = model.predict(H['X'])
    scored[label] = {'oof': counts_after(O, pred >= TAU), 'holdout': counts_after(H, hp >= TAU)}
    a, h = scored[label]['oof'], scored[label]['holdout']
    print(f'== {label:8s}: AUC out of fold {roc_auc_score(O["y"] > 0, pred):.3f}, hold-out {roc_auc_score(H["y"] > 0, hp):.3f}; correlation with y {np.corrcoef(pred, O["y"])[0, 1]:.3f}\n'
          f'   out of fold {PQ(O["base"].sum(0)):.4f} -> {PQ(a.sum(0)):.4f} TP/FP/FN {a.sum(0)[1:].astype(int).tolist()} (dropped {int((pred < TAU).sum())})\n'
          f'   hold-out    {PQ(H["base"].sum(0)):.4f} -> {PQ(h.sum(0)):.4f} TP/FP/FN {h.sum(0)[1:].astype(int).tolist()} (dropped {int((hp < TAU).sum())})')
    out[label] = {'auc_oof': round(roc_auc_score(O['y'] > 0, pred), 4), 'oof': round(PQ(a.sum(0)), 4), 'holdout': round(PQ(h.sum(0)), 4)}
    if label == 'extended':
        imp = sorted(zip(model.booster_.feature_importance('gain'), feats), reverse=True)
        tot = sum(g for g, _ in imp)
        print('   top features by gain: ' + ', '.join(f'{n} {g / tot:.0%}' for g, n in imp[:10]))
        print('   tau grid (information only, vs released at 0.1972): ' + ', '.join(
            f'{t:.2f}: {PQ(counts_after(O, pred >= t).sum(0)) - PQ(scored["released"]["oof"].sum(0)):+.4f}' for t in (0.15, 0.20, 0.25, 0.30)))
d_o, ci_o = paired_ci(scored['extended']['oof'], scored['released']['oof'], rng)
d_h, ci_h = paired_ci(scored['extended']['holdout'], scored['released']['holdout'], rng)
ok = bool(d_o >= 0.005 and ci_o[0] > 0 and d_h >= 0)
print(f'\nextended - released: out of fold {d_o:+.4f} CI95 [{ci_o[0]:+.4f}, {ci_o[1]:+.4f}] | hold-out {d_h:+.4f} CI95 [{ci_h[0]:+.4f}, {ci_h[1]:+.4f}]')
out.update({'oof_diff': round(d_o, 4), 'oof_ci95': [round(ci_o[0], 4), round(ci_o[1], 4)], 'holdout_diff': round(d_h, 4),
            'holdout_ci95': [round(ci_h[0], 4), round(ci_h[1], 4)],
            'verdict': 'ADOPT -> re-run the test set with the refiner features' if ok else 'NOT ADOPTED'})
print('VERDICT:', out['verdict'])
json.dump(out, open('analysis/results/scorer_refiner_feats_summary.json', 'w'), indent=1)
