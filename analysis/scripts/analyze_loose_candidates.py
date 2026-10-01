"""Looser candidate thresholds followed by the instance scorer (analysis/scripts/filament_loose_candidates_cpu.py).

For every candidate configuration the scorer is refitted on that configuration's out-of-fold instances
(5-fold CV grouped by image, same model settings as the released scorer) and instances predicted below
tau are dropped; the official counts are recomputed exactly from the stored per-pass IoUs.

Rules fixed BEFORE the run:
  tau       = 0.1972, the released scorer's threshold (no tuning; a grid is printed for information);
  reference = the released configuration ("prod") with its scorer;
  adopt a looser configuration only if, out of fold (657 images), it beats the reference by >= +0.005
  with a paired bootstrap CI95 above 0, and on the hold-out (scorer fitted on all out-of-fold instances
  of that configuration) it is not below the reference.
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

results, configs = [], None
for p in sorted(glob.glob('analysis/results/loose_candidates/loose_chunk*.json')):
    d = json.load(open(p))
    assert d['status'] == 'done', p
    results += d['results']
    configs = d['configs']
results.sort(key=lambda r: r['name'])
sets = {tag: [r for r in results if r['set'] == tag] for tag in ('oof', 'holdout')}
assert len(sets['oof']) == 657 and len(sets['holdout']) == 50, {k: len(v) for k, v in sets.items()}
print(f'prod label maps identical to the stored ones on {sum(r["same_as_stored"] for r in results)}/{len(results)} images')

scored, out = {}, {'tau': TAU, 'configs': {}}
for cfg, thr in configs.items():
    view = {tag: [{'instances': r['configs'][cfg]['instances'], 'counts': r['configs'][cfg]['counts']} for r in rs] for tag, rs in sets.items()}
    feats = feature_names(view['oof'])
    O, H = table(view['oof'], feats), table(view['holdout'], feats)
    pred = cv_predict(O)
    hp = fit(O).predict(H['X'])
    scored[cfg] = {'oof': counts_after(O, pred >= TAU), 'holdout': counts_after(H, hp >= TAU), 'pred': pred, 'O': O}
    a, h = scored[cfg]['oof'], scored[cfg]['holdout']
    print(f'== {cfg:6s} (p > {thr[0]}, area >= {thr[1]}, mean p >= {thr[2]}): {len(O["y"])} out-of-fold instances, AUC {roc_auc_score(O["y"] > 0, pred):.3f}\n'
          f'   out of fold: unscored {PQ(O["base"].sum(0)):.4f} TP/FP/FN {O["base"].sum(0)[1:].astype(int).tolist()} -> '
          f'scored {PQ(a.sum(0)):.4f} {a.sum(0)[1:].astype(int).tolist()} (dropped {int((pred < TAU).sum())})\n'
          f'   hold-out:    unscored {PQ(H["base"].sum(0)):.4f} TP/FP/FN {H["base"].sum(0)[1:].astype(int).tolist()} -> '
          f'scored {PQ(h.sum(0)):.4f} {h.sum(0)[1:].astype(int).tolist()} (dropped {int((hp < TAU).sum())})')
    out['configs'][cfg] = {'thresholds': thr, 'oof_unscored': round(PQ(O['base'].sum(0)), 4), 'oof_scored': round(PQ(a.sum(0)), 4),
                           'holdout_unscored': round(PQ(H['base'].sum(0)), 4), 'holdout_scored': round(PQ(h.sum(0)), 4)}

print('\nagainst the reference (prod + scorer):')
best = None
for cfg in configs:
    if cfg == 'prod':
        continue
    d_o, ci_o = paired_ci(scored[cfg]['oof'], scored['prod']['oof'], rng)
    d_h, ci_h = paired_ci(scored[cfg]['holdout'], scored['prod']['holdout'], rng)
    ok = bool(d_o >= 0.005 and ci_o[0] > 0 and d_h >= 0)
    print(f'   {cfg:6s} out of fold {d_o:+.4f} CI95 [{ci_o[0]:+.4f}, {ci_o[1]:+.4f}] | hold-out {d_h:+.4f} CI95 [{ci_h[0]:+.4f}, {ci_h[1]:+.4f}] -> {"ADOPT" if ok else "not adopted"}')
    out['configs'][cfg].update({'oof_diff': round(d_o, 4), 'oof_ci95': [round(ci_o[0], 4), round(ci_o[1], 4)],
                                'holdout_diff': round(d_h, 4), 'holdout_ci95': [round(ci_h[0], 4), round(ci_h[1], 4)], 'adopt': ok})
    if ok and (best is None or d_o > out['configs'][best]['oof_diff']):
        best = cfg
    print('      tau grid (information only): ' + ', '.join(
        f'{t:.2f}: {PQ(counts_after(scored[cfg]["O"], scored[cfg]["pred"] >= t).sum(0)) - PQ(scored["prod"]["oof"].sum(0)):+.4f}' for t in (0.15, 0.20, 0.25, 0.30)))
out['verdict'] = f'ADOPT {best}' if best else 'NOT ADOPTED (no configuration beats prod + scorer by the pre-set margin)'
print('\nVERDICT:', out['verdict'])
json.dump(out, open('analysis/results/loose_candidates_summary.json', 'w'), indent=1)
