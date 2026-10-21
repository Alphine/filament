"""E1 (per-epoch patch re-sampling) compared with its baseline AFTER recalibration (path 1 of the 21 October plan).

The E1 gate (analyze_resample.py) compared the retrained models with the candidate thresholds, the refiner and the instance
scorer that were fitted to the maps of models trained on repeated patches: E1 found more true positives (+232) and many more
false positives (+1,212). Here both sets (EMA weights of the checkpoint-selection runs = "base", EMA weights of the
re-sampled runs = "e1") are scored on the same 557 out-of-fold images with the near-limb filter, and the instance scorer is
REFIT on each set's own instances (LightGBM with the released parameters, 5-fold cross-validation by image, 36 features),
with the threshold tau = (that set's near-limb PQ) / 2, fixed in advance exactly as the released tau was.

Rule fixed BEFORE the data exist (same bar as the E1 gate):
  adopt E1's retrained models only if, after the refit, PQ(e1) - PQ(base) >= +0.005 with a paired bootstrap CI95 above 0
  (images resampled together; 4000 resamples). Otherwise E1 stays rejected, and the report gets this recalibrated number.
Also reported (descriptive): both sets without the scorer, and the unrefit comparison (reproduces the E1 gate).
Input: analysis/results/recal/{e1,base}_instances.json (filament_recal_instances_cpu.py), same format as oof_instances.json.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'analysis/scripts')
from scorer_common import PQ, counts_after, cv_predict, feature_names, paired_ci, table   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
sets = {}
for tag in ('base', 'e1'):
    d = json.load(open(f'analysis/results/recal/{tag}_instances.json'))
    assert d['status'] == 'done', (tag, d['status'])
    sets[tag] = sorted(d['results'], key=lambda r: r['name'])
    print(f"{tag}: {len(sets[tag])} images, {sum(len(r['instances']) for r in sets[tag])} instances")
assert [r['name'] for r in sets['base']] == [r['name'] for r in sets['e1']] and len(sets['base']) == 557
feats = feature_names(sets['base'] + sets['e1'])
out = {}
per_image = {}
for tag, rs in sets.items():
    D = table(rs, feats)
    base_counts = D['base']                                    # near-limb only (stored counts, integrity-checked by the kernel)
    pq_nl = PQ(base_counts.sum(0))
    tau = pq_nl / 2
    pred = cv_predict(D)
    kept = counts_after(D, pred >= tau)
    per_image[tag] = {'nl': base_counts, 'sc': kept}
    out[tag] = {'pq_near_limb': round(pq_nl, 4), 'tau': round(tau, 4), 'pq_scorer_refit': round(PQ(kept.sum(0)), 4),
                'kept_share': round(float((pred >= tau).mean()), 3),
                'tp_fp_fn_nl': [int(x) for x in base_counts.sum(0)[1:]], 'tp_fp_fn_scorer': [int(x) for x in kept.sum(0)[1:]]}
    print(f"{tag:4s} near-limb PQ {pq_nl:.4f}  TP/FP/FN {base_counts.sum(0)[1:].astype(int).tolist()} | tau {tau:.4f} | "
          f"refit scorer PQ {PQ(kept.sum(0)):.4f}  TP/FP/FN {kept.sum(0)[1:].astype(int).tolist()}  kept {(pred >= tau).mean():.2f}")
res = {}
for label, key in (('without the scorer (near-limb only)', 'nl'), ('after refitting the scorer on each set', 'sc')):
    d, (lo, hi) = paired_ci(per_image['e1'][key], per_image['base'][key], rng)
    res[key] = (d, lo, hi)
    print(f"E1 minus base, {label}: {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")
d, lo, hi = res['sc']
verdict = ('ADOPT -> retrained models become candidates (ensemble-level hold-out check next)' if d >= 0.005 and lo > 0
           else 'NOT ADOPTED after recalibration')
print('VERDICT:', verdict)
out['diff_no_scorer'] = {'diff': round(res['nl'][0], 4), 'ci95': [round(res['nl'][1], 4), round(res['nl'][2], 4)]}
out['diff_refit_scorer'] = {'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
out['verdict'] = verdict
json.dump(out, open('analysis/results/recal_summary.json', 'w'), indent=1)
