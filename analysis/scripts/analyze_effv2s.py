"""Adoption gate of OUR EfficientNetV2-S stage-A (training/filament_effv2s_stagea.py; gate fixed in commit aaccff2).

Input: analysis/results/effv2s/effv2s_instances.json (filament_effv2s_instances_cpu.py) and the released pipeline's
analysis/results/oof_instances/oof_instances.json (already near-limb filtered).

Details fixed here BEFORE any PQ of the new maps was computed (10 November, fold 1 trained, no map scored):
  new set  = loose candidates of the new maps (p > 0.35 regions with peak > 0.5, area >= 100, mean p >= 0.5; no
             refiner), near-limb filter (centroid beyond 0.88 R dropped);
  old set  = the released pipeline's candidates (stage-A R34 + refiner, near-limb filter);
  each set: scorer refit on its own instances (released LightGBM parameters, 36 features, 5-fold CV by image),
            tau = half of that set's near-limb PQ.
  GATE 1 (657 out-of-fold images): PQ(new) - PQ(old) >= +0.005 with a paired bootstrap CI95 above 0 (4000 resamples).
  GATE 2 (50 hold-out images, only read if gate 1 passes): the 5-fold ensemble of the new maps, scored by the scorer
          fitted on all of the new set's out-of-fold instances at the same tau, must not score lower than the old set
          scored the same way.
Descriptive only: the public card's keep rule on the new maps (`strict`, `strict+nl`), without any scorer.
With fewer than five folds the script prints the same numbers on the images available, labelled PARTIAL: no verdict.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'analysis/scripts')
from scorer_common import PQ, counts_after, cv_predict, feature_names, fit, paired_ci, table   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
new = json.load(open('analysis/results/effv2s/effv2s_instances.json'))
partial = new['status'] != 'done'
old = json.load(open('analysis/results/oof_instances/oof_instances.json'))['results']


def pick(rs, tag, keep=None):
    return sorted((r for r in rs if r['set'] == tag and (keep is None or r['name'] in keep)), key=lambda r: r['name'])


n_oof = pick(new['results'], 'oof')
o_oof = pick(old, 'oof', {r['name'] for r in n_oof})
assert [r['name'] for r in n_oof] == [r['name'] for r in o_oof] and (partial or len(n_oof) == 657)
print(f"{'PARTIAL (folds ' + str(new['folds']) + ')' if partial else 'all five folds'}: {len(n_oof)} out-of-fold images")
out = {'status': new['status'], 'folds': new['folds'], 'n_oof': len(n_oof)}

# descriptive: the card's keep rule on the new maps
feats = feature_names(n_oof)
assert len(feats) == 36 and feats == feature_names(o_oof)
Dall = table(n_oof, feats)
inst = [it for r in n_oof for it in r['instances']]
strict = np.array([it['strict'] for it in inst])
inner = np.array([it['feats']['dist_r'] <= 0.88 for it in inst])
for k, keep in (('loose', np.ones(len(inst), bool)), ('strict', strict), ('strict+nl', strict & inner), ('loose+nl', inner)):
    s = counts_after(Dall, keep).sum(0)
    out[k] = {'pq': round(PQ(s), 4), 'tp_fp_fn': [int(x) for x in s[1:]]}
    print(f"new maps, {k:10s} PQ {PQ(s):.4f}  TP/FP/FN {s[1:].astype(int).tolist()}")


def near_limb(rs):
    return [{**r, 'instances': [it for it in r['instances'] if it['feats']['dist_r'] <= 0.88]} for r in rs]


def score_set(tag, D, base):
    D['base'] = base
    tau = PQ(base.sum(0)) / 2
    pred = cv_predict(D)
    kept = counts_after(D, pred >= tau)
    out[tag] = {'pq_near_limb': round(PQ(base.sum(0)), 4), 'tau': round(tau, 4), 'pq_scorer_refit': round(PQ(kept.sum(0)), 4),
                'tp_fp_fn_scorer': [int(x) for x in kept.sum(0)[1:]]}
    print(f"{tag}: near-limb PQ {PQ(base.sum(0)):.4f} | tau {tau:.4f} | refit scorer PQ {PQ(kept.sum(0)):.4f}  "
          f"TP/FP/FN {kept.sum(0)[1:].astype(int).tolist()}")
    return kept, tau


Dn = table(near_limb(n_oof), feats)
cn, tau_n = score_set('new', Dn, counts_after(Dall, inner))
Do = table(o_oof, feats)
co, tau_o = score_set('old', Do, Do['base'])
d, (lo, hi) = paired_ci(cn, co, rng)
out['gate1'] = {'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
print(f"GATE 1  new - old: {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")
g1 = d >= 0.005 and lo > 0
if partial:
    out['verdict'] = 'PARTIAL: no verdict'
elif not g1:
    out['verdict'] = 'NOT ADOPTED: gate 1 not passed'
else:
    hn_all, ho = pick(new['results'], 'holdout'), pick(old, 'holdout')
    assert [r['name'] for r in hn_all] == [r['name'] for r in ho] and len(ho) == 50
    Hall = table(hn_all, feats)
    hin = np.array([it['feats']['dist_r'] <= 0.88 for r in hn_all for it in r['instances']])
    Hn = table(near_limb(hn_all), feats)
    Hn['base'] = counts_after(Hall, hin)
    Ho = table(ho, feats)
    hcn = counts_after(Hn, fit(Dn).predict(Hn['X']) >= tau_n)
    hco = counts_after(Ho, fit(Do).predict(Ho['X']) >= tau_o)
    pn, po = PQ(hcn.sum(0)), PQ(hco.sum(0))
    out['gate2'] = {'new': round(pn, 4), 'old': round(po, 4)}
    print(f"GATE 2  hold-out: new ensemble {pn:.4f}, old {po:.4f}")
    out['verdict'] = 'ADOPT: both gates passed' if pn >= po else 'NOT ADOPTED: lower on the hold-out images'
print('VERDICT:', out['verdict'])
json.dump(out, open('analysis/results/effv2s_summary.json', 'w'), indent=1)
