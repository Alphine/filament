"""Public third-party stage-A weights (dariushafshar/solar-filament-unet-weights, CC BY 4.0) with our post-processing,
out of fold (10 November). Input: analysis/results/pubw/pubw_instances_fold{0..4}.json (filament_pubw_oof_cpu.py).

Their card: OOF PQ 0.4502 on 707 images with the keep rule peak > 0.99, area >= 200, mean p >= 0.85 (public 0.39).
Ours: 0.4089 OOF with the scorer (657 images), public 0.37.

Variants, all on their out-of-fold maps (two models per image, their folds):
  strict      their keep rule (reproduction check of the card; TTA may differ)
  strict+nl   + our near-limb filter (centroid beyond 0.88 R dropped)
  scorer      loose candidates (peak > 0.5, area >= 100, mean p >= 0.5), near-limb filter, then our instance scorer
              REFIT on these instances (released LightGBM parameters, 36 features, 5-fold CV by image),
              tau = PQ(strict+nl) / 2, fixed in advance like the released tau.

Rules fixed BEFORE the data exist:
  R1 (does our post-processing add to their maps?) `scorer` is preferred over `strict` only if
     PQ(scorer) - PQ(strict) >= +0.005 with a paired bootstrap CI95 above 0 (707 images, 4000 resamples);
     otherwise the candidate built on these weights is `strict+nl` if that passes the same bar, else plain `strict`.
  R2 (is it better than the released pipeline?) on the 657 images that have our out-of-fold instances:
     PQ(best variant by R1) - PQ(ours: near-limb + released scorer, CV predictions) >= +0.005 with CI95 above 0.
     Caveat written here in advance: their maps average two models per image, ours one, so R2 favours them by an
     unknown amount; it decides only whether a leaderboard probe is worth one submission, not a claim in the report.
  Nothing is submitted by this script. Building a final submission on another participant's weights is the
  repository owner's decision and would have to be disclosed in the report.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'analysis/scripts')
from scorer_common import PQ, counts_after, cv_predict, feature_names, paired_ci, table   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
rs = []
for f in range(5):
    d = json.load(open(f'analysis/results/pubw/pubw_instances_fold{f}.json'))
    assert d['status'] == 'done', (f, d['status'])
    rs += d['results']
rs.sort(key=lambda r: r['name'])
assert len(rs) == 707 and len({r['name'] for r in rs}) == 707
feats = feature_names(rs)
assert len(feats) == 36, len(feats)
D = table(rs, feats)
inst = [it for r in rs for it in r['instances']]
strict = np.array([it['strict'] for it in inst])
inner = np.array([it['feats']['dist_r'] <= 0.88 for it in inst])
print(f"707 images, {len(inst)} loose instances, {strict.sum()} strict, {(strict & inner).sum()} strict inside 0.88 R")

C = {'loose': D['base'], 'strict': counts_after(D, strict), 'strict+nl': counts_after(D, strict & inner)}
tau = PQ(C['strict+nl'].sum(0)) / 2
# scorer refit on the instances it would see at inference time: the loose set after the near-limb filter
rs_in = [{**r, 'instances': [it for it in r['instances'] if it['feats']['dist_r'] <= 0.88]} for r in rs]
Din = table(rs_in, feats)
Din['base'] = counts_after(D, inner)
pred = cv_predict(Din)
C['scorer'] = counts_after(Din, pred >= tau)
out = {'tau': round(tau, 4), 'variants': {}}
for k, c in C.items():
    s = c.sum(0)
    out['variants'][k] = {'pq': round(PQ(s), 4), 'tp_fp_fn': [int(x) for x in s[1:]]}
    print(f"{k:10s} PQ {PQ(s):.4f}  TP/FP/FN {s[1:].astype(int).tolist()}")


def diff(a, b, label):
    d, (lo, hi) = paired_ci(a, b, rng)
    print(f"{label}: {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")
    out[label] = {'diff': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
    return d >= 0.005 and lo > 0


best = 'scorer' if diff(C['scorer'], C['strict'], 'scorer - strict') else \
    ('strict+nl' if diff(C['strict+nl'], C['strict'], 'strict+nl - strict') else 'strict')
print('R1 best variant on their maps:', best)

# R2: against the released pipeline on the common images
oi = json.load(open('analysis/results/oof_instances/oof_instances.json'))['results']
ours = sorted((r for r in oi if r['set'] == 'oof'), key=lambda r: r['name'])
Do = table(ours, feature_names(ours))
tau_o = json.load(open('models/instance_scorer.json'))['tau']
co = counts_after(Do, cv_predict(Do) >= tau_o)
idx = {r['name']: i for i, r in enumerate(rs)}
sel = [idx[r['name']] for r in ours]
print(f"ours (near-limb + scorer, CV) on {len(ours)} images: PQ {PQ(co.sum(0)):.4f}; theirs[{best}] same images: {PQ(C[best][sel].sum(0)):.4f}")
r2 = diff(C[best][sel], co, f'{best} - ours')
out['best'] = best
out['verdict'] = ('PROBE-WORTHY: candidate on public weights beats the released pipeline out of fold' if r2
                  else 'NOT BETTER than the released pipeline by the pre-registered bar')
print('VERDICT:', out['verdict'])
json.dump(out, open('analysis/results/pubw_summary.json', 'w'), indent=1)
