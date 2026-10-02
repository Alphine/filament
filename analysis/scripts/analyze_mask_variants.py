"""Per-instance mask-variant selection (analysis/scripts/filament_variants_cpu.py).

Every instance of the released predictions was redrawn in ten ways (refiner thresholds, mirrored crop,
seed-box padding, mean of four refiner passes, the unrefined stage-A candidate); the maximum IoU with the
ground truth is stored per variant and annotator pass. Official counts are recomputed exactly from these
(a pass matches an instance when its IoU is > 0.5; TP = matches, FP = submitted instances - TP,
FN = ground-truth filaments - TP; the released variant reproduces the official counts, checked below).

Stages:
  1. ORACLE   best variant per instance by its true expected matched IoU (an upper bound, not a method);
  2. SELECTOR a LightGBM regressor predicts the expected matched IoU of every (instance, variant) from the
              instance features of the released scorer plus variant geometry; the argmax variant is
              submitted, and instances whose best prediction is below tau = 0.1972 are dropped. 5-fold CV
              grouped by image; the reference is the released variant with the released-style scorer under
              the same folds.

Rules fixed BEFORE the run:
  adopt the selector only if, out of fold (657 images), selector + drop beats the reference by >= +0.005 PQ
  with a paired bootstrap CI95 above 0, AND on the hold-out (models fitted on all out-of-fold instances) it is
  not below the reference. The oracle is reported for context only: it does not decide anything.
"""
import glob
import json
import sys

import lightgbm as lgb
import numpy as np

from scorer_common import PARAMS, PQ, feature_names, paired_ci

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
TAU = 0.1972

# ---- load
imgs, variants, n_same, n_inst = [], None, 0, 0
for p in sorted(glob.glob('analysis/results/mask_variants/variants_chunk*.json')):
    c = json.load(open(p))
    assert c['status'] == 'done', p
    variants = c['variants']
    imgs += c['images']
    n_same, n_inst = n_same + c['prod_same_as_stored'], n_inst + c['n_instances']
imgs.sort(key=lambda r: r['name'])
V, PROD = len(variants), variants.index('prod')
oi = json.load(open('analysis/results/oof_instances/oof_instances.json'))
base_feats = feature_names(oi['results'])
feat_of = {(r['name'], it['id']): it['feats'] for r in oi['results'] for it in r['instances']}
print(f'{len(imgs)} images ({sum(r["set"] == "oof" for r in imgs)} out of fold, {sum(r["set"] == "holdout" for r in imgs)} hold-out), '
      f'{n_inst} instances; released mask equals the stored one on {n_same}/{n_inst}')


def counts(img, choice):
    """Official counts of one image when instance j is submitted as variant choice[j] (-1 = dropped)."""
    P = len(img['n_gt'])
    kept = [(r, c) for r, c in zip(img['instances'], choice) if c >= 0 and r['area'][variants[c]] > 0]
    if not kept:
        return np.array([0.0, 0, 0, sum(img['n_gt'])])
    ious = np.array([r['iou'][variants[c]] for r, c in kept])
    hit = ious > 0.5
    tp = int(hit.sum())
    return np.array([float((ious * hit).sum()), tp, len(kept) * P - tp, sum(img['n_gt']) - tp])


def total(sel, choices):
    return np.array([counts(im, choices[k]) for k, im in enumerate(imgs) if sel(im)])


is_oof, is_hold = (lambda im: im['set'] == 'oof'), (lambda im: im['set'] == 'holdout')
all_prod = [[PROD] * len(im['instances']) for im in imgs]
b_o, b_h = total(is_oof, all_prod), total(is_hold, all_prod)
off_o = np.array([im['official'] for im in imgs if is_oof(im)])
ok = int((np.abs(b_o - off_o) < 1e-6).all(1).sum())
print(f'released variant reproduces the official counts on {ok}/{len(off_o)} out-of-fold images '
      f'(pooled PQ {PQ(b_o.sum(0)):.4f} vs official {PQ(off_o.sum(0)):.4f})')

# ---- instance table
rows = []     # one per instance
for k, im in enumerate(imgs):
    P = len(im['n_gt'])
    for j, r in enumerate(im['instances']):
        iou = np.array([r['iou'][v] for v in variants])                        # (V, P)
        y = (iou * (iou > 0.5)).mean(1) if P else np.zeros(V)
        y = np.where(np.array([r['area'][v] for v in variants]) > 0, y, 0.0)
        rows.append({'img': k, 'j': j, 'y': y, 'w': max(P, 1), 'set': im['set'],
                     'x': [feat_of[(im['name'], r['id'])][f] for f in base_feats],
                     'var': [[r['area'][v], r['area'][v] / max(r['area']['prod'], 1), r['iou_prod'][v]] for v in variants]})
n = len(rows)
Y = np.array([r['y'] for r in rows])                                           # (n, V)
print(f'{n} instances; mean true y per variant: ' + ', '.join(f'{v} {Y[:, i].mean():.3f}' for i, v in enumerate(variants)))
img_idx = np.array([r['img'] for r in rows])


def apply(choice_of_row, drop_mask=None):
    ch = [[PROD] * len(im['instances']) for im in imgs]
    for i, r in enumerate(rows):
        ch[r['img']][r['j']] = -1 if (drop_mask is not None and drop_mask[i]) else int(choice_of_row[i])
    return ch


# ---- stage 1: oracle and fixed variants
print('\nORACLE / fixed variants (out of fold; context only):')
res = {'oracle': {}, 'fixed': {}}
for v in variants:
    t = total(is_oof, apply(np.full(n, variants.index(v))))
    print(f'   all instances as {v:5s}: PQ {PQ(t.sum(0)):.4f} ({PQ(t.sum(0)) - PQ(b_o.sum(0)):+.4f})')
    res['fixed'][v] = round(PQ(t.sum(0)) - PQ(b_o.sum(0)), 4)
best = Y.argmax(1)
best = np.where(Y[np.arange(n), best] > Y[:, PROD] + 1e-9, best, PROD)
orc = total(is_oof, apply(best))
orc_d = total(is_oof, apply(best, Y[np.arange(n), best] < TAU))
print(f'   ORACLE select:        PQ {PQ(orc.sum(0)):.4f} ({PQ(orc.sum(0)) - PQ(b_o.sum(0)):+.4f})  TP/FP/FN {orc.sum(0)[1:].astype(int).tolist()}'
      f'\n   ORACLE select + drop: PQ {PQ(orc_d.sum(0)):.4f} ({PQ(orc_d.sum(0)) - PQ(b_o.sum(0)):+.4f})'
      f'  (oracle drop alone on the released variant: '
      f'{PQ(total(is_oof, apply(np.full(n, PROD), Y[:, PROD] < TAU)).sum(0)) - PQ(b_o.sum(0)):+.4f})')
print('   share of instances whose oracle variant is not the released one: '
      f'{(best != PROD).mean():.1%}; by variant: ' + ', '.join(f'{v} {(best == i).mean():.1%}' for i, v in enumerate(variants)))
res['oracle'] = {'select': round(PQ(orc.sum(0)) - PQ(b_o.sum(0)), 4), 'select_drop': round(PQ(orc_d.sum(0)) - PQ(b_o.sum(0)), 4)}


# ---- stage 2: learned selector
def stack(ix):
    X = np.array([[*rows[i]['x'], v_i, *rows[i]['var'][v_i], rows[i]['var'][PROD][0]] for i in ix for v_i in range(V)], float)
    return X, Y[ix].reshape(-1), np.repeat([rows[i]['w'] for i in ix], V)


def fit_sel(ix):
    X, y, w = stack(ix)
    return lgb.LGBMRegressor(**PARAMS).fit(X, y, sample_weight=w)


def fit_ref(ix):
    X = np.array([rows[i]['x'] for i in ix], float)
    return lgb.LGBMRegressor(**PARAMS).fit(X, Y[ix, PROD], sample_weight=[rows[i]['w'] for i in ix])


def predict(sel_model, ref_model, ix):
    X, _, _ = stack(ix)
    ps = sel_model.predict(X).reshape(len(ix), V)
    pr = ref_model.predict(np.array([rows[i]['x'] for i in ix], float))
    return ps, pr


oof_ix = np.array([i for i, r in enumerate(rows) if r['set'] == 'oof'])
hold_ix = np.array([i for i, r in enumerate(rows) if r['set'] == 'holdout'])
names = sorted({imgs[k]['name'] for k in set(img_idx[oof_ix])})
fold_of = {nm: f % 5 for f, nm in zip(range(len(names)), np.random.default_rng(0).permutation(names))}
fold = np.array([fold_of[imgs[rows[i]['img']]['name']] if rows[i]['set'] == 'oof' else -1 for i in range(n)])
PS, PR = np.zeros((n, V)), np.zeros(n)
for f in range(5):
    tr, va = oof_ix[fold[oof_ix] != f], oof_ix[fold[oof_ix] == f]
    PS[va], PR[va] = predict(fit_sel(tr), fit_ref(tr), va)
sel_model, ref_model = fit_sel(oof_ix), fit_ref(oof_ix)
PS[hold_ix], PR[hold_ix] = predict(sel_model, ref_model, hold_ix)


def report(sel, ch_sel, ch_ref, label):
    a, r = total(sel, ch_sel), total(sel, ch_ref)
    d, (lo, hi) = paired_ci(a, r, rng)
    print(f'   {label:9s}: reference {PQ(r.sum(0)):.4f} -> selector {PQ(a.sum(0)):.4f} diff {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  '
          f'TP/FP/FN {r.sum(0)[1:].astype(int).tolist()} -> {a.sum(0)[1:].astype(int).tolist()}')
    return d, lo, hi


pick = PS.argmax(1)
drop_s, drop_r = PS.max(1) < TAU, PR < TAU
ch_sel, ch_ref = apply(pick, drop_s), apply(np.full(n, PROD), drop_r)
print(f'\nSELECTOR (tau {TAU}); the selector moves {(pick[oof_ix] != PROD).mean():.1%} of out-of-fold instances off the released variant; '
      f'drops {drop_s[oof_ix].mean():.1%} (reference scorer {drop_r[oof_ix].mean():.1%})')
d_o, lo_o, hi_o = report(is_oof, ch_sel, ch_ref, 'out of fold')
d_h, lo_h, hi_h = report(is_hold, ch_sel, ch_ref, 'hold-out')
d_s, _, _ = report(is_oof, apply(pick, drop_r), ch_ref, 'switch only')
print('   (switch only = selector variant, reference drops; information)')
ok = bool(d_o >= 0.005 and lo_o > 0 and d_h >= 0)
verdict = 'ADOPT -> add the selector to the pipeline and re-run the test set' if ok else 'NOT ADOPTED'
print('\nVERDICT:', verdict)
json.dump({'tau': TAU, 'n_instances': n_inst, 'prod_same_as_stored': n_same, 'oracle': res['oracle'], 'fixed_variant_diff': res['fixed'],
           'oof_diff': round(d_o, 4), 'oof_ci95': [round(lo_o, 4), round(hi_o, 4)], 'holdout_diff': round(d_h, 4),
           'holdout_ci95': [round(lo_h, 4), round(hi_h, 4)], 'switch_only_diff': round(d_s, 4), 'verdict': verdict},
          open('analysis/results/mask_variants_summary.json', 'w'), indent=1)
