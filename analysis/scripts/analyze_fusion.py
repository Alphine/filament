"""Instance-level fusion of our EfficientNetV2-S stage-A (N) with the released ResNet34 + refiner pipeline (O).
CPU only, stored out-of-fold instances; helpers and the matching rule are in fusion_common.py.

Fixed before the first run (15 November):
  reference  = N alone, scorer refit by image-grouped CV, tau = 0.1885 (the configuration of the 0.38 probe; 0.4318).
  arms       = B: N masks, scorer with cross features of the O partner;
               C: B + the O (refined) mask replaces the N mask when its predicted matched IoU is higher;
               D: C + O instances without an N partner are added.
  tau        = 0.1885 in every arm (no tuning; a sweep is printed as context only).
  GATE 1     = 657 out-of-fold images: arm - reference >= +0.005 with a paired bootstrap CI95 above 0.
  GATE 2     = 50 hold-out images (ensembles of both families, models fitted on all out-of-fold instances): the arm
               must not score lower than the reference scored the same way.
  The arm submitted is the one with the highest out-of-fold PQ among those passing both gates.
Caveats: (i) the refiner behind O was trained on the pool images, so O masks are slightly optimistic out of fold (the
hold-out is clean); (ii) an added O instance could hit a ground-truth filament that a kept N instance already hits;
both would then cover more than half of it and would be centroid partners in nearly all cases; the number of
suspicious cases is printed and counted as false positives in the conservative row.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'analysis/scripts')
import fusion_common as fc                                              # noqa: E402
from scorer_common import PQ, feature_names, paired_ci                   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
TAU = 0.1885
new = json.load(open('analysis/results/effv2s/effv2s_instances.json'))['results']
old = json.load(open('analysis/results/oof_instances/oof_instances.json'))['results']
old = {r['name']: r for r in old}
out = {'tau': TAU}


def load(tag):
    rs = sorted((r for r in new if r['set'] == tag), key=lambda r: r['name'])
    imgs = [(fc.near_limb(r['instances']), fc.near_limb(old[r['name']]['instances'])) for r in rs]
    n_gt = np.array([r['counts'][1] + r['counts'][3] for r in rs], float)
    assert all(old[r['name']]['counts'][1] + old[r['name']]['counts'][3] == g for r, g in zip(rs, n_gt))
    return rs, imgs, n_gt


rs, imgs, n_gt = load('oof')
assert len(rs) == 657
feats = feature_names(rs)
assert len(feats) == 36
T, objs = fc.build(imgs, feats)
n = len(rs)
print(f"out of fold: {n} images, N {len(T['N']['grp'])}, O {len(T['O']['grp'])} instances; "
      f"partnered {sum(i >= 0 and j >= 0 for _, i, j in objs)}, N only {sum(j < 0 for _, i, j in objs)}, "
      f"O only {sum(i < 0 for _, i, j in objs)}")
# integrity: the O set alone must reproduce the stored counts of the released pipeline's instances
co = fc.counts(objs, ['O' if j >= 0 else None for _, i, j in objs], T, n_gt, n)
ref_o = np.array([old[r['name']]['counts'] for r in rs], float)
assert np.allclose(co, ref_o, atol=1e-6), 'O counts do not reproduce the stored ones'

p0 = fc.cv_predict(T['N'], n, cross=False)
pn = fc.cv_predict(T['N'], n, cross=True)
po = fc.cv_predict(T['O'], n, cross=True)
po0 = fc.cv_predict(T['O'], n, cross=False)
rng = np.random.default_rng(0)


def arms(p0, pn, po, tau):
    return {'ref': fc.decide(objs, p0, po, tau, use_o_mask=False, add_o=False),
            'B': fc.decide(objs, pn, po, tau, use_o_mask=False, add_o=False),
            'C': fc.decide(objs, pn, po, tau, use_o_mask=True, add_o=False),
            'D': fc.decide(objs, pn, po, tau, use_o_mask=True, add_o=True)}


A = arms(p0, pn, po, TAU)
C = {k: fc.counts(objs, v, T, n_gt, n) for k, v in A.items()}
C['O alone'] = fc.counts(objs, ['O' if j >= 0 and po0[j] >= 0.1972 else None for _, i, j in objs], T, n_gt, n)
for k, c in C.items():
    s = c.sum(0)
    line = f"{k:8s} PQ {PQ(s):.4f}  TP/FP/FN {s[1:].astype(int).tolist()}"
    out[k] = {'pq': round(PQ(s), 4), 'tp_fp_fn': [int(x) for x in s[1:]]}
    if k not in ('ref', 'O alone'):
        d, (lo, hi) = paired_ci(c, C['ref'], rng)
        out[k].update(diff=round(d, 4), ci95=[round(lo, 4), round(hi, 4)], gate1=bool(d >= 0.005 and lo > 0))
        line += f"   vs ref {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  gate 1 {'PASS' if out[k]['gate1'] else 'fail'}"
    print(line)
ch = A['D']
print('arm D submits', {s: sum(c == s for c in ch) for s in ('N', 'O')},
      '| of partnered kept, O mask chosen',
      f"{np.mean([c == 'O' for (g, i, j), c in zip(objs, ch) if i >= 0 and j >= 0 and c]):.1%}")

# conservative row: an added O-only hit in a pass where a kept N instance of the same image within 0.5 x length also hits
kept_n = {}
for (g, i, j), c in zip(objs, ch):
    if c == 'N':
        kept_n.setdefault(g, []).append(i)
cons, n_susp = C['D'].copy(), 0
flat_n = [it for a, _ in imgs for it in a]
flat_o = [it for _, b in imgs for it in b]
for (g, i, j), c in zip(objs, ch):
    if c == 'O' and i < 0:
        v, f = T['O']['iou'][j], flat_o[j]['feats']
        for p in np.flatnonzero(v > 0.5):
            for k in kept_n.get(g, []):
                h = flat_n[k]['feats']
                if T['N']['iou'][k][p] > 0.5 and np.hypot(f['cx'] - h['cx'], f['cy'] - h['cy']) < 0.5 * max(f['length'], h['length']):
                    cons[g] += [-v[p], -1, 1, 1]
                    n_susp += 1
                    break
d, (lo, hi) = paired_ci(cons, C['ref'], rng)
out['D conservative'] = {'pq': round(PQ(cons.sum(0)), 4), 'suspicious_hits': n_susp, 'diff': round(d, 4)}
print(f"D conservative (possible double hits {n_susp} counted as FP): PQ {PQ(cons.sum(0)):.4f}  vs ref {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

print('context only, tau sweep (ref / B / C / D):')
out['sweep'] = {}
for tau in (0.15, 0.17, 0.1885, 0.2, 0.21, 0.22, 0.24):
    v = [PQ(fc.counts(objs, a, T, n_gt, n).sum(0)) for a in arms(p0, pn, po, tau).values()]
    out['sweep'][str(tau)] = [round(x, 4) for x in v]
    print(f"  tau {tau:.4f}: " + ' / '.join(f"{x:.4f}" for x in v))

# gate 2: hold-out, ensembles of both families, models fitted on all out-of-fold instances
hrs, himgs, hgt = load('holdout')
assert len(hrs) == 50
H, hobjs = fc.build(himgs, feats)
m0, mn, mo = fc.fit(T['N'], cross=False), fc.fit(T['N']), fc.fit(T['O'])
h0, hn, ho = (m0.predict(fc.design(H['N'], False)), mn.predict(fc.design(H['N'])), mo.predict(fc.design(H['O'])))
HA = {'ref': fc.decide(hobjs, h0, ho, TAU, use_o_mask=False, add_o=False),
      'B': fc.decide(hobjs, hn, ho, TAU, use_o_mask=False, add_o=False),
      'C': fc.decide(hobjs, hn, ho, TAU, use_o_mask=True, add_o=False),
      'D': fc.decide(hobjs, hn, ho, TAU, use_o_mask=True, add_o=True)}
HC = {k: fc.counts(hobjs, v, H, hgt, 50) for k, v in HA.items()}
out['holdout'] = {}
for k, c in HC.items():
    s = c.sum(0)
    line = f"hold-out {k:4s} PQ {PQ(s):.4f}  TP/FP/FN {s[1:].astype(int).tolist()}"
    out['holdout'][k] = {'pq': round(PQ(s), 4), 'tp_fp_fn': [int(x) for x in s[1:]]}
    if k != 'ref':
        d, (lo, hi) = paired_ci(c, HC['ref'], rng)
        out['holdout'][k].update(diff=round(d, 4), ci95=[round(lo, 4), round(hi, 4)])
        out[k]['gate2'] = bool(d >= 0)
        line += f"   vs ref {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  gate 2 {'PASS' if d >= 0 else 'fail'}"
    print(line)
ok = [k for k in 'BCD' if out[k]['gate1'] and out[k]['gate2']]
out['verdict'] = (f"ADOPT arm {max(ok, key=lambda k: out[k]['pq'])}" if ok else 'NOT ADOPTED: no arm passes both gates')
print('VERDICT:', out['verdict'])
json.dump(out, open('analysis/results/fusion_summary.json', 'w'), indent=1)
