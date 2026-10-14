"""Rubric metrics of the final evaluation (Overview > Evaluation) for the released pipeline, computed from
analysis/results/rubric/rubric_raw.json (filament_rubric_cpu.py).

Definitions, from the organizers' self-evaluation notebook:
  pairwise IoU / Dice  per annotator-image, (n_gt x n_pred) matrices; distributions use the pairs with a score > 0
  one-to-many / many-to-one  per ground-truth filament and per prediction, the number of partners with IoU > 0
                             (0 partners = miss: 1:0 and 0:1)
  mean Dice  "torchmetrics.segmentation.DiceScore": its exact configuration is not stated. We report the Dice of the
             union masks (foreground) per annotator-image and per image (mean over passes), plus the macro
             (foreground + background) and micro (pixel accuracy) variants, so the reader can see the spread.
Two settings: near-limb filter only (the stored masks) and the released pipeline (near-limb + instance scorer; scorer
decisions: cross-validated by image for the 657 out-of-fold images, the released model for the 50 hold-out images).
"""
import json
import sys

import lightgbm as lgb
import numpy as np

sys.path.insert(0, 'analysis/scripts')
from scorer_common import PQ, cv_predict, feature_names, table   # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
THR = 0.197
raw = json.load(open('analysis/results/rubric/rubric_raw.json'))
assert raw['status'] == 'done' and len(raw['results']) == 707, (raw['status'], len(raw['results']))
R = {r['name']: r for r in raw['results']}
oi = json.load(open('analysis/results/oof_instances/oof_instances.json'))['results']
feats = feature_names(oi)

# keep decisions of the scorer, per image, in prediction order
keep, score = {}, {}
oof = [r for r in oi if r['set'] != 'holdout']
hold = [r for r in oi if r['set'] == 'holdout']
D = table(oof, feats)
pred = cv_predict(D)
k = 0
for r in oof:
    n = len(r['instances']); score[r['name']] = pred[k:k + n]; keep[r['name']] = pred[k:k + n] >= THR; k += n
bst = lgb.Booster(model_file='models/instance_scorer.txt')
for r in hold:
    X = np.array([[it['feats'][f] for f in feats] for it in r['instances']], float).reshape(-1, len(feats))
    score[r['name']] = bst.predict(X) if len(X) else np.zeros(0)
    keep[r['name']] = score[r['name']] >= THR

# alignment: stored prediction i is the i-th instance of oof_instances (same label-map order)
bad = 0
for r in oi:
    a = [it['feats']['area'] for it in r['instances']]
    b = R[r['name']]['passes'][0]['pred_area']
    bad += len(a) != len(b) or any(abs(x - y) > 1 for x, y in zip(a, b))
print(f'alignment of stored predictions with the instance table: {len(oi) - bad}/{len(oi)} images match')
assert bad == 0


def per_pass(name, sel, p):
    """quantities of one annotator-image for the predictions selected by `sel`"""
    g = np.asarray(p['gt_area'], float)
    pa = np.asarray(p['pred_area'], float)[sel]
    inter = np.asarray(p['inter'], float).reshape(len(g), len(sel))[:, sel]
    gu = float(p['gt_union'])
    pgtu = np.asarray(p['pred_gtu'], float)[sel]
    n = float(R[name]['h'] * R[name]['w'])
    tp = pgtu.sum()
    fp, fn = pa.sum() - tp, gu - tp
    tn = n - tp - fp - fn
    d = lambda a, b, c: 1.0 if (2 * a + b + c) == 0 else 2 * a / (2 * a + b + c)
    union = g[:, None] + pa[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dice = np.divide(2 * inter, g[:, None] + pa[None, :], out=np.zeros_like(inter), where=(g[:, None] + pa[None, :]) > 0)
    hit = iou > 0
    gt_deg = hit.sum(1) if pa.size else np.zeros(len(g), int)
    pr_deg = hit.sum(0) if len(g) else np.zeros(len(pa), int)
    h5 = iou > 0.5
    return dict(dice_fg=d(tp, fp, fn), dice_bg=d(tn, fn, fp), acc=(tp + tn) / n,
                iou=iou[hit], dice=dice[hit], gt_deg=gt_deg, pr_deg=pr_deg,
                pq=np.array([iou[h5].sum(), h5.sum(), (h5.sum(0) == 0).sum() if len(g) else len(pa),
                             (h5.sum(1) == 0).sum() if len(pa) else len(g)], float))


def evaluate(names, use_scorer):
    ann, per_img = [], []
    ious, dices, gdeg, pdeg, pq = [], [], [], [], np.zeros(4)
    for nm in names:
        rr = R[nm]
        sel = keep[nm] if use_scorer else np.ones(rr['n_pred'], bool)
        sel = np.asarray(sel, bool)
        rows = [per_pass(nm, sel, p) for p in rr['passes']]
        ann += [(x['dice_fg'], x['dice_bg'], x['acc']) for x in rows]
        per_img.append(np.mean([(x['dice_fg'], x['dice_bg'], x['acc']) for x in rows], axis=0))
        for x in rows:
            ious.append(x['iou']); dices.append(x['dice']); gdeg.append(x['gt_deg']); pdeg.append(x['pr_deg']); pq += x['pq']
    return dict(ann=np.array(ann), img=np.array(per_img), iou=np.concatenate(ious), dice=np.concatenate(dices),
                gt_deg=np.concatenate(gdeg), pr_deg=np.concatenate(pdeg), pq=pq)


def boot_mean(x, n=3000):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def hist(deg, label_gt):
    n = len(deg)
    c = [int((deg == 0).sum()), int((deg == 1).sum()), int((deg == 2).sum()), int((deg >= 3).sum())]
    lab = ['1:0 (GT missed)', '1:1', '1:2', '1:3+'] if label_gt else ['0:1 (false positive)', '1:1', '2:1', '3+:1']
    return {l: [v, round(100 * v / n, 1)] for l, v in zip(lab, c)}, n


out = {}
for setting, use in (('near-limb only', False), ('released pipeline (near-limb + scorer)', True)):
    for grp, names in (('out-of-fold 657', [r['name'] for r in oof]), ('hold-out 50', [r['name'] for r in hold])):
        E = evaluate(names, use)
        print(f"\n=== {setting} | {grp} ({len(E['ann'])} annotator-images, {len(names)} images)")
        print(f"  PQ {PQ(E['pq']):.4f}  (TP/FP/FN {E['pq'][1]:.0f}/{E['pq'][2]:.0f}/{E['pq'][3]:.0f})")
        lo, hi = boot_mean(E['img'][:, 0])
        print(f"  mean Dice, foreground, per annotator-image {E['ann'][:, 0].mean():.4f} | per image {E['img'][:, 0].mean():.4f} "
              f"CI95 [{lo:.4f}, {hi:.4f}] | median {np.median(E['img'][:, 0]):.4f}")
        print(f"  mean Dice, macro fg+bg {((E['ann'][:, 0] + E['ann'][:, 1]) / 2).mean():.4f} | micro / pixel accuracy {E['ann'][:, 2].mean():.4f}")
        q = lambda a: np.percentile(a, [10, 25, 50, 75, 90])
        print(f"  pairwise IoU>0: n {len(E['iou'])}, mean {E['iou'].mean():.3f}, quartiles {np.round(q(E['iou'])[[1, 2, 3]], 3).tolist()}, share > 0.5 {np.mean(E['iou'] > .5):.2f}")
        print(f"  pairwise Dice>0: n {len(E['dice'])}, mean {E['dice'].mean():.3f}, quartiles {np.round(q(E['dice'])[[1, 2, 3]], 3).tolist()}")
        gh, ng = hist(E['gt_deg'], True); ph, npr = hist(E['pr_deg'], False)
        print(f"  one-to-many, per GT filament (n {ng}): " + ', '.join(f'{k} {v[0]} ({v[1]}%)' for k, v in gh.items()))
        print(f"  many-to-one, per prediction (n {npr}): " + ', '.join(f'{k} {v[0]} ({v[1]}%)' for k, v in ph.items()))
        out[f'{setting} | {grp}'] = {
            'pq': round(PQ(E['pq']), 4), 'n_annotator_images': len(E['ann']), 'n_images': len(names),
            'dice_fg_per_annotator_image': round(float(E['ann'][:, 0].mean()), 4),
            'dice_fg_per_image': round(float(E['img'][:, 0].mean()), 4), 'dice_fg_per_image_ci95': [round(float(lo), 4), round(float(hi), 4)],
            'dice_macro': round(float(((E['ann'][:, 0] + E['ann'][:, 1]) / 2).mean()), 4), 'pixel_accuracy': round(float(E['ann'][:, 2].mean()), 4),
            'pairwise_iou': {'n': int(len(E['iou'])), 'mean': round(float(E['iou'].mean()), 4), 'quartiles': np.round(q(E['iou'])[[1, 2, 3]], 4).tolist(),
                             'hist10': np.histogram(E['iou'], bins=10, range=(0, 1))[0].tolist()},
            'pairwise_dice': {'n': int(len(E['dice'])), 'mean': round(float(E['dice'].mean()), 4), 'quartiles': np.round(q(E['dice'])[[1, 2, 3]], 4).tolist(),
                              'hist10': np.histogram(E['dice'], bins=10, range=(0, 1))[0].tolist()},
            'one_to_many_per_gt': gh, 'many_to_one_per_pred': ph}

# paired: what the scorer does to the image-level Dice (the rubric weighs Dice and PQ), and a threshold sweep
print('\n=== scorer effect on the per-image foreground Dice (paired, bootstrap over images)')
for grp, names in (('out-of-fold 657', [r['name'] for r in oof]), ('hold-out 50', [r['name'] for r in hold])):
    a = evaluate(names, True)['img'][:, 0]
    b = evaluate(names, False)['img'][:, 0]
    d = a - b
    lo, hi = boot_mean(d)
    print(f'  {grp}: {b.mean():.4f} -> {a.mean():.4f}  diff {d.mean():+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]  (images lower {np.mean(d < -1e-9):.2f}, higher {np.mean(d > 1e-9):.2f})')
    out[f'scorer_dice_effect | {grp}'] = {'without': round(float(b.mean()), 4), 'with': round(float(a.mean()), 4),
                                         'diff': round(float(d.mean()), 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)]}

print('\n=== threshold sweep of the scorer, out-of-fold 657 (released threshold 0.197; "none" keeps everything)')
names = [r['name'] for r in oof]
sweep = {}
for thr in (0.0, 0.05, 0.10, 0.15, 0.197, 0.25, 0.30):
    saved = {k: v.copy() for k, v in keep.items()}
    for nm in names:
        keep[nm] = score[nm] >= thr
    E = evaluate(names, True)
    keep.update(saved)
    kept = np.mean(np.concatenate([score[nm] >= thr for nm in names]))
    sweep[str(thr)] = {'pq': round(PQ(E['pq']), 4), 'dice_img': round(float(E['img'][:, 0].mean()), 4), 'kept_share': round(float(kept), 3)}
    print(f"  thr {thr:5.3f}: PQ {PQ(E['pq']):.4f}  Dice(per image) {E['img'][:, 0].mean():.4f}  kept {kept:.2f}")
out['threshold_sweep_oof'] = sweep
json.dump(out, open('analysis/results/rubric_summary.json', 'w'), indent=1)
