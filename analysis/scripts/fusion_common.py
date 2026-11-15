"""Instance-level fusion of the two stage-A families (shared by analyze_fusion.py and make_fusion_submission.py).

N = loose candidates of our EfficientNetV2-S stage-A (no refiner); O = the released pipeline's instances (ResNet34
stage-A + crop refiner). Both near-limb filtered (centroid beyond 0.88 R dropped).

Cross-model matching uses centroids only, because the out-of-fold files store features and per-pass IoU but no masks:
an N and an O instance are partners if their centroid distance is below 0.35 x the longer of the two lengths, matched
one to one, closest first. On the 180 test images, where both masks exist, this rule agrees with mask IoU > 0.3 at
precision 0.985 / recall 0.996 (15,901 same-image pairs).

Each mask (N or O) gets its own 36 features plus cross features of its partner (NaN without partner); one LightGBM
per source predicts the mask's expected matched IoU (same target and parameters as the released scorer).
"""
import numpy as np

from scorer_common import PARAMS

LIMB_T = 0.88
MATCH_K = 0.35
CROSS = ('area', 'mean_p', 'p10_p', 'max_p', 'length', 'width', 'contrast', 'cand_mean_p', 'refined_ratio', 'solidity')


def near_limb(inst):
    return [it for it in inst if it['feats']['dist_r'] <= LIMB_T]


def match(a, b):
    """One-to-one partners between instance lists a and b: {index in a: index in b}."""
    pairs = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d = float(np.hypot(x['feats']['cx'] - y['feats']['cx'], x['feats']['cy'] - y['feats']['cy']))
            if d < MATCH_K * max(x['feats']['length'], y['feats']['length']):
                pairs.append((d, i, j))
    m, used = {}, set()
    for d, i, j in sorted(pairs):
        if i not in m and j not in used:
            m[i] = j
            used.add(j)
    return m


def cross_row(own, partner):
    """Cross features of a mask given its partner from the other family (None: no partner)."""
    if partner is None:
        return [0.0] + [np.nan] * (len(CROSS) + 3)
    f, g = own['feats'], partner['feats']
    d = float(np.hypot(f['cx'] - g['cx'], f['cy'] - g['cy']))
    return ([1.0, d / max(f['length'], g['length'], 1e-6), g['area'] / max(f['area'], 1.0),
             g['length'] / max(f['length'], 1e-6)] + [float(g[k]) for k in CROSS])


def n_other(own, others):
    """How many instances of the other family lie within 150 px (context when there is no partner)."""
    f = own['feats']
    return float(sum(np.hypot(f['cx'] - o['feats']['cx'], f['cy'] - o['feats']['cy']) < 150 for o in others))


def build(images, feats, labelled=True):
    """images: list of (N instances, O instances) per image, near-limb filtered.
    Returns per-source tables and the object list [(image, i_N or -1, j_O or -1)] with row indices into the tables."""
    T = {s: dict(X=[], Xc=[], y=[], w=[], grp=[], iou=[]) for s in 'NO'}
    objs = []
    for g, (a, b) in enumerate(images):
        m = match(a, b)
        inv = {j: i for i, j in m.items()}
        rn, ro = {}, {}
        for src, own, oth, part, rows in (('N', a, b, m, rn), ('O', b, a, inv, ro)):
            t = T[src]
            for i, it in enumerate(own):
                p = oth[part[i]] if i in part else None
                rows[i] = len(t['grp'])
                t['X'].append([it['feats'][k] for k in feats])
                t['Xc'].append(cross_row(it, p) + [n_other(it, oth), float(len(oth))])
                t['grp'].append(g)
                if labelled:
                    v = np.array(it['iou'])
                    t['y'].append(float((v * (v > 0.5)).mean()))
                    t['w'].append(len(v))
                    t['iou'].append(v)
        objs += [(g, rn[i], ro[m[i]]) for i in m]
        objs += [(g, rn[i], -1) for i in range(len(a)) if i not in m]
        objs += [(g, -1, ro[j]) for j in range(len(b)) if j not in inv]
    for t in T.values():
        t['X'] = np.array(t['X'], float).reshape(-1, len(feats))
        t['Xc'] = np.array(t['Xc'], float).reshape(len(t['grp']), -1)
        t['grp'] = np.array(t['grp'], int)
        t['y'], t['w'] = np.array(t['y']), np.array(t['w'], float)
    return T, objs


def design(t, cross=True):
    return np.hstack([t['X'], t['Xc']]) if cross else t['X']


def fit(t, cross=True, idx=None):
    import lightgbm as lgb
    X = design(t, cross)
    idx = np.arange(len(X)) if idx is None else idx
    return lgb.LGBMRegressor(**PARAMS).fit(X[idx], t['y'][idx], sample_weight=t['w'][idx])


def cv_predict(t, n_img, cross=True, folds=5):
    """Out-of-image predictions; the image split is the same for both sources (image index modulo a fixed shuffle)."""
    perm = np.random.default_rng(0).permutation(n_img)
    fold_of = np.empty(n_img, int)
    fold_of[perm] = np.arange(n_img) % folds
    f = fold_of[t['grp']]
    X, pred = design(t, cross), np.zeros(len(t['grp']))
    for k in range(folds):
        pred[f == k] = fit(t, cross, np.flatnonzero(f != k)).predict(X[f == k])
    return pred


def decide(objs, pn, po, tau, use_o_mask=True, add_o=True, bonus=0.0):
    """Per object: which mask is submitted ('N', 'O' or None).
    Partnered: the mask with the higher predicted matched IoU (N unless use_o_mask), kept if >= tau.
    N only: kept if >= tau. O only: kept if add_o and >= tau + bonus."""
    out = []
    for g, i, j in objs:
        if i >= 0 and j >= 0:
            if use_o_mask and po[j] > pn[i]:
                out.append('O' if po[j] >= tau else None)
            else:
                out.append('N' if pn[i] >= tau else None)
        elif i >= 0:
            out.append('N' if pn[i] >= tau else None)
        else:
            out.append('O' if add_o and po[j] >= tau + bonus else None)
    return out


def counts(objs, choice, T, n_gt, n_img):
    """Per-image [sum IoU, TP, FP, FN] of the submitted masks; n_gt = annotated filaments summed over passes."""
    c = np.zeros((n_img, 4))
    for (g, i, j), ch in zip(objs, choice):
        if ch is None:
            continue
        v = T['N']['iou'][i] if ch == 'N' else T['O']['iou'][j]
        hit = v > 0.5
        c[g] += [v[hit].sum(), hit.sum(), (~hit).sum(), 0]
    c[:, 3] = n_gt - c[:, 1]
    return c
