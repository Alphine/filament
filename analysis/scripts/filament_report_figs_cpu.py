"""
Solar Filament Segmentation 2026 -- qualitative figure of the report: four hold-out images with the instance scorer's
decisions, CPU-ONLY, no inference (the stored hold-out predictions are drawn).

The images were fixed by a rule before the pictures were looked at (analysis in analyze_rubric.py data; CASES below):
  (a) the image the scorer helps most, (b) the image it hurts most, (c) the lowest PQ with the scorer among images with
  >= 4 annotated filaments (the next one when it repeats (a) or (b)), (d) the image with the median PQ.
Top row: the annotator pass with the most instances, green = matched by a kept prediction (IoU > 0.5), yellow = missed.
Bottom row: kept predictions, cyan = matched, red = unmatched; orange = dropped by the instance scorer (tau 0.1972);
magenta = dropped by the near-limb filter. Per-image PQ pools the annotator passes (without -> with the scorer).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pycocotools.mask as mu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = {}      # filled by the kernel wrapper: {"tau": ..., "cases": [{"label","name","pq_no","pq","scores","n_pass"}]}
OUT = os.environ.get('OUT_DIR', '/kaggle/working')
COL = dict(match=(60, 230, 60), miss=(255, 210, 0), pred=(0, 220, 255), fp=(255, 60, 60), drop=(255, 140, 0), limb=(230, 0, 230))


def show_img(g):
    lo, hi = np.percentile(g, [1, 99.5])
    x = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) ** 0.8
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def outline(mask, img, color, th):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, th)


def rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))


def render_case(g, stored, dropped_ids, scores, gt_masks, tau):
    """-> (top, bottom, info). stored: label map before the near-limb filter; dropped_ids: ids removed by it."""
    ids = [i for i in np.unique(stored) if i > 0 and i not in dropped_ids]
    assert len(ids) == len(scores), (len(ids), len(scores))
    keep = [s >= tau for s in scores]
    pred = {i: stored == i for i in ids}
    gt_r = [rle(m) for m in gt_masks]
    iou = {}
    for i in ids:
        iou[i] = np.asarray(mu.iou([rle(pred[i])], gt_r, [0] * len(gt_r)), float)[0] if gt_r else np.zeros(0)
    kept_ids = [i for i, k in zip(ids, keep) if k]
    gt_hit = [any(iou[i][j] > 0.5 for i in kept_ids) for j in range(len(gt_masks))]
    base = show_img(g)
    top, bot = base.copy(), base.copy()
    for j, m in enumerate(gt_masks):
        outline(m, top, COL['match'] if gt_hit[j] else COL['miss'], 9)
    for i, k in zip(ids, keep):
        if k:
            outline(pred[i], bot, COL['pred'] if (iou[i] > 0.5).any() else COL['fp'], 9)
        else:
            outline(pred[i], bot, COL['drop'], 9)
    for i in dropped_ids:
        outline(stored == i, bot, COL['limb'], 9)
    return top, bot, dict(n_gt=len(gt_masks), n_gt_hit=int(sum(gt_hit)), n_kept=len(kept_ids), n_scorer_dropped=int(len(ids) - len(kept_ids)),
                          n_limb_dropped=len(dropped_ids))


def draw(panels, path):
    fig, axes = plt.subplots(2, len(panels), figsize=(7.2, 3.9))
    for c, (top, bot, title) in enumerate(panels):
        for row, im in enumerate((top, bot)):
            ax = axes[row, c]
            ax.imshow(cv2.resize(im, (800, 800), interpolation=cv2.INTER_AREA))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0, c].set_title(title, fontsize=7)
    axes[0, 0].set_ylabel('annotator pass', fontsize=7)
    axes[1, 0].set_ylabel('prediction', fontsize=7)
    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.03, right=0.995, top=0.92, bottom=0.005)
    plt.savefig(path, dpi=170)
    plt.close(fig)


def main():
    sys.path.insert(0, '.')
    import filament_pipeline as fp
    import filament_metric as fm
    INPUT = '/kaggle/input'
    ann = next(os.path.join(r, f) for r, _, fs in os.walk(INPUT) for f in fs if f.endswith('.json') and 'nnotation' in f)
    train_dir = next(r for r, _, _ in os.walk(INPUT) if os.path.basename(r) == 'train_images')
    per_file, dims = fm.load_annotations(ann, train_dir)
    stores = {}
    for p in glob.glob(f'{INPUT}/**/preds_chunk*.npz', recursive=True):
        z = np.load(p)
        for k in z.files:
            if k.endswith('__lab'):
                stores[k[:-5]] = (p, z)
    panels, meta = [], {}
    tau = CASES['tau']
    for c, case in enumerate(CASES['cases']):
        name = case['name']; stem = name.rsplit('.', 1)[0]
        g = fp.read_gray(os.path.join(train_dir, name))
        h, w = g.shape
        stored = stores[stem][1][stem + '__lab'].astype(np.int32)
        kept_lab = fp.drop_near_limb(stored, g)
        dropped = [i for i in np.unique(stored) if i > 0 and i not in set(np.unique(kept_lab).tolist())]
        passes = per_file[name]
        best = max(passes.values(), key=len)
        gt_masks = [mu.decode(r) for r in fm._gt_rles(best, h, w)]
        top, bot, info = render_case(g, stored, set(dropped), case['scores'], gt_masks, tau)
        meta[name] = {**info, 'label': case['label'], 'pq_no_scorer': round(case['pq_no'], 4), 'pq': round(case['pq'], 4), 'n_pass': case['n_pass']}
        print(name, meta[name], flush=True)
        panels.append((top, bot, f"({chr(97 + c)}) {name[:8]}.. PQ {case['pq_no']:.2f}->{case['pq']:.2f}"))
    draw(panels, f'{OUT}/fig_cases.png')
    json.dump(meta, open(f'{OUT}/cases_meta.json', 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
