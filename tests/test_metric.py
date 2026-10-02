"""filament_metric: the official PQ on tiny hand-made cases (no data, no weights)."""
import numpy as np
import pycocotools.mask as mu

import filament_metric as fm

H = W = 100


def ann(x0, y0, x1, y1):
    return {"segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}


def pred(x0, y0, x1, y1):
    m = np.zeros((H, W), np.uint8, order="F")
    m[y0:y1, x0:x1] = 1
    return mu.encode(m)["counts"].decode("ascii")


def test_perfect_match_is_one_tp():
    s, tp, fp, fn = fm.pq_counts([pred(10, 10, 50, 50)], {1: [ann(10, 10, 50, 50)]}, H, W)
    assert (tp, fp, fn) == (1, 0, 0) and s > 0.9


def test_iou_at_most_half_is_not_a_match():
    s, tp, fp, fn = fm.pq_counts([pred(10, 10, 50, 50)], {1: [ann(30, 10, 70, 50)]}, H, W)   # IoU = 1/3
    assert (s, tp, fp, fn) == (0, 0, 1, 1)


def test_each_annotator_pass_is_scored_separately():
    passes = {1: [ann(10, 10, 50, 50)], 2: []}      # the second pass marked nothing: the prediction is a FP there
    _, tp, fp, fn = fm.pq_counts([pred(10, 10, 50, 50)], passes, H, W)
    assert (tp, fp, fn) == (1, 1, 0)


def test_no_prediction_leaves_every_filament_a_miss():
    passes = {1: [ann(10, 10, 50, 50), ann(60, 60, 90, 90)], 2: [ann(10, 10, 50, 50)]}
    assert fm.pq_counts([], passes, H, W) == (0, 0, 0, 3)


def test_pq_formula():
    assert fm.pq(3.0, 4, 2, 2) == 3.0 / (4 + 1 + 1)
    assert fm.pq(0.0, 0, 0, 0) == 0.0


def test_holdout_split_is_fixed():
    names = [f"img{i:04d}.jpeg" for i in range(707)]
    a, b = fm.holdout_names(names), fm.holdout_names(list(reversed(names)))
    assert a == b and len(a) == 50 and len(set(a)) == 50 and set(a) <= set(names)
