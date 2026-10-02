"""filament_scorer: features, the keep/drop rule, and the stored model (no weights needed)."""
import json
import os

import cv2
import numpy as np
import pytest

import filament_pipeline as fp
import filament_scorer as fs

HERE = os.path.dirname(os.path.abspath(__file__))


def scene():
    gray = np.full((2048, 2048), 20, np.uint8)
    cv2.circle(gray, (1024, 1024), 900, 150, -1)
    lab = np.zeros((2048, 2048), np.int32)
    for i in range(1, 5):
        cv2.ellipse(lab, (500 + 220 * i, 700 + 80 * i), (60 + 8 * i, 7 + i), 15 * i, 0, 360, i, -1)
    gray[lab > 0] = 90
    prob = np.where(lab > 0, 0.9, 0.05).astype(np.float32)
    return gray, lab, prob


def test_quantise_is_one_over_255_steps():
    q = fs.quantise(np.array([0.0, 0.5, 0.999, 1.0], np.float32))
    assert np.allclose(q * 255, np.round(q * 255)) and q[0] == 0 and q[-1] == 1


def test_stored_model_reproduces_the_stored_scores():
    """Cross-version check: the text model, fitted with LightGBM 4.7.0, must score identically elsewhere."""
    fx = json.load(open(os.path.join(HERE, "fixtures", "scorer_fixture.json")))
    sc = fs.InstanceScorer.load()
    assert sc.features == fx["features"] and sc.tau == fx["tau"]
    assert np.allclose(sc.scores(fx["instances"]), fx["expected"], atol=1e-6)


def test_image_instances_features():
    gray, lab, prob = scene()
    inst, disk = fs.image_instances(gray, lab, lab, fs.quantise(prob))
    assert [r["id"] for r in inst] == [1, 2, 3, 4] and abs(disk[2] - 900) < 5
    sc = fs.InstanceScorer.load()
    for r in inst:
        assert set(sc.features) <= set(r["feats"])
        assert all(np.isfinite(r["feats"][k]) for k in sc.features)
    assert all(r["feats"]["n_inst"] == 4 for r in inst)


def test_filter_drops_exactly_the_instances_below_tau():
    gray, lab, prob = scene()
    sc = fs.InstanceScorer.load()
    inst, _ = fs.image_instances(gray, lab, lab, fs.quantise(prob))
    expect = {r["id"] for r, v in zip(inst, sc.scores(inst)) if v >= sc.tau}
    assert set(np.unique(sc.filter(gray, lab, lab, prob))) - {0} == expect
    keep_all = fs.InstanceScorer(sc.booster, sc.features, tau=-1.0)
    assert (keep_all.filter(gray, lab, lab, prob) == lab).all()
    drop_all = fs.InstanceScorer(sc.booster, sc.features, tau=10.0)
    assert not drop_all.filter(gray, lab, lab, prob).any()


def test_empty_label_map_is_left_alone():
    gray, _, prob = scene()
    empty = np.zeros((2048, 2048), np.int32)
    assert not fs.InstanceScorer.load().filter(gray, empty, empty, prob).any()


def test_refiner_probability_shape_and_bounds():
    smp = pytest.importorskip("segmentation_models_pytorch")
    import torch
    torch.manual_seed(0)
    net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).eval()
    gray, _, _ = scene()
    p, (x0, y0, x1, y1) = fp.refiner_prob(gray, [800, 700, 120, 40], net, torch.device("cpu"))
    assert p.shape == (fp.REFINE_CROP, fp.REFINE_CROP) and 0 <= p.min() and p.max() <= 1
    assert 0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 2048
