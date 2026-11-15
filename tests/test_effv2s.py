"""filament_effv2s: the final submission's instance rule, selection and stored scorer (no weights needed)."""
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch

import filament_effv2s as fe
import filament_pipeline as fp
import filament_scorer as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "scripts"))


def test_loose_instances_rule():
    p = np.zeros((400, 400), np.uint8)
    p[10:30, 10:30] = 200          # 400 px, kept
    p[100:105, 100:105] = 255      # 25 px: below the minimum area
    p[200:230, 200:230] = 120      # peak 0.47: no pixel above 0.5
    p[300:320, 300:330] = 95       # mean just above 0.35 ...
    p[300, 300] = 255              # ... with one peak: mean p < 0.5
    p[350:370, 350:380] = 140      # 600 px, kept
    lab = fe.loose_instances(p)
    assert lab.max() == 2
    assert (lab[10:30, 10:30] == 1).all() and (lab[350:370, 350:380] == 2).all()
    assert (lab > 0).sum() == 400 + 600


def test_loose_instances_are_8_connected():
    p = np.zeros((300, 300), np.uint8)
    p[10:25, 10:25] = 200
    p[25:40, 25:40] = 200          # touches the first square at one corner only
    assert fe.loose_instances(p).max() == 1


def scene():
    gray = np.full((2048, 2048), 20, np.uint8)
    cv2.circle(gray, (1024, 1024), 900, 150, -1)
    p = np.zeros((2048, 2048), np.uint8)
    for i in range(1, 5):
        cv2.ellipse(p, (500 + 220 * i, 700 + 80 * i), (60 + 8 * i, 7 + i), 15 * i, 0, 360, 230, -1)
    cv2.ellipse(p, (1024, 160), (50, 8), 0, 0, 360, 230, -1)          # near the limb: 0.96 R
    gray[p > 0] = 90
    return gray, p


def test_select_applies_limb_filter_and_scorer_and_renumbers():
    gray, p = scene()
    lab = fe.loose_instances(p)
    assert lab.max() == 5
    inner = fe.select(gray, lab, p, None, fp.LIMB_DROP_T)
    assert inner.max() == 4 and not inner[150:170].any()
    assert ((inner > 0) == ((lab > 0) & (lab != lab[160, 1024]))).all()
    sc = fs.InstanceScorer.load(name="effv2s_scorer")
    inst, _ = fs.image_instances(gray, lab, lab, p.astype(np.float32) / 255.0)
    expect = [r["id"] for r, v in zip(inst, sc.scores(inst)) if r["feats"]["dist_r"] <= fp.LIMB_DROP_T and v >= sc.tau]
    out = fe.select(gray, lab, p, sc, fp.LIMB_DROP_T)
    assert out.max() == len(expect)
    for k, i in enumerate(expect, 1):
        assert ((out == k) == (lab == i)).all()
    assert (fe.select(gray, lab, p, None, None) == lab).all()


def test_stored_scorer_rebuilds_the_submitted_csv():
    """models/effv2s_scorer.txt + the stored test instances give the 0.38 submission byte for byte."""
    import fit_effv2s_scorer as fit
    sc = fs.InstanceScorer.load(name="effv2s_scorer")
    assert len(sc.features) == 36 and abs(sc.tau - 0.1885) < 1e-4
    test = json.load(open(os.path.join(ROOT, "analysis", "results", "effv2s", "effv2s_test_instances.json")))
    data, n = fit.submission_bytes(sc, test["images"])
    assert n == 1048 and hashlib.sha256(data).hexdigest() == fit.SUBMITTED_SHA256


def test_predict_ensemble_arithmetic():
    """Views are undone correctly and the ensemble is the rounded mean of the per-model uint8 maps."""
    class Shift(torch.nn.Module):                     # not flip-equivariant: a wrong inverse view would show
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return torch.roll(x, 3, dims=3) * 2.0 + self.b

    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (64, 96), dtype=np.uint8)
    dev = torch.device("cpu")
    mean, std = fe.disk_stats(gray)
    z = (torch.from_numpy(gray).float() / 255.0 - mean) / std
    ref = torch.zeros(gray.shape)
    for k, flip in ((0, False), (0, True), (2, False), (2, True)):
        x = torch.flip(z, dims=[1]) if flip else z
        x = torch.rot90(x, k, dims=[0, 1])
        y = torch.sigmoid(torch.roll(x, 3, dims=1) * 2.0 + 0.3)
        y = torch.rot90(y, -k, dims=[0, 1])
        ref += torch.flip(y, dims=[1]) if flip else y
    one = fe.predict_model(Shift(0.3), gray, dev)
    assert one.dtype == np.uint8 and np.array_equal(one, (ref / 4 * 255).round().numpy().astype(np.uint8))
    two = fe.predict_model(Shift(-0.4), gray, dev)
    ens = fe.predict_ensemble([Shift(0.3), Shift(-0.4)], gray, dev)
    assert np.array_equal(ens, np.round((one.astype(np.float32) + two) / 2).astype(np.uint8))
