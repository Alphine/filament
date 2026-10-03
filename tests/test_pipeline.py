"""filament_pipeline: candidates, near-limb filter, RLE and submission checks (no weights)."""
import cv2
import numpy as np
import pycocotools.mask as mu
import pytest

import filament_pipeline as fp


def blob_prob():
    prob = np.full((2048, 2048), 0.02, np.float32)
    prob[200:240, 200:300] = 0.95            # 4,000 px, confident
    prob[500:510, 500:510] = 0.95            # 100 px: below the area limit
    prob[800:840, 800:900] = 0.55            # 4,000 px, weak: below the mean-probability limit
    return prob


def test_released_candidate_thresholds():
    cand = fp.extract_candidates(blob_prob())
    assert cand.max() == 1 and (cand > 0).sum() >= 4000


def test_looser_thresholds_keep_more_candidates():
    cand = fp.extract_candidates(blob_prob(), thr=0.3, min_area=50, min_mean_prob=0.4)
    assert cand.max() == 3


def test_near_limb_filter_drops_only_the_limb_instance():
    gray = np.zeros((2048, 2048), np.uint8)
    cv2.circle(gray, (1024, 1024), 800, 180, -1)
    lab = np.zeros((2048, 2048), np.int32)
    lab[1000:1040, 1000:1100] = 1          # near the centre
    lab[1000:1040, 1790:1830] = 2          # centroid at about 0.99 R
    out = fp.drop_near_limb(lab, gray)
    assert set(np.unique(out)) == {0, 1}
    cx, cy, r = fp.estimate_disk(gray)
    assert abs(cx - 1024) < 3 and abs(cy - 1024) < 3 and abs(r - 800) < 5


def test_rles_round_trip():
    lab = np.zeros((64, 64), np.int32)
    lab[5:15, 5:20] = 1
    lab[30:40, 30:60] = 3                   # ids need not be contiguous
    rles = fp.label_map_to_rles(lab)
    masks = [mu.decode({"size": [64, 64], "counts": r.encode("ascii")}) for r in rles]
    assert len(masks) == 2 and masks[0].sum() == 150 and masks[1].sum() == 300


def rle(x0, x1):
    m = np.zeros((2048, 2048), np.uint8, order="F")
    m[100:150, x0:x1] = 1
    return mu.encode(m)["counts"].decode("ascii")


def test_submission_validation():
    ok = [{"filament_id": "a_1", "segmentation_rle": rle(10, 60)}, {"filament_id": "a_2", "segmentation_rle": rle(100, 160)}]
    assert fp.validate_submission_rows(ok, {"a", "b"}) == {"rows": 2, "images_with_predictions": 1, "images_without_predictions": 1}
    with pytest.raises(ValueError):                                   # duplicate id
        fp.validate_submission_rows(ok + ok[:1], {"a", "b"})
    with pytest.raises(ValueError):                                   # unknown image
        fp.validate_submission_rows([{"filament_id": "z_1", "segmentation_rle": rle(10, 60)}], {"a"})
    empty = mu.encode(np.zeros((2048, 2048), np.uint8, order="F"))["counts"].decode("ascii")
    with pytest.raises(ValueError):                                   # empty mask
        fp.validate_submission_rows([{"filament_id": "a_1", "segmentation_rle": empty}], {"a"})


def test_multiscale_inference_uses_the_resized_image_and_returns_full_size_maps():
    import torch
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen.append(tuple(x.shape[-2:]))
            return torch.zeros(x.shape[0], 1, *x.shape[-2:])

    model, dev = [Probe()], torch.device("cpu")
    gray = np.full((2048, 2048), 100, np.uint8)
    one = fp.predict_scaled(model, gray, 1.0, dev)
    assert np.array_equal(one, fp.predict_ensemble(model, gray, dev)) and one.shape == (2048, 2048)
    seen.clear()
    p = fp.predict_scaled(model, gray, 0.75, dev)
    assert p.shape == (2048, 2048) and seen and all(s == (512, 512) for s in seen)
    ms = fp.predict_multiscale(model, gray, (0.75, 1.0), dev)
    assert ms.shape == (2048, 2048) and np.allclose(ms, 0.5)             # sigmoid(0) at every scale


def test_pipeline_defaults_to_the_single_scale_setting():
    assert fp.Pipeline.__dataclass_fields__["scales"].default == (1.0,) and fp.MSC_SCALES == (0.75, 1.0, 1.25)
