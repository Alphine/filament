"""Inference of the final submission: EfficientNetV2-S stage-A, loose candidates, near-limb filter, instance scorer.

Stage-A : five U-Nets with an EfficientNetV2-S encoder (training/filament_effv2s_stagea.py, one per fold). Each sees
          the full 2048 x 2048 image, normalised by the mean / SD inside a 950 px disk, in four flip / rotation
          views; a model's map is the mean of its views stored as uint8 (p x 255), and the ensemble map is the mean
          of the five uint8 maps, rounded to uint8 -- the same arithmetic as the kernels that produced the submission.
Instances: 8-connected regions of p > 0.35 with peak > 0.5, area >= 100 px and mean p >= 0.5 (no refiner).
Filter  : instances whose centroid lies beyond 0.88 of the solar-disk radius are dropped, then those whose predicted
          matched IoU (models/effv2s_scorer.txt, LightGBM on 36 features) is below tau = 0.1885. The features are
          computed on all loose instances of the image, before the near-limb filter, as in the submission.

Public leaderboard 0.38; out of fold 0.4318 on 657 images; hold-out 0.4309 (report, section 6).
The earlier ResNet34 + refiner pipeline (0.37) is filament_pipeline.Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

import filament_pipeline as fp
import filament_scorer as fs

ENCODER = "tu-tf_efficientnetv2_s"
DISK_R = 950                                             # normalisation disk radius (px, image centre)
VIEWS = ((0, False), (0, True), (2, False), (2, True))   # (rot90 count, left-right flip)

T_LO = 0.35              # region threshold
T_HI = 0.5               # a region needs a peak above this
MIN_AREA = 100
MIN_MEAN_P = 0.5


def load_stage_a(paths: list[str], device: torch.device | None = None) -> list[nn.Module]:
    import segmentation_models_pytorch as smp
    device = device or fp.default_device()
    models = []
    for p in paths:
        m = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=1, classes=1)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        models.append(m.eval().to(device))
    return models


def disk_stats(gray: np.ndarray) -> tuple[float, float]:
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    v = gray[((yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2) <= DISK_R ** 2].astype(np.float32) / 255.0
    return float(v.mean()), max(float(v.std()), 1e-3)


@torch.inference_mode()
def predict_model(model: nn.Module, gray: np.ndarray, device: torch.device) -> np.ndarray:
    """Full-image probability of one model, mean over the four views, as uint8 (p x 255)."""
    mean, std = disk_stats(gray)
    z = ((torch.from_numpy(gray.copy()).to(device).float() / 255.0 - mean) / std)[None, None]
    acc = torch.zeros(gray.shape, device=device)
    amp = device.type == "cuda"
    for k, flip in VIEWS:
        x = torch.flip(z, dims=[3]) if flip else z
        x = torch.rot90(x, k, dims=[2, 3]) if k else x
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            p = torch.sigmoid(model(x).float())
        p = torch.rot90(p, -k, dims=[2, 3]) if k else p
        acc += (torch.flip(p, dims=[3]) if flip else p)[0, 0]
    return (acc / len(VIEWS) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def predict_ensemble(models: list[nn.Module], gray: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Mean of the models' uint8 maps, rounded to uint8."""
    device = device or fp.default_device()
    maps = [predict_model(m, gray, device).astype(np.float32) for m in models]
    return np.clip(np.round(np.mean(maps, axis=0)), 0, 255).astype(np.uint8)


def _u8(t: float) -> int:
    return int(round(float(t) * 255))


def loose_instances(p_u8: np.ndarray) -> np.ndarray:
    """Label map (0 = background, 1..n) of the loose-rule instances of a uint8 probability map."""
    lab, n = ndimage.label(p_u8 > _u8(T_LO), structure=np.ones((3, 3), bool))
    lv, pv = lab.ravel(), p_u8.ravel()
    nz = np.flatnonzero(lv)
    area = np.bincount(lv[nz], minlength=n + 1)
    psum = np.bincount(lv[nz], weights=pv[nz], minlength=n + 1)
    pmax = np.zeros(n + 1, np.uint8)
    np.maximum.at(pmax, lv[nz], pv[nz])
    keep = (pmax > _u8(T_HI)) & (area >= MIN_AREA) & (psum / np.maximum(area, 1) / 255.0 >= MIN_MEAN_P)
    keep[0] = False
    ids = np.flatnonzero(keep)
    remap = np.zeros(n + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[lab]


def select(gray: np.ndarray, labels: np.ndarray, p_u8: np.ndarray, scorer, limb_drop: float | None) -> np.ndarray:
    """Near-limb filter and scorer on the loose instances; kept instances are renumbered 1..k in id order."""
    inst, _ = fs.image_instances(gray, labels, labels, p_u8.astype(np.float32) / 255.0)
    if limb_drop is not None:
        inst = [r for r in inst if r["feats"]["dist_r"] <= limb_drop]
    if scorer is not None:
        inst = [r for r, v in zip(inst, scorer.scores(inst)) if v >= scorer.tau]
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    for k, r in enumerate(inst, 1):
        remap[r["id"]] = k
    return remap[labels]


@dataclass
class Pipeline:
    stage_a: list[nn.Module]
    device: torch.device
    scorer: object | None = None                 # fs.InstanceScorer.load(name="effv2s_scorer"); None disables it
    limb_drop: float | None = fp.LIMB_DROP_T     # None disables the near-limb filter

    def segment(self, gray: np.ndarray) -> np.ndarray:
        """uint8 grayscale image -> instance label map (0 = background)."""
        p_u8 = predict_ensemble(self.stage_a, gray, self.device)
        labels = loose_instances(p_u8)
        if self.scorer is None and self.limb_drop is None:
            return labels
        return select(gray, labels, p_u8, self.scorer, self.limb_drop)
