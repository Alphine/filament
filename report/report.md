# Two-stage solar filament instance segmentation, and what limits it

*Technical report draft for the Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup). Target: 4 pages in the organizers' Overleaf template; this file is the content, to be typeset there. Bracketed items need the author's input.*

## Abstract
We segment solar filaments in H-alpha images with a two-stage pipeline: a five-fold ResNet34 U-Net ensemble with dihedral test-time augmentation proposes candidate instances, and a crop-level refiner redraws each candidate from a box-shaped seed. On a 50-image hold-out never used for training or model selection the pipeline reaches PQ 0.4325 (bootstrap SE 0.017) and 0.36 on the public leaderboard. Two analyses shape our interpretation. First, two annotators disagree with each other at PQ 0.34 and mean matched IoU 0.63, so a large share of the remaining error is annotation ambiguity rather than model error. Second, most post-processing and loss ideas we tried (watershed, shape filters, topology losses, TTA gating, ensembling with a public model) did not help at the resolution our hold-out can measure.

## 1. Data and evaluation protocol
**Data.** 707 annotated training images (2048x2048, MAGFiLO) and 180 test images. 296 training images have two or more annotator passes (145 with two, 151 with three); 411 have one.

**Metric.** PQ = sum(IoU of TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|), TP if IoU > 0.5, evaluated against each annotator pass separately and pooled (`filament_metric.py`; verified equal to our earlier implementation on randomized data, identical TP/FP/FN counts).

**Hold-out.** 50 images chosen with a fixed seed (1234), excluded from every training fold, from checkpoint selection and from the refiner's training. All model comparisons below are on this set. Its resolution is limited: a per-image bootstrap gives PQ 0.4325 +- 0.017 (95% CI 0.399-0.465), so unpaired differences below roughly 0.03 cannot be distinguished from noise. We therefore report per-image results (`analysis/results/fpdiag`) and treat gains under about 0.005 as unproven.

## 2. Method
**Stage-A.** Five U-Nets with ImageNet-initialised ResNet34 encoders, one per fold of the 657 non-hold-out images (about 526 train / 131 checkpoint-selection images per fold; 30 epochs of 220 steps x batch 8 of 512-px patches, 75% of them centred on filaments; loss = Tversky with alpha 0.3, beta 0.7 (penalises missed pixels more) + BCE; AdamW, one-cycle LR peaking at 1e-4). Targets are the pixel union of all annotator passes of an image. Inference averages 5 models x 4 flips (identity, left-right, up-down, 180 deg) over 512-px windows with stride 384.

**Candidates.** `prob > 0.4`, 3x3 closing, 8-connected components; keep area 400-40000 px and mean probability >= 0.7. These thresholds were tuned on the hold-out; a later simulation of tightening them found no out-of-sample gain (split-half cross-validation mean -0.0005), i.e. they sit near a local optimum.

**Crop refiner.** For each candidate a square crop (1.5x the longest bounding-box side, min 96 px) is resized to 512x512 and fed as `[crop, CLAHE(crop), seed]`, where the seed is the bounding box padded by 40% (during training the pad is drawn from 20-60%). A `smp.Unet(resnet34)` predicts the mask (threshold 0.5). Training uses every annotator instance as one sample (6,474 crops) with the seed box jittered (shift +-15%, scale 0.85-1.25) so the network cannot memorise the exact box; loss 0.45 BCE + 0.55 soft Dice, 13 epochs. Using the real mask as seed at training time makes the refiner copy its input and fail at test time.

## 3. Results
| Configuration | Hold-out PQ | Public LB |
|---|---|---|
| Single U-Net (proxy metric) | 0.3348 | - |
| 5-fold ensemble, no TTA (proxy metric) | 0.3754 | 0.33 |
| + dihedral TTA (official PQ) | 0.4151 | - |
| + crop refiner (**final**) | **0.4325-0.4328** | **0.36** |

(Early rows use an earlier proxy of the metric, which scores against the annotator pass with the most instances; official PQ is used from the TTA row on.) The refiner is the only change that produced a measurable gain: +0.0177 hold-out PQ, and +0.02 on the leaderboard. The 0.4325 vs 0.4328 difference is CPU-vs-GPU floating-point noise: a re-run on CPU reproduced TP/FP/FN = 400/239/211 exactly.

## 4. Negative and null results
All measured against the same hold-out, official PQ unless noted.

| Idea | Effect | Note |
|---|---|---|
| Hard-negative training of the refiner | +0.0006 | below noise |
| Skeleton-recall (spine) loss | -0.0037 | neutral to slightly negative |
| Watershed instance splitting | 0.4328 -> 0.0599 | false positives up 3.5x |
| Shape filter (aspect/circularity, sunspot rejection) | -0.0044 | rejects more true positives than false positives; it lowers PQ because a lost TP costs both numerator and denominator |
| Topology-safe TTA gating on the refiner | -0.0009 | gate accepted 419 of 420 instances |
| Ensemble with a public Mask R-CNN model | +0.0028 | not distinguishable from zero; that model alone scored 0.4236; excluded from the released pipeline |
| Second ResNet34 ensemble / R34+R50 mix | about +-0.002 | TTA absorbs the diversity gain |

## 5. Where the remaining error comes from
**Error decomposition** (final pipeline on the hold-out, TP/FP/FN = 400/239/211): 54% of the 450 errors are near-misses, i.e. a prediction and a ground-truth filament overlap but with IoU <= 0.5 (132 of 239 FP, 110 of 211 FN); the rest have no overlap at all. The mean IoU of true positives is 0.676.

**Annotator agreement.** Using one pass as the prediction and another as ground truth with the same rules (296 multi-pass images, 1,196 ordered pairs): PQ 0.340 (SE 0.008), mean matched IoU 0.634, and only 0.14% of matched pairs reach IoU 0.85. Near-misses make up 62% of the disagreement errors. Our pipeline exceeds this pairwise agreement (0.4325, IoU 0.676), plausibly because it predicts an average of several annotators; boundary quality beyond that is largely unresolvable from the labels. It is an indication of a ceiling, not a strict bound: a consensus of many annotators could do better than one.

**Training target.** A leave-one-pass-out simulation on 151 three-pass images builds predictions from two passes and scores them against the third. Predicting only objects that both passes marked, with the union shape, beats the pixel-union target (the one stage-A is trained on) by +0.0145 (paired bootstrap 95% CI +0.006 to +0.023); intersection shapes lose. Objects marked by one annotator only match a third pass about 22% of the time, below the break-even fraction of roughly 0.32 that follows from the PQ formula. This is a simulation of the target, not a trained model; because the tuned thresholds already suppress low-confidence objects, we expect most of this effect to be captured already and did not retrain.

## 6. Discussion and limitations
* Hold-out resolution (SE 0.017) means several of our null results are inconclusive rather than negative.
* The leaderboard score (0.36) is below the hold-out PQ (0.4325); we did not identify a cause. Different label sets or the 180-image test distribution are candidates. [Author: add if known.]
* Roughly 58% of training images have one annotator pass and give no consensus signal; a consensus-target retraining would be limited to the multi-pass subset.
* We did not use, and the pipeline does not depend on, any labels outside the competition annotations.

## 7. Reproducibility
Code, pinned requirements, weights (Apache-2.0, public Kaggle dataset with checksums) and an end-to-end notebook are in the repository; the notebook was validated on Kaggle from the public weights (identical per-image official counts on 3 hold-out images; format-valid output on 2 test images). The full 180-image submission was produced by the same code path in earlier kernels. [Author: add repository URL.]

## References
[Add: Panoptic Quality (Kirillov et al., CVPR 2019); U-Net (Ronneberger et al., 2015); ResNet (He et al., 2016); segmentation_models_pytorch; the MAGFiLO dataset paper.]
