# Two-stage solar filament instance segmentation, and what limits it

*Technical report draft for the Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup). Target: 4 pages in the organizers' Overleaf template; this file is the content, to be typeset there.*

## Abstract
We segment solar filaments in H-alpha images with a two-stage pipeline: a five-fold ResNet34 U-Net ensemble with dihedral test-time augmentation proposes candidate instances, and a crop-level refiner redraws each candidate from a box-shaped seed. On a 50-image hold-out never used for training or model selection the pipeline reaches PQ 0.4325 (bootstrap SE 0.017) and 0.36 on the public leaderboard. Two analyses shape our interpretation. First, two annotators disagree with each other at PQ 0.34 and mean matched IoU 0.63, so a large share of the remaining error is annotation ambiguity rather than model error. Second, most post-processing and loss ideas we tried (watershed, shape filters, topology losses, TTA gating, ensembling with a public model) did not help at the resolution our hold-out can measure.

## 1. Data and evaluation protocol
**Data.** 707 annotated training images (2048x2048, MAGFiLO [9]) and 180 test images. 296 training images have two or more annotator passes (145 with two, 151 with three); 411 have one.

**Metric.** PQ [1] = sum(IoU of TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|), TP if IoU > 0.5, evaluated against each annotator pass separately and pooled (`filament_metric.py`; verified equal to our earlier implementation on randomized data, identical TP/FP/FN counts).

**Hold-out.** 50 images chosen with a fixed seed (1234), excluded from every training fold, from checkpoint selection and from the refiner's training. All model comparisons below are on this set. Its resolution is limited: a per-image bootstrap gives PQ 0.4325 +- 0.017 (95% CI 0.399-0.465), so unpaired differences below roughly 0.03 cannot be distinguished from noise. We therefore report per-image results (`analysis/results/fpdiag`) and treat gains under about 0.005 as unproven.

## 2. Method
**Stage-A.** Five U-Nets [2] with ImageNet-initialised ResNet34 [3] encoders, one per fold of the 657 non-hold-out images (about 526 train / 131 checkpoint-selection images per fold; 30 epochs of 220 steps x batch 8 of 512-px patches, 75% of them centred on filaments; loss = Tversky [4] with alpha 0.3, beta 0.7 (penalises missed pixels more) + BCE; AdamW [5], one-cycle LR [6] peaking at 1e-4). Targets are the pixel union of all annotator passes of an image. Inference averages 5 models x 4 flips (identity, left-right, up-down, 180 deg) over 512-px windows with stride 384.

**Candidates.** `prob > 0.4`, 3x3 closing, 8-connected components; keep area 400-40000 px and mean probability >= 0.7. These thresholds were tuned on the hold-out; a later simulation of tightening them found no out-of-sample gain (split-half cross-validation mean -0.0005), i.e. they sit near a local optimum.

**Crop refiner.** For each candidate a square crop (1.5x the longest bounding-box side, min 96 px) is resized to 512x512 and fed as `[crop, CLAHE(crop) [7], seed]`, where the seed is the bounding box padded by 40% (during training the pad is drawn from 20-60%). A `smp.Unet(resnet34)` [8] predicts the mask (threshold 0.5). Training uses every annotator instance as one sample (6,474 crops) with the seed box jittered (shift +-15%, scale 0.85-1.25) so the network cannot memorise the exact box; loss 0.45 BCE + 0.55 soft Dice, 13 epochs. Using the real mask as seed at training time makes the refiner copy its input and fail at test time.

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
* **Hold-out versus leaderboard.** The leaderboard score (0.36) is below the hold-out PQ (0.4325). We do not have the test labels and did not identify the cause, so the following are hypotheses only: (i) the hold-out is a random subset of the training images and the test images may be distributed differently (for example by observation date); (ii) if the test ground truth comes from fewer annotator passes per image than our multi-pass average, the attainable PQ is lower, since two annotators agree with each other at only 0.34; (iii) 50 images give a standard error of 0.017. What does hold is that the ordering of our own submissions agreed with the hold-out ordering: 0.3754 (proxy) scored 0.33, an R34+R50 mix (0.4030, proxy) scored 0.34, and the refiner pipeline (0.4325) scored 0.36.
* Roughly 58% of training images have one annotator pass and give no consensus signal; a consensus-target retraining would be limited to the multi-pass subset.
* We did not use, and the pipeline does not depend on, any labels outside the competition annotations.

## 7. Leaderboard context
We report where the pipeline stands rather than optimise against the upper end of the leaderboard, because that end is
not consistent with legitimate modelling quality. Descriptive facts from the public leaderboard (657 teams, downloaded on
20 September 2026; scores rounded to two decimals): the scores form a smooth, bell-shaped distribution that ends at 0.41
(2 teams at 0.41, 9 at 0.40); there is then no team between 0.42 and 0.48 and one at 0.49, followed by 41 teams between
0.545 and 0.555 and five teams at 0.56-0.62. Our 0.36 ranks about 143rd of 657, or about 96th of the roughly 610 teams
below that gap. This is context, not a statement about any particular team.

The organizers have stated that scores of this kind are not considered valid ("we can easily weed out the fake ones",
competition forum topic 740400), that any PQ above 0.35 is "of great value" to them (competition announcement of
20 August), and that ground-truth annotations beyond those provided by the competition must not be used, otherwise
submissions are disregarded (topic 729396). Participants report in the forum that a public release of MAGFiLO contains
annotations for the competition's test images (topics 728062, 737177, 741892); we could not and did not verify this.
The organizers also state that their evaluation "is not based on the scores shown in the leaderboard", which is only
"a preliminary way of filtering some submissions out" (topic 728062), and that the final evaluation combines quantitative
and qualitative criteria (competition "Evaluation" page).

## 8. Compliance and data provenance
* **Training data.** Only the competition's own training images and annotations were used. Every model comparison uses a
  50-image hold-out drawn from the training images.
* **Public MAGFiLO release.** MAGFiLO [9] is also released publicly (Harvard Dataverse, CC BY-NC-ND 4.0). We did not
  download or consult that release or any dataset derived from it, and we used no test-image labels, dumped predictions or
  third-party submission files for training, tuning, model selection or evaluation.
* **Disclosure.** To understand the upper part of the leaderboard we decoded one public Kaggle notebook that embeds a
  pre-computed submission and computed only descriptive statistics of it (instance count, areas, overlaps). We never
  compared our predictions with it, and it played no role in any training, tuning, selection or evaluation decision.
* **External models.** ImageNet-pretrained ResNet34 encoders initialise the networks. A public third-party Mask R-CNN
  model (Kaggle Model `phuongncn/solar-filament-i079-weights`, Apache-2.0) was evaluated once as an ablation on our
  hold-out and is excluded from the released pipeline because its training data cannot be verified.
* **Inference.** Predictions use only the H-alpha test images.
* **AI assistance.** The code, analyses and this document were developed with an AI coding assistant (Claude Code) under
  the author's direction.

## 9. Reproducibility
Code, pinned requirements, weights (Apache-2.0, public Kaggle dataset with checksums) and an end-to-end notebook are in the repository; the notebook was validated on Kaggle from the public weights (identical per-image official counts on 3 hold-out images; format-valid output on 2 test images). The full 180-image submission was produced by the same code path in earlier kernels. Repository: https://github.com/Alphine/filament (default branch `final`). Weights: https://www.kaggle.com/datasets/abukayyisahbasyid/filament-segmentation-2026-weights. Public demo notebook: https://www.kaggle.com/code/abukayyisahbasyid/filament-two-stage-inference.

## References
[1] A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollar. Panoptic segmentation. CVPR 2019. doi:10.1109/CVPR.2019.00963
[2] O. Ronneberger, P. Fischer, T. Brox. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.
[3] K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. CVPR 2016.
[4] S. S. M. Salehi, D. Erdogmus, A. Gholipour. Tversky loss function for image segmentation using 3D fully convolutional deep networks. MLMI 2017.
[5] I. Loshchilov, F. Hutter. Decoupled weight decay regularization. ICLR 2019.
[6] L. N. Smith, N. Topin. Super-convergence: very fast training of neural networks using large learning rates. arXiv:1708.07120, 2017.
[7] K. Zuiderveld. Contrast limited adaptive histogram equalization. Graphics Gems IV, 1994.
[8] P. Iakubovskii. Segmentation Models PyTorch. https://github.com/qubvel-org/segmentation_models.pytorch
[9] A. Ahmadzadeh, R. Adhyapak, K. Chaurasiya, L. A. Nagubandi, V. Aparna, P. C. Martens, A. Pevtsov, L. Bertello, A. Pevtsov, N. Douglas, S. McDonald, A. Bawa, E. Kang, R. Wu, D. J. Kempton, A. Abdelkarem, P. M. Copeland, S. H. Seelamneni. A dataset of manually annotated filaments from H-alpha observations. Scientific Data 11, 1031 (2024). doi:10.1038/s41597-024-03876-y
