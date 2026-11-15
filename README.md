# Solar Filament Segmentation Challenge 2026 -- two-stage instance segmentation

Instance segmentation of solar filaments in H-alpha images (MAGFiLO) for the IEEE BigData Cup
2026 challenge. The metric is Panoptic Quality (PQ), computed against every annotator pass.

| | Final submission: EfficientNetV2-S stage-A | Earlier pipeline: ResNet34 stage-A + crop refiner |
|---|---|---|
| **Public leaderboard PQ** | **0.38** | 0.37 (0.36 without the instance scorer) |
| Out-of-fold PQ (657 images, one fold model per image, scorer cross-validated by image) | **0.4318** | 0.4089 (0.394 without the scorer; 0.388 also without the near-limb filter) |
| Hold-out PQ (50 images, never used for training or model selection; bootstrap SE 0.017) | 0.4309 | 0.4427 (0.4383 without the scorer; 0.4325 also without the near-limb filter) |
| Code | `filament_effv2s.py`, `models/effv2s_scorer.*` | `filament_pipeline.py`, `models/instance_scorer.*` |

Annotator-vs-annotator PQ on the training data (same rule): 0.34. Weights (Apache-2.0): Kaggle dataset
[`abukayyisahbasyid/solar-filament-segmentation-2026-weights`](https://www.kaggle.com/datasets/abukayyisahbasyid/solar-filament-segmentation-2026-weights).

The two pipelines disagree on the two evaluation sets: the EfficientNetV2-S stage-A is better out of fold (+0.0229,
CI95 +0.0166 to +0.0293) and on the leaderboard, and lower on the 50 hold-out images (-0.012, inside that set's
standard error; the earlier pipeline's thresholds were tuned on those images). Our pre-registered rule required both,
so by that rule it was "not adopted"; we chose it as the final submission anyway and disclose this in the report.

## Method in one paragraph
**Final submission.** Five U-Nets with an EfficientNetV2-S encoder (one per fold; `training/filament_effv2s_stagea.py`:
10,500 steps, soft multi-annotator target) each predict the full 2048 x 2048 image in four flip / rotation views; the
five maps are averaged. 8-connected regions of `prob > 0.35` with a peak above 0.5, area >= 100 px and mean prob >= 0.5
are the instances (no refiner). Instances whose centroid lies beyond 0.88 of the solar-disk radius are dropped, and a
LightGBM scorer (36 inference-time features, fitted on the 6,460 out-of-fold instances of these models) drops those
whose predicted matched IoU is below the PQ break-even, tau = 0.1885 (half of the unscored out-of-fold PQ).

**Earlier pipeline (still in the repository, selectable in the notebook).** A 5-fold ResNet34 U-Net ensemble with dihedral TTA produces a probability map (stage-A). Connected
components of `prob > 0.4` (area 400-40000 px, mean prob >= 0.7) become candidates. A crop refiner
(`smp.Unet`, ResNet34) redraws each candidate's mask from a box-shaped seed. Finally, instances whose
centroid lies beyond 0.88 of the solar-disk radius (disk from an Otsu threshold) are dropped, which
removes near-limb false positives (`Pipeline(limb_drop=None)` turns this off). Finally a learned
instance scorer (LightGBM on 36 inference-time features, fitted on 4,694 out-of-fold instances)
predicts each instance's expected matched IoU and drops those below the PQ break-even, PQ/2 = 0.197
(`Pipeline(scorer=None)` turns this off). One mask is one submission row. Details, ablations and negative results are in [`report/report.md`](report/report.md).

## Repository layout
```
pipeline.ipynb            end-to-end notebook: load weights -> (optional hold-out PQ) -> test predictions -> submission.csv
                          (STAGE_A = "effv2s": the final submission; "resnet34": the earlier pipeline)
filament_effv2s.py        inference code of the final submission (EfficientNetV2-S stage-A, loose instances, selection)
filament_pipeline.py      inference code of the earlier pipeline (models, TTA, candidates, refiner, near-limb filter) and
                          shared helpers (disk estimate, RLE, format validation)
filament_scorer.py        learned instance scorer: instance features and the keep/drop rule
models/                   the fitted scorers (LightGBM text model, feature list and threshold): effv2s_scorer.* for the
                          final submission, instance_scorer.* for the earlier pipeline
filament_metric.py        the official PQ, per annotator pass; fixed hold-out split
requirements.txt          pinned versions taken from the Kaggle image used for all runs
training/                 filament_effv2s_stagea.py produced effv2s_fold1..5.pt (one fold per Kaggle T4 kernel, about 2.15 h
                          each); filament_unet_ensemble.py and filament_refiner.py produced unet_fold1..5.pt and
                          refiner_best.pt; the other scripts are follow-up GPU studies (checkpoint selection, consensus
                          targets, refiner generation 2, spine head, per-epoch patch re-sampling)
analysis/                 studies (scripts + results): filament_*_cpu.py / *_gpu.py are the Kaggle kernel scripts, analyze_*.py the
                          local analyses that read their outputs from analysis/results/
tests/                    pytest suite: metric, candidates, near-limb filter, submission checks, instance scorer (including a
                          cross-version check of the stored model), and the final submission's instance rule, selection
                          and scorer (the stored scorer must rebuild the submitted file from the stored test instances);
                          run with `python -m pytest tests`
report/report.md          long-form technical report (all findings)
report/overleaf/          the 4-page report in the organizers' ACM (acmart) Overleaf template: main.tex, preamble.tex,
                          main.bib, figures, overleaf_project.zip (upload to Overleaf) and report_compiled_tectonic.pdf
                          (4 pages including acknowledgment and references; compiled with tectonic, so Overleaf's
                          pdfLaTeX output may differ slightly)
LICENSE                   Apache-2.0
```

## Weights
All weights are on the public Kaggle dataset
[`abukayyisahbasyid/solar-filament-segmentation-2026-weights`](https://www.kaggle.com/datasets/abukayyisahbasyid/solar-filament-segmentation-2026-weights)
(Apache-2.0): `effv2s_fold1..5.pt` (final submission), `unet_fold1..5.pt` and `refiner_best.pt` (earlier pipeline), and
`submission_effv2s_probe.csv`, the submitted file itself. Their SHA256 checksums are in `SHA256SUMS.txt`
(`sha256sum -c SHA256SUMS.txt` inside the dataset folder). The scorers are in this repository (`models/`).

## Scorer threshold configurations (earlier pipeline)
The instance scorer drops instances predicted below `tau`. `InstanceScorer.load()` uses the stored value (0.1972, the
out-of-fold PQ / 2, the PQ-optimal choice); `InstanceScorer.load(tau=0.15)` keeps more instances: out-of-fold PQ
-0.0027 and per-image Dice +0.0072 (hold-out -0.0009 and +0.0046). The test-set files are written by
`python analysis/scripts/make_scorer_submission.py <output_dir> [tau]`. See section 4 of the report.

## Reproduce the submission
**On Kaggle (recommended).** Create a notebook from `pipeline.ipynb`, attach (1) the weights dataset above and (2) the
competition data, enable a GPU (T4) and internet (only to `pip install segmentation-models-pytorch==0.5.0 timm==1.0.26`),
copy `filament_effv2s.py`, `filament_pipeline.py`, `filament_scorer.py`, `filament_metric.py` and the `models/` folder
next to the notebook, and run all cells. `STAGE_A = "effv2s"` (the default) writes the final submission to
`/kaggle/working/submission.csv`; `STAGE_A = "resnet34"` runs the earlier pipeline.

**Locally.** `pip install -r requirements.txt`, download the weights dataset and the competition data, then set
`FILAMENT_WEIGHTS` (the dataset folder) and `FILAMENT_INPUT` (root that contains `train_images/`, `test_images/` and
the annotation JSON) and run the notebook from this folder.

`MAX_TEST_IMAGES` limits the run for a smoke test; `EVAL_HOLDOUT_IMAGES = 50` scores the hold-out with the official PQ
(expected with `"effv2s"`: 0.4309; with `"resnet34"`: 0.4427, 0.4383 with `scorer=None`, 0.4325 with `scorer=None,
limb_drop=None`; up to CPU/GPU floating-point noise, about +-0.002).

**How the submitted file was produced, and what was checked.** The submitted maps were written by the five training
kernels right after training (GPU, float16 autocast), averaged and turned into instances and features by
`analysis/scripts/filament_effv2s_test_cpu.py`, and filtered by `analysis/scripts/make_effv2s_submission.py`, which
refit the scorer in memory. `filament_effv2s.py` is the same procedure as one code path, and
`analysis/scripts/fit_effv2s_scorer.py` stored that scorer as `models/effv2s_scorer.txt`. Two checks:
* *Selection, exact.* From the stored test instances (`analysis/results/effv2s/effv2s_test_instances.json`) the stored
  scorer rebuilds the submitted file byte for byte (1,048 rows, sha256 `031f80ba...29d520`; `tests/test_effv2s.py`).
* *From the weights, end to end.* `analysis/scripts/check_effv2s_e2e.py` ran `filament_effv2s.Pipeline` from the
  published weights (checksums verified) on four test images on a laptop CPU (15 November, torch 2.14, float32; about
  2.5 minutes per image; `analysis/results/effv2s/e2e_check_cpu.json`): the same 20 instances as in the submitted file
  (2, 5, 11 and 2 per image), 15 of the 20 rows identical as RLE strings, and every submitted mask matched by a new one
  with IoU >= 0.9986 (mean 0.9998). The five rows that differ do so by a few boundary pixels, because the submitted
  maps were computed on a GPU in float16. The full 180-image run from the weights has not been repeated, and the
  notebook itself has not been run on Kaggle since this code path was added.

**Earlier pipeline.** Cost: about 3 minutes per image on the 4-core Kaggle CPU (stage-A with 5 models x 4 flips
dominates). `pipeline.ipynb` was run on Kaggle (CPU) from the weights: on the first 3 hold-out images it reproduced
the per-image official TP/FP/FN and summed IoU of our earlier runs exactly. Its scorer submission was built from the
per-instance features of a full 180-image CPU run (`analysis/scripts/filament_test_instances_cpu.py`,
`make_scorer_submission.py`), and the code path was then checked end to end on Kaggle
(`filament_e2e_check_cpu.py`, 2 October): `Pipeline(..., scorer=InstanceScorer.load())` reproduced the per-image
official counts of all 50 hold-out images (pooled PQ 0.4427) and the submitted rows of four test images exactly, and
the `tests/` suite passed on the Kaggle image (LightGBM 4.6.0, torch 2.11).

## Retraining
`training/filament_effv2s_stagea.py` (env `FOLD=1..5`, one fold per kernel; fail-fast checks, atomic checkpoints,
resume, time budget) produced the final submission's weights; it also writes the out-of-fold, hold-out and test
probability maps. `training/filament_unet_ensemble.py` (stage-A, 5 folds x 30 epochs) and
`training/filament_refiner.py` (refiner, 13 epochs) are the exact Kaggle scripts that produced the
released weights (single T4 each; they read the competition data from `/kaggle/input`, and
`filament_refiner.py` loads the stage-A checkpoints only to evaluate the refiner on the hold-out). Both use the same fixed
hold-out (seed 1234, 50 images) which is excluded from every split.

**Note on the scripts.** The training and analysis scripts were originally written and run with
non-English comments and log messages; those were translated to English afterwards. The code itself
is unchanged (checked by comparing the syntax trees of the original and translated files with all
string constants masked). The only non-text edits are renames of a few identifiers, status values and
error-category labels to English (for example `passes_gate`, `REF_PREVIOUS_ENS_PQ`, `running`/`done`,
`*_localization`/`*_pure`), with the matching keys in the result JSON files in `analysis/results/`;
their numbers are unchanged.

## Compliance notes
* **Data and labels.** Training used only the competition's own images and annotations. No
  annotations outside those provided by the competition and no test-set labels were used. MAGFiLO is
  also released publicly; we did not download or consult that release (the organizers allow only the
  competition's annotations as ground truth). Inference uses only the H-alpha test images.
* **External assets.** Only ImageNet-pretrained encoders initialise the networks (EfficientNetV2-S via `timm`,
  ResNet34 via `torchvision` and `segmentation_models_pytorch`). The EfficientNetV2-S training recipe follows the
  description on a public model card of another participant (Kaggle dataset `dariushafshar/solar-filament-unet-weights`);
  the weights are our own, trained with our code on our splits. That participant's weights were scored once out of
  fold as a comparison (report, section 6) and are not part of any submission. No third-party competition model is
  part of either pipeline. (A separate ensemble with a public third-party Mask R-CNN was evaluated as an
  ablation; it gave +0.0028 hold-out PQ, statistically indistinguishable from zero, and is excluded
  because its training provenance cannot be verified here.)
* **Leaderboard.** No attempt was made to exploit scoring behaviour of the public leaderboard or any public annotations. To understand the leaderboard we decoded one public notebook that embeds a pre-computed submission and computed descriptive statistics only; it was never used for training, tuning, selection or evaluation. See sections 7-8 of the report.
* **AI assistance.** The code, analyses and this documentation were developed with an AI coding
  assistant (Claude Code) under the author's direction.

## Known limitations
* The hold-out has only 50 images; as unpaired numbers, PQ differences below about 0.03 are not
  statistically distinguishable (bootstrap SE 0.017). Per-image results are stored in
  `analysis/results/fpdiag/` so paired comparisons can be made.
* Annotators disagree strongly on filament boundaries (mean IoU of matched pairs between two
  annotators is 0.63), which caps the achievable PQ; see the report.

## License
Apache-2.0 for the code and the released weights. The competition data is not redistributed here and
may only be used under the competition's terms.
