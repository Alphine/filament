# Solar Filament Segmentation Challenge 2026 -- two-stage instance segmentation

Instance segmentation of solar filaments in H-alpha images (MAGFiLO) for the IEEE BigData Cup
2026 challenge. The metric is Panoptic Quality (PQ), computed against every annotator pass.

| | |
|---|---|
| **Hold-out PQ** (50 images, never used for training or model selection) | **0.4427** full pipeline; 0.4384 without the instance scorer; 0.4325 also without the near-limb filter (bootstrap SE 0.017) |
| Out-of-fold PQ (657 images, one fold model per image) | 0.409 full pipeline (scorer cross-validated by image); 0.394 without the scorer; 0.388 also without the near-limb filter (SE 0.005) |
| **Public leaderboard PQ** | **0.37** (0.36 without the instance scorer) |
| Annotator-vs-annotator PQ on the training data (same rule) | 0.34 |
| Weights | Not currently published (see note below); Apache-2.0 |

## Method in one paragraph
A 5-fold ResNet34 U-Net ensemble with dihedral TTA produces a probability map (stage-A). Connected
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
filament_pipeline.py      inference code (models, TTA, candidates, refiner, near-limb filter, RLE, format validation)
filament_scorer.py        learned instance scorer: instance features and the keep/drop rule
models/                   the fitted scorer (LightGBM text model, feature list and threshold)
filament_metric.py        the official PQ, per annotator pass; fixed hold-out split
requirements.txt          pinned versions taken from the Kaggle image used for all runs
training/                 filament_unet_ensemble.py and filament_refiner.py produced the released weights (Kaggle GPU scripts);
                          the other scripts are follow-up GPU studies (checkpoint selection, consensus targets, refiner
                          generation 2, spine head; filament_resample_stagea.py is prepared but has not been run)
analysis/                 studies (scripts + results): filament_*_cpu.py / *_gpu.py are the Kaggle kernel scripts, analyze_*.py the
                          local analyses that read their outputs from analysis/results/
tests/                    pytest suite: metric, candidates, near-limb filter, submission checks, instance scorer (including a
                          cross-version check of the stored model); run with `python -m pytest tests`
report/report.md          long-form technical report (all findings)
report/overleaf/          the 4-page report in the organizers' ACM (acmart) Overleaf template: main.tex, preamble.tex,
                          main.bib, figures, overleaf_project.zip (upload to Overleaf) and report_compiled_tectonic.pdf
                          (3 pages including acknowledgment and references; compiled with tectonic, so Overleaf's
                          pdfLaTeX output may differ slightly)
LICENSE                   Apache-2.0
```

## Weights availability
The trained weights (`unet_fold1..5.pt`, `refiner_best.pt`) are not published on Kaggle right now.
They were briefly on a public Kaggle dataset and are being kept unpublished until shortly before the
15 Nov 2026 submission deadline, to limit how early competitors can build on this exact checkpoint
while the competition is still open; this does not affect the code, which is complete and public here.
The files, with the same SHA256 checksums that were on the dataset (`SHA256SUMS.txt` below), and the
exact `training/` scripts and data split that produced them, are kept privately by the author and will
be re-published as a Kaggle dataset before the deadline, with this README updated to link it.

## Scorer threshold configurations
The instance scorer drops instances predicted below `tau`. `InstanceScorer.load()` uses the stored value (0.1972, the
out-of-fold PQ / 2, the PQ-optimal choice); `InstanceScorer.load(tau=0.15)` keeps more instances: out-of-fold PQ
-0.0027 and per-image Dice +0.0072 (hold-out -0.0009 and +0.0046). The test-set files are written by
`python analysis/scripts/make_scorer_submission.py <output_dir> [tau]`. See section 4 of the report.

## Reproduce the submission
**On Kaggle (recommended, once the weights dataset is re-published).** Create a notebook from
`pipeline.ipynb`, attach (1) the weights dataset and (2) the competition data, enable a GPU (T4) and
internet (only to `pip install segmentation-models-pytorch==0.5.0`), copy `filament_pipeline.py`,
`filament_scorer.py`, `filament_metric.py` and the `models/` folder next to the notebook, and run all cells. `submission.csv` is written to
`/kaggle/working`.

**Locally.** `pip install -r requirements.txt`, obtain the weights (see above) and the competition
data, then set `FILAMENT_WEIGHTS` (folder with `unet_fold1..5.pt`, `refiner_best.pt`) and
`FILAMENT_INPUT` (root that contains `train_images/`, `test_images/` and the annotation JSON) and run
the notebook from this folder.

`MAX_TEST_IMAGES` limits the run for a smoke test; `EVAL_HOLDOUT_IMAGES = 50` scores the hold-out
with the official PQ (expected 0.4427; 0.4384 with `scorer=None`; 0.4325 with `scorer=None,
limb_drop=None`; up to CPU/GPU floating-point noise, about +-0.002).
Checksums of every weight file are in `SHA256SUMS.txt` below.

Measured cost: about 3 minutes per image on the 4-core Kaggle CPU when one notebook runs alone (up to
5 when several run in parallel); stage-A with 5 models x 4 flips dominates. GPU inference time was not
measured.

**Validation.** `pipeline.ipynb` was run on Kaggle (CPU) from the public weights dataset: on the first 3
hold-out images it reproduced the per-image official TP/FP/FN and summed IoU of our earlier runs
exactly, and a 2-image test run produced a format-valid `submission.csv`. The full 180-image run was
not repeated with this notebook. The instance scorer was added afterwards. Its submission was built from the
per-instance features of a full 180-image CPU run of the same pipeline code
(`analysis/scripts/filament_test_instances_cpu.py`, `make_scorer_submission.py`), and the released code path was
then checked end to end on Kaggle (`filament_e2e_check_cpu.py`, 2 October): `Pipeline(..., scorer=InstanceScorer.load())`
reproduced the per-image official counts of all 50 hold-out images (pooled PQ 0.4427) and the submitted rows of four
test images exactly, and the `tests/` suite passed on the Kaggle image (LightGBM 4.6.0, torch 2.11).

## Retraining
`training/filament_unet_ensemble.py` (stage-A, 5 folds x 30 epochs) and
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
* **External assets.** Only ImageNet-pretrained ResNet34 encoders (via `torchvision` and
  `segmentation_models_pytorch`) initialise the networks. No third-party competition model is part of
  the released pipeline. (A separate ensemble with a public third-party Mask R-CNN was evaluated as an
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
