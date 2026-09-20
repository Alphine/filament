# Solar Filament Segmentation Challenge 2026 -- two-stage instance segmentation

Instance segmentation of solar filaments in H-alpha images (MAGFiLO) for the IEEE BigData Cup
2026 challenge. The metric is Panoptic Quality (PQ), computed against every annotator pass.

| | |
|---|---|
| **Hold-out PQ** (50 images, never used for training or model selection) | **0.4325** (bootstrap SE 0.017) |
| **Public leaderboard PQ** | **0.36** |
| Annotator-vs-annotator PQ on the training data (same rule) | 0.34 |
| Weights | Kaggle dataset [`abukayyisahbasyid/filament-segmentation-2026-weights`](https://www.kaggle.com/datasets/abukayyisahbasyid/filament-segmentation-2026-weights) (Apache-2.0) |

## Method in one paragraph
A 5-fold ResNet34 U-Net ensemble with dihedral TTA produces a probability map (stage-A). Connected
components of `prob > 0.4` (area 400-40000 px, mean prob >= 0.7) become candidates. A crop refiner
(`smp.Unet`, ResNet34) redraws each candidate's mask from a box-shaped seed; one mask is one
submission row. Details, ablations and negative results are in [`report/report.md`](report/report.md).

## Repository layout
```
pipeline.ipynb            end-to-end notebook: load weights -> (optional hold-out PQ) -> test predictions -> submission.csv
filament_pipeline.py      inference code (models, TTA, candidates, refiner, RLE, format validation)
filament_metric.py        the official PQ, per annotator pass; fixed hold-out split
requirements.txt          pinned versions taken from the Kaggle image used for all runs
training/                 the two Kaggle GPU scripts that produced the weights
analysis/                 error decomposition, annotator agreement, consensus-target and threshold studies (scripts + results)
report/report.md          technical report
LICENSE                   Apache-2.0
```

## Reproduce the submission
**On Kaggle (recommended).** Create a notebook from `pipeline.ipynb`, attach (1) the weights dataset
above and (2) the competition data, enable a GPU (T4) and internet (only to `pip install
segmentation-models-pytorch==0.5.0`), copy `filament_pipeline.py` and `filament_metric.py` next to the
notebook, and run all cells. `submission.csv` is written to `/kaggle/working`.

**Locally.** `pip install -r requirements.txt`, download the weights dataset and the competition
data, then set `FILAMENT_WEIGHTS` (folder with `unet_fold1..5.pt`, `refiner_best.pt`) and
`FILAMENT_INPUT` (root that contains `train_images/`, `test_images/` and the annotation JSON) and run
the notebook from this folder.

`MAX_TEST_IMAGES` limits the run for a smoke test; `EVAL_HOLDOUT_IMAGES = 50` scores the hold-out
with the official PQ (expected 0.4325 up to CPU/GPU floating-point noise, about +-0.002).
Checksums of every weight file are in the dataset's `SHA256SUMS.txt`.

Measured cost: about 3 minutes per image on the 4-core Kaggle CPU when one notebook runs alone (up to
5 when several run in parallel); stage-A with 5 models x 4 flips dominates. GPU inference time was not
measured.

**Validation.** `pipeline.ipynb` was run on Kaggle (CPU) from the public weights dataset: on the first 3
hold-out images it reproduced the per-image official TP/FP/FN and summed IoU of our earlier runs
exactly, and a 2-image test run produced a format-valid `submission.csv`. The full 180-image run was
not repeated with this notebook.

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
