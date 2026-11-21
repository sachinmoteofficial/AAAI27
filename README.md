# Lesion or Location? Negative controls for lesion-level fidelity in accelerated knee MRI

Code, subject splits and per-region results for the W3PHIAI-27 paper
*Lesion or Location? Negative Controls for Lesion-Level Fidelity Evaluation in
Accelerated Knee MRI*.

The study asks whether scoring a reconstruction inside a fastMRI+ lesion box
measures what happens to the finding. No model is trained. Reconstructions are
the zero-filled adjoint and unregularised CG-SENSE (30 iterations) at nominal
R = 2, 4, 8.

## Main results (CG-SENSE, nominal R = 4, SSIM; 217 boxes, 18 patients)

Lesion effect = in-box SSIM of the lesion minus the mean of its comparator boxes.
Intervals are patient-clustered bootstrap; p is a Wilcoxon test on patient means.

| comparator | effect | 95% CI | patients +/- (p) |
|---|---|---|---|
| size-matched tissue, same slice | +0.044 | [+0.005, +0.084] | 12/6 (0.13) |
| appearance-matched tissue, same slice | +0.050 | [+0.016, +0.085] | 11/7 (0.05) |
| same box, all unannotated slices | -0.073 | [-0.138, -0.007] | 6/12 (0.12) |
| same box, slices within 3 | -0.023 | [-0.050, +0.000] | 6/12 (0.14) |
| same box, adjacent slice | -0.003 | [-0.017, +0.011] | 8/10 (0.90) |

* The estimated lesion effect is set by the comparator and grows with the
  distance to the comparator slice.
* Partial-thickness cartilage defects score -0.185 against all unannotated
  slices (8 of 9 patients worse), but +0.016 against the adjacent slice: the
  apparent deficit comes from distant comparators.
* Registration check: with the correct bottom-left origin
  (`row_top = 320 - y - h`), 93.8% of meniscus-tear and 95.9% of
  cartilage-defect boxes lie within 15 px of the joint line; with a top-left
  origin, none do.

## Coordinate convention

fastMRI+ `y` is measured from the bottom of the 320x320 image. The conversion is
defined once in `scripts/common.py` (`box_row_top`). Every analysis script runs a
registration check first and stops if fewer than 80% of meniscus/cartilage boxes
lie within 15 px of the joint line.

## Reproducing

Requires Python 3.10+, NumPy, SciPy, pandas, h5py, matplotlib. Data paths are set
at the top of each script.

    python scripts/run_main_R4.py                 # all comparators, ZF and CG-SENSE, R=4
    python scripts/run_stats_R4.py                # pooled and per-finding contrasts
    python scripts/run_accel.py 2                 # CG-SENSE acceleration arm
    python scripts/run_accel.py 8
    python scripts/run_accel_stats.py
    python scripts/run_distance_stats.py results  # same-position effect by slice distance
    python scripts/make_figure_data.py
    python scripts/make_paper_figures.py . figures

## Not included

* fastMRI imaging data (requires the NYU Langone fastMRI data use agreement).
* fastMRI+ annotations (download from the fastMRI+ repository under its licence).

## Layout

    scripts/      coordinate conversion, reconstruction, comparators, statistics, figures
    splits/       subject-wise split (15 train, 19 test of which 18 carry boxes, 15 unused)
    results/      per-region and per-placement tables, JSON statistics
    figure_data/  figure source data
    figures/      paper figures
    paper/        LaTeX source

The `results_v1/` folder, if present, holds the earlier analysis that used the
wrong vertical origin; it is kept for transparency and is not used in the paper.
