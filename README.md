# Pathology-dependent fidelity in accelerated knee MRI

Code, splits and results for *"A Single Fidelity Score Conceals Pathology-Dependent
Loss in Accelerated Knee MRI"* (W3PHIAI-27 submission).

We measure whether accelerated reconstruction treats radiologist-annotated
findings differently from surrounding tissue, using fastMRI knee data with
fastMRI+ bounding boxes. **No model is trained anywhere in this pipeline** — the
reconstructions are zero-filling and CG-SENSE, the latter being the operator
that raw-data-free training methods use to build their training targets.

## Headline results

| quantity | value |
|---|---|
| test slices / patients | 380 / 19 |
| annotated slices / lesion regions | 141 / 218 |
| lesion vs matched control, CG-SENSE, SSIM | **+0.038** (95% CI +0.006 to +0.070), p = 4.6e-5 |
| lesion vs matched control, zero-filled, SSIM | **−0.024** (95% CI −0.036 to −0.013), p = 1.9e-7 |
| lesion vs matched control, CG-SENSE, PSNR | −0.064 dB, p = 0.81 |

The comparison reverses sign between reconstructions, and again between metrics.
Per finding (CG-SENSE): meniscus tear +0.055 (97 regions, 15 patients),
partial-thickness cartilage defect −0.028 (49 regions, 9 patients).

## What is NOT in this repository

- **fastMRI imaging data.** Requires a signed licence from NYU Langone; obtain it
  at <https://fastmri.med.nyu.edu/>. Nothing derived from the raw k-space is
  redistributed here.
- **fastMRI+ annotations.** Fetch `Annotations/knee.csv` from
  <https://github.com/microsoft/fastmri-plus> under its own licence.
- **Preprocessed inputs** (`cupid_data/`, ~26 GB of `.npy`) — regenerate with
  `preprocessing/prepare_cupid_data.py`.
- **The CUPID repository.** Clone separately from
  <https://github.com/ualcalar17/CUPID>; only our loader patch is included here.

## Reproducing

Requires Python 3.10+, `pip install -r requirements.txt`. Paths are set at the
top of each script. All timings are from a 16-core CPU; no GPU is used.

```
python preprocessing/prepare_cupid_data.py      # build inputs, 680 slices,  ~9 min
python preprocessing/run_baseline_metrics.py    # whole-image vs in-box,     ~39 min
python preprocessing/analyze_baseline.py        # statistics for the above
python preprocessing/run_roi_controlled.py      # lesion vs matched controls,~16 min
python preprocessing/run_sweep_covariates.py    # R=2/4/8 + ROI covariates,  ~43 min
python preprocessing/run_arms_acs.py            # zero-filled arm, ACS sweep,~42 min
python preprocessing/make_figures.py            # figures 2 and 3
```

## Layout

```
preprocessing/   data preparation, reconstruction, metrics, figures, GPU probe
splits/          subject-wise split, slice lists, volume index, split_config.json
results/         per-slice and per-region tables plus derived statistics (JSON)
figures/         figures as they appear in the paper
paper/           LaTeX source and generated tables
```

## Protocol

- Acquisition: fastMRI knee `CORPD_FBK`, restricted to the 49 volumes with 368
  phase-encode lines so matrix size is constant. Readout oversampling removed
  640 -> 320 in the image domain.
- Coil maps: Hann-windowed ACS block, inverse FFT per coil, divided by
  root-sum-of-squares, zeroed below 2% of max. Validated against
  `reconstruction_rss`: correlation 0.999, NRMSE 0.062.
- Mask: equidistant phase encode, central ACS block, signal-free lines marked
  known. Realised sampled fraction 0.582 / 0.372 / 0.266 at R = 2 / 4 / 8.
- Reconstructions: zero-filled adjoint, and CG-SENSE at 30 iterations, mu = 0.
- Controls: five regions per lesion box, identical size, inside eroded tissue,
  disjoint from every annotated box on that slice.
- Statistics: two-sided Wilcoxon signed-rank over regions; 95% intervals are
  subject-clustered bootstrap percentiles (5000 resamples over patients).

## A note on the GPU probe

`preprocessing/probe_cupid_cost.py` measures CUPID's per-slice VRAM and step
time. On an RTX 3050 (4 GB) the published configuration (10 unrolls, 15 CG
iterations, 6 perturbations) does not fit; measured costs with a
per-perturbation backward are 2.34 GiB / 7.4 s per step at 6 unrolls, and
3.54 GiB / 69.7 s per step at 10 unrolls with 8 CG iterations. Results in
`results/gpu_probe.json`. This supports the follow-up study and is not used in
the paper.

## Follow-up

`splits/` also contains a 15-patient / 300-slice training split this paper does
not use. It is released for the next experiment: training a raw-data-free
reconstruction on those slices and re-running this protocol on its outputs.
