# Six-stage geometry: Monkey T, channel 7

This is a new, isolated experiment on 198 saved short-delay trials from
`monkeyT_session-y070316009-12_lfp-7`. It does not overwrite earlier results.

## Start here

- `report.md`: concise results and limitations.
- `png/geometry_trial_6_observed.png`: six observed clouds from the first original trial ID.
- `png/geometry_trial_6_fitted.png`: the same clouds with fitted annular outlines.
- `png/distribution_R1.png`, `distribution_R2.png`, and `distribution_band_half_width.png`: feature distributions.
- `png/orientation_normal.png`, `orientation_u.png`, `orientation_v.png`: canonical orientation components.
- `png/stage_confusion.png` and `png/stage_f1.png`: held-out stage decoding.
- `captions.md`: interpretation, units, sample accounting, and statistical definitions.

All 53 figure exports are PNG, 600 dpi. Each individual cloud is also exported
separately. Trial 9 and 10 galleries are additional examples chosen by trial ID,
not fit quality or separation. No Betti numbers, spectral comparison, PCA
reduction, PINN, or embedding search are included.

## Data and reproducibility

Six stages are ordered pre-TC, post-TC, pre-SC, post-SC, pre-GO, post-GO.
Pre/post mean [-300,0) and [0,300) ms from actual event onset. Event codes are
202 (TC), 204 (SC), and 207 (GO). MATLAB timestamps are converted from one-based
to zero-based indices. At 1000 Hz, each window has 300 samples; m=3 and tau=3
samples give 294 observed delay vectors.

Each window is detrended, median/MAD-normalized, and filtered independently
using the existing fourth-order 2-55 Hz Butterworth SOS zero-phase filter.
Default odd reflection padding uses only that window. This is offline analysis;
the short duration and filter edges limit low-frequency interpretation.

The existing one-torus fitter's full-dimensional SVD initializes its orientation
and extent estimates; the observed input coordinates are not reduced or replaced.
Its three angular penalty harmonics do not mean three oscillatory modes.
The fifteenth feature is the native band half-width, (R1_out-R1_in+R2_out-R2_in)/4.
It is not the difference between the two outer radii or an out-of-plane thickness.

`inputs/` contains frozen raw/processed windows, clouds, event samples, trial
identities, and grouped folds. `checkpoints/` retains every fitted parameter and
diagnostic, including unusable fits. `tables/` contains plotting inputs and
scores. `models/`, held-out predictions, bootstrap trial indices, and permutation
checkpoints support numerical reproduction. CSVs are reproducibility records,
not additional requested figure formats.

`config.json`, `environment.json`, source snapshots, and hash manifests freeze
the calculation. `validation.json` records checks against all protected previous
outputs. The plotter reads saved results; it does not invoke fitting or decoding.

## Entry points

Use the existing `neuralmanifold` environment. All commands below are explicit
about the new experiment and can be run from any directory.

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python /home/nochen/code/NeuralFieldManifold/notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_stage_geometry_decoding.py --phase all --jobs 6 --resume

/home/nochen/miniconda3/envs/neuralmanifold/bin/python /home/nochen/code/NeuralFieldManifold/notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_stage_geometry_decoding.py --phase verify --resume

/home/nochen/miniconda3/envs/neuralmanifold/bin/python /home/nochen/code/NeuralFieldManifold/notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/plot_stage_geometry_decoding.py --panels all
```

Analysis phases are `prepare`, `pilot`, `fit`, `decode`, `verify`, and `all`.
Plot groups are `geometry`, `features`, `decoding`, and `all`. Resume refuses
changed configurations or calculation sources; fitting reuses cloud-hash-checked
checkpoints. Restarting `all` repeats the inexpensive decoding summaries while
reusing completed fitting and permutation checkpoints.
