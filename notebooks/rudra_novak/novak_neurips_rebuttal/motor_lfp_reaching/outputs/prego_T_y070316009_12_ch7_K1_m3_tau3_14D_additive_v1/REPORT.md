# Updated Comparison and Direction-Colored Geometry

Two requested figures only. The original experiment and its plots remain unchanged.

## Comparison

| Representation | Dimensions | Macro-F1 | Conditional 95% interval |
|---|---:|---:|---:|
| relevant band | 1 | 0.139 | 0.105-0.177 |
| geometry relevant band | 15 | 0.227 | 0.175-0.283 |
| average psd | 1 | 0.169 | 0.121-0.216 |
| geometry average psd | 15 | 0.233 | 0.177-0.289 |
| all bands | 5 | 0.226 | 0.169-0.282 |
| geometry | 14 | 0.228 | 0.176-0.281 |

Six-direction decoding from 198 short-delay trials (33/direction), Monkey T session y070316009-12, channel 7, final 500 ms before GO. Fixed K=1, m=3, tau=3 ms. Bars: pooled held-out macro-F1, using the same saved five folds and training-only median imputation/scaling with shrinkage LDA. Whiskers: 95% percentile intervals from 2,000 class-stratified resamples of held-out predictions, conditional on fitted models. Relevant band: log10 integrated beta PSD (13-30 Hz), 1D. Average PSD: log10 arithmetic mean density across 2-55 Hz, 1D. Torus + baseline: concatenate the unchanged 14D geometry with the 1D baseline, 15D. All band powers: 5D; geometry: 14D. The dashed line at 1/6=0.1667 is the nominal balanced uniform-guessing reference, not a method-specific empirical macro-F1 null or a significance threshold. No shuffled-label markers, gridlines, or significance stars are displayed. No geometry was refitted. One unusable fit remains in the trial cohort via training-fold imputation. This is exploratory single-recording performance; bar differences are not significance tests.

All spectral features here refer to the saved raw pre-GO epochs with linear detrending. Average PSD uses the same Hann/Welch settings as this run's frozen all-band features (500 samples, 250 overlap, nfft=10000), then takes log10 of the mean density at 2-55 Hz. It is not the average of five log band powers. Unlike the legacy movement analysis's average-PSD helper, this definition does not re-normalize amplitude per trial and does not change the frozen all-band comparison. Old movement-aligned scores are not reused.

## LDA View

Descriptive supervised LDA projection of the 14 geometric features, colored by true reach direction. A single common projection is fitted on the 158 training trials of saved fold 0; 40 held-out trials are transformed without refitting. Open faint points: training; filled points: held out. All 198 trials are shown. Contours and marginal densities use training trials only; two-dimensional contours enclose approximately 50% and 80% of each estimated class density on the displayed training-derived grid. Gaussian KDE uses Scott bandwidths. The projection uses eigen-solver LDA with automatic shrinkage solely to expose two visualization axes; classifier scores use the unchanged lsqr shrinkage LDA on full feature vectors. LD1 and LD2 are centered and scaled using training coordinates only. No PCA is used. Axis signs are deterministic; axes from different folds are not pooled. One unusable geometric fit is imputed with training medians. Training separation is supervised and is not evidence of held-out accuracy; the held-out points represent one fold, not all five-fold predictions.

The 14D features and K/m/tau settings are unchanged; the plot is a view of feature space, not a 2D replacement for fitting or decoding. Colors identify directions, not decoding performance. Displayed training densities must not be read as generalization.

## Files and Reproduction

- `figures/`: each panel has PDF, editable-text SVG, 600-dpi PNG, caption and CSV.
- `models/`, `features.npz`, `heldout_predictions.npz`, `bootstrap.npz`: full numerical record.
- `projection.npz` and its CSV: common-axis coordinates, roles, labels, folds and projection transform.
- `tables/projection_density.npz`: saved KDE grids, marginal densities and contour levels.
- `provenance.json`, `parent_manifest.json`, `numerical_environment.json`, `source/`: tracking.
- Analysis: `run_prego_fixed_geometry_additive.py`; plot-only: `plot_prego_fixed_geometry_additive.py`.
- Saved geometry/all-band held-out predictions and bootstrap values exactly match the parent run.
- No new significance tests were run; no claim of additive superiority follows from bar ordering.
