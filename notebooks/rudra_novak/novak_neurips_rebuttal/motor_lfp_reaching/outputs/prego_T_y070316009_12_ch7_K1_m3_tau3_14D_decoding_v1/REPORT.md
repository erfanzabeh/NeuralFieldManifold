# Fixed-Geometry Decoding: T, Channel 7

Monkey T, session y070316009-12, LFP channel 7; 198 short-delay trials, 33 per direction. Final 500 ms before GO. Fixed K=1, m=3, tau=3 ms (494 points/trial). Five saved trial-level folds; one held-out prediction per trial. Observed-score confidence intervals are 95% percentile intervals from 2,000 class-stratified resamples of held-out predictions, conditional on the fitted models; folds are not independent recordings. 

## Results

| Representation | Dimensions | Macro-F1 (95% conditional interval) | Accuracy |
|---|---:|---:|---:|
| peak power | 1 | 0.162 (0.116-0.208) | 0.187 |
| peak frequency | 1 | 0.171 (0.123-0.219) | 0.187 |
| power frequency | 2 | 0.186 (0.136-0.238) | 0.192 |
| all bands | 5 | 0.226 (0.169-0.282) | 0.237 |
| geometry | 14 | 0.228 (0.176-0.281) | 0.247 |

Geometry permutation p = 0.016983 (1,000 within-fold shuffles).
This is a test against shuffled labels, not a significance test against another representation.

## Fit Accounting

- 197/198 usable fits; 1 unusable fits.
- Unusable reasons: {'nonconvergence': 1}. All 198 trials still receive predictions.
- Active-bound fits: 12; near-bound fits: 13. These flags overlap and were not exclusion criteria.

## Inputs and Features

- Geometry reuses the exact frozen 500-sample epochs, previously detrended, median/MAD normalized, and zero-phase filtered at 2-55 Hz. No filtering was repeated.
- Clouds are [x(t), x(t-3), x(t-6)], 494 points per trial. Each trial is fitted independently using the existing 1-torus planar elliptical annular-band fitter (lam=0.1, hole_ratio=0.5, Huber loss, three angular regularization harmonics, max_nfev=6000). The three regularization harmonics are not K. There is no PCA, PINN, or shape-matrix representation. The native full-dimensional SVD initialization is unchanged; it does not discard coordinates.
- Decoder columns, in saved order: R1, R2, MSE, mean_error, frac_inside, normal_x/y/z, u_x/y/z, v_x/y/z (14 total). R1 >= R2. The normal and major-axis vectors have their largest absolute component positive; the minor-axis vector is their cross product. No center or width feature.
- MSE is native mean squared 3D distance to the fitted annular set; mean_error is the native in-plane containment-distance average; frac_inside is the fraction whose planar projection falls inside the band. These are distinct, not three equivalent fit errors. Band width remains internal to the unchanged fitter and therefore affects fit-quality measures, although it is not a decoder column.
- Spectral features use the saved raw pre-GO epochs, with linear detrending but without geometric normalization or its bandpass. Peak power is log10 peak PSD and peak frequency is its frequency within 12-40 Hz: Hamming Welch, 300 samples, 200 overlap, nfft=10000. All-band features are log10 integrated powers in 2-4, 4-8, 8-13, 13-30, and 30-55 Hz: Hann Welch, 500 samples, 250 overlap, nfft=10000. Zero padding interpolates the frequency grid, not spectral resolution.

## Evaluation and Interpretation

- All methods use identical trials and saved folds. Median imputation and scaling are fitted on training trials only, followed by lsqr LDA with automatic covariance shrinkage. Each trial contributes exactly one held-out prediction. Macro-F1 is the unweighted average of the six class F1 scores.
- The null preserves the class counts within each saved fold and refits all 25 method/fold pipelines for each of 1,000 shuffles. Only geometry has a reported inferential p-value. Null percentiles are descriptive reference intervals, not uncertainty intervals on the observed score.
- The 2,000 bootstrap samples resample 33 held-out predictions per true class, jointly across methods. Intervals are conditional on the fitted models: no refitting, no recording/day/animal uncertainty.
- This is an exploratory, selected single-recording analysis. Tau=3 ms was visually chosen from earlier examples, outside nested validation; these results are not a confirmatory assessment of that choice. Trial-level folds are not time-blocked. Permutations assume exchangeability within folds and do not preserve temporal trial dependence.
- No superiority claim follows from bar ordering or overlapping/nonoverlapping intervals. These results do not establish the correct topology or demonstrate cross-recording or cross-animal generalization.
- Direction 4 has zero observed true positives for geometry. Its [0, 0] conditional bootstrap interval cannot quantify uncertainty about future performance; resampling these predictions cannot create previously unobserved correct classifications.
- No retuning was performed in response to these scores.

## Files

- `figures/`: six standalone PDF, editable-text SVG, and 600-dpi PNG panels, each with CSV and caption.
- `previews/`: screen-sized copies of each panel.
- `tables/`: pooled scores, per-direction F1, all five confusion matrices, features and trial fit diagnostics.
- `checkpoints/`: all 198 native fit parameters, diagnostics, failures and input-cloud hashes.
- `models/`, `training_audit.json`: fitted fold pipelines and training preprocessing statistics.
- `heldout_predictions.npz`, `permutations.npz`, `bootstrap.npz`: predictions, shuffled labels/predictions, bootstrap trial indices and their scores.
- `config.json`, `provenance.json`, `source/`, `validation.json`: frozen definitions, source/input hashes, analysis source snapshot and numerical checks.

## Regeneration

Plot-only: run `plot_prego_fixed_geometry_decoding.py --output <this directory>`. It reads frozen tables and cannot invoke fitting or decoding.
Validation-only: run `run_prego_fixed_geometry_decoding.py --output <this directory> --verify-only`. Analysis resume requires `--resume` and unchanged configuration/source/input hashes.
