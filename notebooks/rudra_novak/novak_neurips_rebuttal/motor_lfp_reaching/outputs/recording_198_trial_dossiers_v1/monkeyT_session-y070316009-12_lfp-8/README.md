# Monkey T | 2007-03-16 | channel 8

Recording ID: `monkeyT_session-y070316009-12_lfp-8`. Selected because it has 198 trials, not because of its decoding performance.

## Start here

- [Recording_Report.pdf](Recording_Report.pdf): decoding, all confusion matrices, per-direction F1, geometric residuals, saved PSD/AMI selection, example traces, and the existing AR-order curve.
- `inputs/selected_trials.csv`: exactly 198 trials, 33 in each of six directions. Fold IDs are zero-based.
- `decoding_no_pca/cache/predictions.csv`: every saved held-out prediction.
- `decoding_no_pca/spectral_features.csv` and `geometry_features_fold_*.csv`: readable feature matrices.
- `ols_audit/tables/`: all recording-specific order sweeps, coefficients, poles, selections, and held-out signal-prediction scores.

## Trial accounting

The source recording contains 404 trials: 203 short-delay and 201 long-delay.
The selected 198 are balanced short-delay trials only. Every trial contributes one 500-ms segment from this channel.
This is a single recording, not a pooled multi-channel decoder. Other channels can observe the same behavioral trials.

Test counts by fold and direction:

| heldout_fold_zero_based | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7 | 7 | 6 | 7 | 6 | 7 |
| 1 | 7 | 6 | 7 | 7 | 6 | 7 |
| 2 | 7 | 6 | 7 | 6 | 7 | 7 |
| 3 | 6 | 7 | 7 | 6 | 7 | 6 |
| 4 | 6 | 7 | 6 | 7 | 7 | 6 |

Sampling rate: 1000 Hz. Audited GO is zero-based sample 4500; use samples
[4000:4500] for the pre-GO interval. Each short delay is 700 ms.
IMPORTANT: `inputs/full_converted_recording.npz` preserves the old conversion exactly, including its legacy GO field.
For alignment, use `inputs/audited_alignment.json` or the already-correct `selected_raw_prego_epochs.npz`.

## Earlier no-PCA decoding

| method | macro_f1 |
| --- | --- |
| Peak power | 0.221324 |
| Peak frequency | 0.209823 |
| Power + frequency | 0.272800 |
| All band powers | 0.244339 |
| K-mode geometry | 0.171177 |
| Geometry + power/frequency | 0.209018 |
| Geometry + all bands | 0.220834 |

These are macro-F1 values pooled across the held-out predictions, not accuracy and not a mean across recordings.
The report shows empirical central 95% ranges of the 200 saved shuffle scores. These are not confidence intervals.
No new significance tests or claims of above-chance performance are introduced in this dossier.

Fold-specific geometry choices from THAT decoding run:

| fold | K | dimension | tau_samples | feature_count | frequencies_hz |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 3 | 13.0 | 10 | [22.0] |
| 1 | 2 | 5 | 13.0 | 36 | [22.0, 46.0] |
| 2 | 2 | 5 | 13.0 | 36 | [22.0, 46.0] |
| 3 | 2 | 5 | 13.0 | 36 | [22.0, 46.0] |
| 4 | 2 | 5 | 13.0 | 36 | [22.0, 46.0] |

The geometric feature vector contains the upper-triangular mode-specific shape matrices, coordinate residual RMSEs,
and total normalized residual. It is NOT the original 15D R1/R2/r tube feature vector. The named CSV columns expose every feature.
For one mode in 3D, this is 6 shape terms + 3 coordinate errors + 1 total error = 10 features.
Feature dimensions may differ across folds; do not concatenate fold-specific features as if they were one fitted coordinate system.
Median held-out geometric residual SSE/TSS: 0.738124; successful-fit fraction: 1.000000.
Numerical convergence does not establish good geometric recovery or a true torus topology.

Spectral peaks: 12_40Hz; power is log10 peak PSD and frequency is peak location in Hz.
The five band features are log10 integrated delta, theta, alpha, beta, and low-gamma power.
Classifier: training-fold median imputation, standardization, and shrinkage LDA. Only saved predictions are used here.
There is no serialized fitted LDA object in this package; the features and fold assignments needed to refit are included.

## Later OLS audit: NOT a new decoder

Selected AR orders in fold order 0-4: 16,16,19,16,19.
Held-out 10-ms recursive AR NMSE: 0.002399; R-squared: 0.997601.
These measure waveform prediction, not direction decoding. Coefficients, intercepts, numerical conditioning,
poles and complete validation candidates are retained in `ols_audit/`.
The auxiliary lag-readout m=9/tau=1 results remain archived there for completeness; they are not emphasized in this report
and were not used to update the geometric decoder. Preprocessing uses the whole pre-GO epoch, so prediction is offline, not causal online forecasting.

## Historical material

`historical_pca_predecessor/` preserves the earlier pre-GO tube-feature/PCA pipeline, including matched short/long analyses.
Those features, trial subsets and null controls must not be confused with the main no-PCA results above.
The underlying run configurations and source paths are in `../source_context/` and `../source_manifest.csv`.
Earlier movement-aligned experiments are not relabeled as these 198-trial pre-GO results; their project source remains at `/home/nochen/code/NeuralFieldManifold/notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching`.

## Verification

The selected trial identities, directions and folds match between the no-PCA decoding and OLS audit.
All seven macro-F1 scores were recomputed from saved predictions and checked against saved tables; all confusion rows sum to one.
No model was fit, no new decoding was run, and all packaged source files were checked for unchanged hashes.
