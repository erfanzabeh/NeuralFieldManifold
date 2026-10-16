# Macaque pre-GO OLS audit

## Bottom line

The prescribed audit was run on all 210 eligible single-channel recordings,
using exactly the earlier balanced short-delay trial identities and five outer
folds. Ordinary AR models predict these preprocessed signals very well over
10 ms. However, the auxiliary embedding selection reaches the search boundary
in every fold. **This audit does not identify the intrinsic manifold dimension,
the number of oscillatory modes, or an appropriate torus model.**

No PCA, torus fitting, shape-matrix features, PINN training, or decoding was
performed. The previous experiments were checked against their initial file
hashes and remain unchanged.

## Results

| Cohort | LFP recordings | Recording days | Balanced trial-by-recording epochs |
| --- | ---: | ---: | ---: |
| Monkey M | 115 | 48 | 13,338 |
| Monkey T | 95 | 30 | 12,468 |

These are two animals, not 210 independent animals. Trial epochs from different
channels can correspond to the same behavioral trial. Recording-level SD is
descriptive; it is not an animal-level confidence interval.

Held-out 10-ms NMSE, recording-level mean +/- SD (lower is better):

| Predictor | Monkey M | Monkey T |
| --- | ---: | ---: |
| Selected AR, recursive | 0.002748 +/- 0.001084 | 0.002488 +/- 0.000793 |
| Selected lag-vector direct readout | 0.004698 +/- 0.002118 | 0.003177 +/- 0.001027 |
| Persistence | 1.526763 +/- 0.339089 | 1.733110 +/- 0.470493 |

The corresponding mean AR R-squared is 0.997252 for M and 0.997512 for T.
These are signal-prediction scores, not direction-decoding accuracy or F1.
One-step AR NMSE is approximately 2.8e-11 and 2.6e-11, respectively; the
near-perfect one-step result must be interpreted in the preprocessing context.

- AR p: fold-wise median 18 for M and 16 for T. Choices span 11-20 and 12-20,
  respectively. The most frequent recording-level modal p is 19 for M
  (56/115 recordings) and 14 for T (32/95 recordings).
- p=20 is selected in 13/575 M folds and 16/475 T folds. These boundary choices
  are explicitly flagged, without enlarging the grid.
- Embedding: all 1,050 outer folds select m=9 and tau=1 ms. Thus every embedding
  choice is at the upper dimension boundary and lower delay boundary. The
  selected lag vector spans only 8 ms, not the full permitted 200 ms.
- Median within-recording agreement on p is 4/5 folds for M and 3/5 for T.
- Median standardized-design condition numbers are about 1.31e8 for selected
  AR models and 2.77e7 for embedding readouts. All fits are numerically full-rank
  under the recorded least-squares tolerance, but **full rank does not imply
  well-conditioned coefficients**.

## Interpretation

1. **Predictive adequacy:** conventional OLS AR models successfully describe
   short-horizon variation in these specific preprocessed epochs, including
   recursive forecasts without inserting intermediate true values.
2. **Geometry remains unresolved:** nearby samples of a smooth signal support
   accurate prediction. Their predictive usefulness does not establish an
   embedding that unfolds the attractor or supplies independent phase coverage.
   Neither p nor m should be substituted for the mode count K.
3. **Preprocessing matters:** each 500-ms epoch is detrended, robustly scaled,
   and zero-phase bandpass-filtered at 2-55 Hz at a 1-kHz sampling rate. Scaling
   and filtering use the whole epoch, including times later than a forecast
   origin, although all data are before GO. These are offline preprocessed-
   signal predictions, not causal online forecasts. Filtering and oversampling
   can substantially influence apparent predictability and selected AR order.
4. **Do not freeze a geometric decoder from these choices:** m=9/tau=1 is the
   winner of a predictive grid, not a validated manifold dimension/delay.
   Strong collinearity also limits interpretation of individual AR coefficients
   and poles. No fitted pole pair is automatically classified as a sustained
   neural oscillatory mode.

The next decision is therefore about how to validate a geometry-appropriate
embedding, not whether to accept high prediction R-squared as proof of a torus.
No further experiment has been started.

## Where to look

- `plots/summary/selected_parameters.*`: all five fold-specific p, m, tau
  distributions, showing the full candidate grids.
- `plots/summary/modal_ar_order.*` and `selection_agreement.*`: recording-level
  summaries and fold-to-fold variation.
- `plots/summary/monkey_M_order_sweep.*` and `monkey_T_order_sweep.*`: matched
  AR order curves, including persistence.
- `plots/summary/monkey_M_embedding_sweep.*` and `monkey_T_embedding_sweep.*`:
  complete dimension/delay error maps; gray means span >200 ms.
- `plots/summary/heldout_prediction.*`: held-out recording scores versus
  persistence; the common axis is logarithmic.
- `plots/summary/selected_conditioning.*`: explicit conditioning diagnostics.
- `plots/examples/`: prediction traces, residuals, and original-coordinate
  subsets for one label-blind illustrative recording per monkey. Choose the
  recording nearest the animal's median fold-0 inner-validation AR NMSE, then
  its fold-0 test trial with the lowest original trial number. Other folds'
  scores do not influence example selection. The separate time traces show
  every selected coordinate, without PCA.
- `plots/recordings/<recording>/`: each recording's standalone order curve and
  embedding heatmap. Every figure has separate PDF, editable-text SVG, 300-dpi
  PNG, underlying CSV, and caption files. No panel letters are embedded.

## Tables and reproducibility

- `tables/cohort.csv`, `recording_audit.csv`: complete cohort and GO-alignment
  metadata. Every recording is present; there are no recording failures.
- `tables/inner_validation_scores.csv`: all 598,500 candidate/horizon/inner-fold
  score rows, with coefficients, rank, conditioning, and undefined-metric flags.
  Coefficients are serialized matrices; row zero is the intercept. AR matrices
  have one column; direct embedding matrices have columns for 1 and 10 ms.
- `tables/validation_scores.csv`: all 199,500 candidate/horizon/outer-fold
  validation summaries. `candidate_failures.csv` is empty apart from its header
  because all candidates produced finite NMSE and full numerical rank.
- `tables/selected_models.csv`: the 2,100 selected outer models, selection
  thresholds, boundary flags, ranks, and condition numbers.
- `tables/heldout_trial_scores.csv`, `heldout_recording_scores.csv`,
  `population_scores.csv`: equal-trial scores and matched recording summaries.
- `tables/coefficients.csv`: selected model coefficients in original
  preprocessed-signal units. AR one-step coefficients also define the recursive
  forecast; there is no separately fitted 10-ms AR model.
- `tables/selected_ar_poles.csv`, `inner_ar_poles.csv`: complex poles, magnitudes,
  and signed frequencies for selected and swept AR fits. Poles solve
  z^p - a1*z^(p-1) - ... - ap = 0; the intercept is excluded.
- `cache/<recording>/trials.npz`: frozen processed epochs and exact saved trial
  identities, labels, and outer folds.
- `cache/<recording>/fold_*/`: inner split indices, models, full validation
  scores, held-out predictions, and artifact hashes. Split indices index rows
  in that recording's `trials.npz`, not concatenated time samples.
- `config.json`, `manifest.json`, `source/`: immutable scientific configuration,
  original analysis source copies, environment versions, all raw/converted
  source hashes, and all predecessor output hashes.
- `verification.json`, `tables/numerical_reconstruction_checks.csv`: independent
  reconstruction and integrity checks. All 1,050 folds passed. Maximum forecast
  difference versus an extended-precision sequential recurrence is 4.1841e-7
  normalized signal units, below the explicit absolute limit of 1e-6.
- `figure_manifest.json`: plotting-code hash, input-table hashes, and figure
  artifact hashes. Plotting is independently repeatable and never fits models.
- `OLS_Audit_Overview.pdf`: a vector summary booklet, in addition to the
  standalone figures; `figure_verification.json` records export-quality checks.

### Exact metric and selection conventions

For each trial and horizon, NMSE is sum of squared prediction errors divided by
the sum of squared deviations of the target samples from their trial mean.
R-squared is 1-NMSE and may be negative. Pearson correlation is computed within
trial. Persistence predicts x(t) at both horizons. `nmse_improvement` is
NMSE(persistence)-NMSE(model); `persistence_skill` is 1-NMSE(model)/NMSE(persistence).
Scores are averaged equally over held-out trials, then equally over recordings.

Inner selection first averages trial NMSE within each validation fold, then
averages the three fold means. The one-SE allowance is their sample SD/sqrt(3),
used as a selection heuristic, not an inferential confidence interval. Choose
the smallest qualifying p; for embeddings choose the smallest qualifying m,
then the lowest validation-error tau at that m, then the smaller tau on a tie.
Singular or nonfinite candidates would be retained but excluded from selection.

All models include an intercept. Design columns are standardized using only
training samples for numerical stability and coefficients are transformed back;
there is no regularization or dimension reduction. The inner splitter's actual
seed is derived deterministically from the recording-ID SHA256 (first eight
hex digits, modulo 2^31-10), plus outer-fold number. All actual split indices
are saved. The generic `seed` field in the configuration is not used by this
recording-ID-based splitter.

## Entry points

From the repository root, use the `neuralmanifold` Python environment:

```bash
python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_prego_ols_audit.py --resume --jobs 8
python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/report_prego_ols_audit.py --stage verify
python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/report_prego_ols_audit.py --stage tables
python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/report_prego_ols_audit.py --stage plots
```

The analysis refuses changed source/input hashes or configuration on resume.
It reuses verified complete checkpoints, rather than refitting them. The four-
recording benchmark completed in 13.8 seconds; the subsequent full-cohort run
took approximately 3.8 minutes with eight workers, reusing those four checkpoints.
Verification and figure production are additional to this compute time.
