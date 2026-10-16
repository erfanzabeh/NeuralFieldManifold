# Pre-GO OLS audit: accepted plan and execution record

## Fixed scope

Audit the 210 previously decoding-eligible recordings (M 115, T 95), using the
saved balanced short-delay trial identities and five outer folds. Every epoch
is the final 500 samples before the raw-audited GO event at 1 kHz. No decoding,
geometric fitting, shape descriptors, PCA, or PINN is performed.

AR candidates are orders 1 through 20. Delay predictors use dimensions 2
through 9 and delays 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100 ms, retaining
only spans <=200 ms. All candidates use origins 200 through 489 inclusive.
AR coefficients are fitted to next-sample targets and evaluated at horizons
1 and 10, with the latter a recursive forecast without truth refresh. Delay
predictors are separate direct OLS readouts at those two horizons.

Select settings within three inner trial-level folds of each outer training
set, using mean trial-normalized 10-ms squared error and the one-standard-error
rule. Test data do not select settings. Keep raw per-fold selections and all
candidate validation results, including failures. A modal recording summary
is descriptive only. Report baseline-relative errors, R2, correlation,
conditioning, coefficients and AR poles; no torus quality claims.

## Rulings

- Work in the existing `novak/eeg_decode_update` checkout: the approved plan
  fixes this workspace and sibling output path, and depends on uncommitted
  prior artifacts. Do not move, revert, or commit existing work.
- Use the existing independent per-epoch detrend/robust-scale/2-55-Hz filter.
  These are offline preprocessed-signal diagnostics, not causal online
  forecasting, because zero-phase filtering and scaling use the whole epoch.
- Average normalized errors equally across trials within each validation
  fold; the selection mean and standard error use the three fold means.
  Standard error is sample SD / sqrt(3). Singular candidates cannot be selected.
- Center/scale design columns using training rows only for numerical stability,
  then convert coefficients back to the original preprocessed-signal units.
  This is unregularized OLS, not a dimension reduction or ridge regression.
- Among embeddings within the one-SE threshold, choose minimum m, then minimum
  mean validation NMSE among qualifying delays at that m, then minimum tau.
- Report all roots of the selected AR models without defining a new threshold
  for sustained modes. Roots are not automatically interpreted as topology.

## Execution Ledger

1. Numerical core and leakage tests: complete; 15 tests pass (core + runner).
2. Isolated runner, checkpoints and four-recording benchmark: complete. Four
   recordings took 8.8-12.9 seconds each; 13.8 seconds concurrent wall time.
   Projected full computation: 5-10 minutes at eight workers, plus validation
   and figure production. Both animals' median and largest trial counts tested.
3. Complete cohort execution: complete; all 210 recordings have five completed
   outer folds, with no recording failures.
4. Frozen-table reporting and standalone figures: complete; 435 standalone
   figures (420 recording-specific + 15 summary/example figures), each as PDF,
   SVG, PNG, CSV and caption. Separate plot-only entrypoint; summary booklet.
5. Independent result checks, visual inspection and closeout: complete.
   All 1,050 outer folds reconstruct; all 210 recordings retained; all prior
   output and input/source hashes verified. The 86-test suite passes. Every
   figure has nonblank PNG and in-bounds PDF text; sweep CSVs and summary
   markers match frozen tables. Representative summary and recording figures
   inspected visually; log-axis and label-layout problems corrected.

## Review and qualifications

- An independent code review found no remaining important core/runner bugs.
  Its example-selection finding was corrected: examples use only fold-0 inner
  validation scores before displaying a fold-0 test trial. A regression test
  prevents other folds from changing the illustrative recording.
- Independent metric checks promote saved float32 arrays to float64, matching
  the original metric arithmetic. No scientific results were changed.
- High-order recursive predictions were also reconstructed with an extended-
  precision sequential recurrence. Maximum difference: 4.1841e-7 normalized
  units, below an explicit 1e-6 absolute tolerance. Per-fold discrepancies saved.
- Every embedding selected m=9/tau=1 ms, so the result lies at grid boundaries.
  This is a predictability finding, not an intrinsic dimension or torus claim.
- Selected designs are full numerical rank but ill-conditioned. Strong
  short-horizon accuracy after zero-phase filtering does not establish causal
  prediction, reliable physical AR coefficients, or the correct neural geometry.

No experiment is complete merely because an optimizer or command finished.
