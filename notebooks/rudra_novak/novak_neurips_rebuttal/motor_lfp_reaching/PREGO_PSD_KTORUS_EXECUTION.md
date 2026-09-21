# Execution ledger: prego_psd_ktorus_nopca_v1

The user approved implementation and execution. Work stays in the existing
workspace to honor the requested absolute experiment location and avoid moving
the large raw/converted datasets. Existing dirty `.gitignore` and diagnostic
files are left untouched. No automatic commit or push of experiment outputs.

## Work sequence

1. Test and implement K-aware full-coordinate geometry and fold-local decoding.
2. Test and implement an isolated, hashed, locked, resumable experiment runner.
3. Run synthetic validation, then the complete primary short-delay cohort.
4. Independently reproduce metrics, statistics and old-output hashes; render
   and inspect new figures; write a candid results report.

## Decisions

- Implement the accepted affine-product-of-circles specification explicitly.
  Its parameters are not the predecessor's 15 tube-fit features.
- Keep the predecessor's exact eligible cohort/trial/fold selection as the
  starting population. Geometry-unresolved recordings will be reported, not
  silently assigned K=1 or projected to three dimensions.
- Rank/condition computations and least-squares solves may use numerical SVD;
  this is not PCA and does not discard input coordinates.
- The first run covers short-delay main results only, as specified.

## Progress

- Specification accepted; implementation started.
- Synthetic and leakage-boundary tests written before geometry implementation.
- Geometry tests failed for the absent module, then passed after implementation.
- Runner tests failed for the absent module, then passed after implementation.
- First 18 tests passed. Added explicit PSD recovery checks for K=1,2,3.
- Full-dimensional variable-projection regression keeps all lag residuals. It is
  a locally quasiperiodic model; it is not claimed to prove topology or to be
  equivalent to the old elliptical tube fit.
- All 49 macaque tests passed after adding reporting tests; JUnit evidence is
  stored in the new run's `logs/tests.xml`.
- Full 341-recording run launched with 12 workers, 120 peak bootstraps and 200
  method-specific null permutations; no model settings were tuned on results.
- Reporting code separately reproduces classifier predictions from frozen
  features, and checks all four spectral predictions against the predecessor.
- Primary fit/decode finished across all 341 source recordings. Of 210
  trial-count-eligible recordings, 203 have resolved geometry in all five folds
  (110 M, 93 T). The seven others retain their spectral-only results.
- Independent code review found no observed fitter-math or held-out leakage
  failure, but identified five robustness/coverage gaps. Regression tests failed
  before fixes and passed afterward: all seven null methods, fold bindings,
  interrupted diagnostic writes, original trial IDs, and environment identity.
- The scientific fitting code stayed byte-identical throughout the run.
  Completion uses a separately snapshotted, explicitly logged stage to audit
  saved fits against raw full-coordinate clouds and append fused-method nulls;
  original held-out predictions must remain unchanged.
- All 59 tests passed after checkpoint fixes. Original-coordinate display and
  frozen-table rendering tests were then added, failed before implementation,
  and passed after implementation. Figure views do not change fitting inputs.
- Completion finished: 126,920 successful trial/fold fits independently checked
  against their full-coordinate inputs. All seven methods have 200 label-null
  permutations per geometry-evaluable recording. Predictions were unchanged.
- Reporting reproduced all 178,230 held-out predictions and all four original
  spectral decoders, independently recomputed F1, verified confusion-row sums,
  and confirmed predecessor output hashes were unchanged.
- Follow-up review prompted archived reporting source and integrity checks for
  render-only examples. New tamper/source-coverage tests failed first, then
  passed. All 63 tests passed, including figure values and integer K/m ticks.
- Eighteen standalone panels exported as PDF, editable-text SVG and 600-dpi PNG,
  each with CSV and caption. Visual inspection corrected crowded annotations
  and legends. PDF checks found minimum 8-pt text and no out-of-page text; all
  PNGs were nonblank and all SVGs retained editable text.
- Scientific result: geometry alone did not beat the two spectral baselines;
  no added-geometry gain survived the eight-test Holm family. Geometry was
  significantly worse than power/frequency in T (adjusted p=0.003681).
  Median full-coordinate normalized residuals were 0.811 (M) and 0.820 (T).
  These weak fits are an explicit limitation, not grounds for tuning this run
  after inspecting decoding scores. No PINN or second experiment was launched.
