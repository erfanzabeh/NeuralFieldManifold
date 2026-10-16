# T channel 7: fixed K=1, m=3, tau=3; 14D decoding

Approved scope: 198 saved short-delay trials, 33 per direction; final 500 ms
before GO; frozen preprocessing; native one_torus_fit; no width input, PCA,
shape matrix, PINN, or parameter search. Five existing folds and five standalone
representations; 1000 permutations and 2000 conditional prediction bootstraps.

## Execution

1. Features and evaluation tests: complete, including canonical axes, exact
   feature width, unusable fits, train-only preprocessing, metric reproduction,
   permutation refitting, and frozen-coordinate indexing.
2. Frozen fitting/decoding/permutation run: complete. The repeat execution
   exactly reproduced all 198 fit checkpoints and 22 numerical result arrays.
3. Separate EEG-style figure exports and captions: six panels complete and
   visually inspected, with final PDF/SVG/600-dpi PNG and underlying tables.
4. Numerical reproduction and rendering inspection: complete. Review found
   one resume-environment defect, fixed with a regression test. All PDF labels
   are within bounds, at least 8 pt; all SVGs contain editable text.

## Decisions and safeguards

- Work in the existing novak/eeg_decode_update checkout: the approved inputs and
  helpers are uncommitted there and the user explicitly requested a sibling
  output experiment. Preserve unrelated edits; do not commit, push, or relocate.
- Keep prior five-fold, within-fold label-permutation logic, and shrinkage LDA.
  Geometry has 14 columns: R1, R2, MSE, mean error, fraction inside, then the
  three components of the normal and each ellipse axis. Names lock their order.
- Retain optimizer-bound and near-circular-axis flags; exclude invalid,
  nonconverged, or nonfinite fits from feature extraction, not from the trial
  cohort. All 14 values for an unusable fit are missing and imputed training-only.
- Bootstrap intervals resample class-stratified held-out predictions; they are
  conditional on trained models and assume exchangeable trials within class.
  Null p-value is primary for geometry only; other nulls remain descriptive.
- This is one selected recording after label-blind visual delay inspection,
  not independent confirmatory validation or a population/day-level analysis.
- Baseline test state from preceding completed task: 138 passing and three
  pre-existing poster artifact failures. Poster content is out of scope.

## Review and repeat-run record

- Initial complete run: geometry macro-F1 0.227921, all bands 0.225966,
  power/frequency 0.186021; geometry null p 0.016983. 197 usable fits, one
  nonconverged fit retained via missing-feature imputation. No result-driven edits.
- Review identified that the environment log could be overwritten on resume.
  The runner now freezes Python/NumPy/SciPy/scikit-learn/joblib versions and
  machine architecture; any mismatch stops before checkpoint reuse or writes.
- First pass preserved at `/tmp/nfm_fixed_geometry_validation_R1Ns14/first_pass`
  for a full comparison. The approved output directory is regenerated with the
  final source hashes, the same configuration, and the same seeds. Earlier
  experiments are not moved or changed.
- First full-suite verification: 156 passed, the same three unrelated poster
  failures as the baseline, and 10 subtests passed.
- Final full-suite verification after the guard fix: 157 passed, the same
  three unrelated poster failures, and 10 subtests passed. All 19 new tests pass.
- Final output contains REPORT.md, QUALITY_CHECKS.md, full machine-readable
  results and validation, frozen source/configuration/environment, and six
  standalone panels. Earlier 25,312 result files passed unchanged-hash checks.
