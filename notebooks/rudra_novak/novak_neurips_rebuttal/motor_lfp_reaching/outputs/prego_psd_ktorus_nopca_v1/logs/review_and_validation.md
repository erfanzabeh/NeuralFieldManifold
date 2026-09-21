# Review and validation

## Independent read-only review

An independent coding agent reviewed the scientific implementation, fold-local
preprocessing, trial selection, checkpoints, null controls and reporting.
No observed mathematical or held-out leakage failure was identified.

The review did identify robustness/coverage omissions:

1. Initial null controls covered only five standalone methods. The explicitly
   logged completion stage added the two fused methods, using the same seeds,
   trials and folds. Each method now has 200 null permutations per recording.
2. Fold-specific checkpoint bindings and artifact hashes were missing. The
   guarded runner adds them. The completed original run was independently
   audited against raw full-coordinate data and saved coefficients/features;
   126,920 successful trial/fold fits passed.
3. Interrupted writes could leave a feature file without a diagnostics CSV.
   New runner tests and guards require both before reuse. The current completed
   run had its complete diagnostic files checked by the completion stage.
4. Original behavioral trial IDs were not explicitly checked against the prior
   selection. They now are, alongside trial/fold and spectral-array equality.
5. Resume could overwrite environment metadata. It now rejects a changed
   environment; the completion stage checks the initial environment too.
6. Reporter source was hashed but not snapshotted. Content-addressed reporter
   versions now live in logs/report_source, referenced by validation.json.
7. Render-only example inputs were not integrity checked. Raw source/cache
   tamper tests now enforce these checks. Final table hashes include the
   example-selection table written after the main freeze stage.

The original runner snapshot remains in logs/source; completion guardrails are
in logs/integrity_source. The scientific core was not changed after fitting.
An environment, input or scientific-setting change requires a new experiment.

## Verification performed

- 63 tests pass; see tests.xml for the final run.
- 178,230 stored held-out predictions reproduced from frozen features.
- F1 independently recomputed from class counts; confusion rows sum to one.
- Identical original spectral predictions and predecessor output hashes.
- Eight paired recording-day Wilcoxon tests, Holm adjusted as one family;
  recording-day-clustered bootstrap intervals saved with paired differences.
- Eighteen individual plot sets, each PDF/SVG/PNG/CSV/caption.
- Native PDF text at least 8 pt, no text outside the page, nonblank 600-dpi
  PNGs, editable SVG text. Figures visually inspected for overlap and labels.

## Limits

Tests and audits establish numerical integrity, not geometric adequacy.
Median normalized residuals around 0.81-0.82 limit interpretation of this
constant-frequency model. PSD peaks do not establish phase independence.
Some folds have many features relative to training trials. No significant
added-value improvement survived Holm correction. Detailed results and
exclusions are retained even when unfavorable.

The exhaustive interrupted-process/resume crash matrix was not simulated;
unit tests cover the identified checkpoint/tampering paths. These analyses
are within recording, not cross-animal transfer or multichannel decoding.
