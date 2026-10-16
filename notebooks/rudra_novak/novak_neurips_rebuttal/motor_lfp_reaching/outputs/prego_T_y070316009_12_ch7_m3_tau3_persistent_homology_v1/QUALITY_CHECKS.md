# Validation record

## Input and calculation
- Exactly 198 saved short-delay trials, 33 per direction, 494 x 3 points each.
- Each coordinate equals its frozen epoch sample exactly: offsets 6, 3, 0 over
  494 rows; latest epoch indices 6 through 499. Sampling is 1 kHz and all samples
  precede GO (recording indices 4000 through 4499; GO index 4500).
- No cloud transformation, sphere projection, PCA, geometric refit, or decoder.
- Ripser.py 0.6.15: Euclidean, coeff=2, maxdim=2, thresh=infinity, n_perm=None.
- Six lowest original IDs formed the pilot. All 198 calculations succeeded,
  using approximately 38.03 minutes of measured worker time. No timeouts.
- All six pilot diagram/checkpoint pairs retain their original hashes. The final
  resume reused all 198 checkpoints without executing Ripser again.

## Numerical checks
- 13 scoped tests pass: exact cloud provenance, synthetic square H1 and octahedral
  H2 intervals, half-open Betti boundaries and infinite intervals, empty versus
  nonfinite lifetimes, translation/rotation/scaling behavior, CSV missing-value
  parsing, source-calculation immutability, success/failure checkpoint reuse,
  direction/example selection, plotted trial values, saved ellipse outlines,
  and fresh-process plot import isolation.
- Independent validator reproduced 129,862 interval rows and 304,128 sampled
  Betti values using sorted birth/death event counts, a separate implementation
  from the analysis's interval-broadcast calculation.
- Means, quartiles, trial counts, normalized lifetimes, radius validity, all
  example identities, and PNG source coordinates reproduce the saved inputs.
- Infinite H0 intervals remain in diagrams and curves. No beta value or
  threshold-based persistent-loop count is imposed on the measurements.
- All 25,822 protected previous files match their before-run hashes.

## Visual checks
- Inspected all 19 PNGs at reading scale in four QA sheets and an individual
  geometry/persistence preview. White backgrounds; no background grids,
  significance stars, panel letters, decorative boxes, or interpretation text.
- All dots, extrema, curve ranges, labels, and legends are visible and readable.
- All geometry examples have identical limits/viewpoints; all six persistence
  diagrams have identical limits. All six directions share each Betti grid.
- All final PNGs have 600-dpi metadata and nonblank pixels; each panel is separate.
  No PDF or SVG was generated. Low-resolution QA sheets are previews only.
- A fresh plot-only rerun reproduced all 19 PNGs byte-for-byte and preserved all
  396 diagram/checkpoint files. No Ripser, decoder, or fitter module was imported.

## Review and implementation correction
An independent read-only reviewer found a CSV-verification type mismatch involving
the intentionally missing radius, an unchecked example table, and a test that
mistook other tests' imports for renderer imports. Regression tests cover all
three corrections. The already-running parent's old verifier stopped after all
198 diagrams and tables had been written. The corrected runner then verified
them successfully, reusing every checkpoint.

The initial source snapshot and provenance were not rewritten. Verification-only
source revisions are archived separately; an AST check confirms that calculation
functions and global settings are unchanged. No result was retuned or recomputed.

The existing model-training/decoder/fitter test suites were deliberately not run,
because this task excludes those model calls. Validation covers the new isolated
experiment rather than claiming repository-wide test coverage.
