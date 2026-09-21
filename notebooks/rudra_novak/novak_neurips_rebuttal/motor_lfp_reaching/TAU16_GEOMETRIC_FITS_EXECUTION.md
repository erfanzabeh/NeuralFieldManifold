# T channel 7, fixed-delay geometric fitting

Approved scope: 198 saved short-delay trials; final 500 ms before GO; original
preprocessing; m=3, tau=16 samples; existing one_torus_fit and two_torus_fit;
fitting report only. No decoding, shape-matrix features, or PINN.

## Execution record

- Input audit complete: 198 x 500 samples, six directions x 33; audited GO=4500,
  epochs [4000:4500], fs=1000 Hz. Output directory does not yet exist.
- Working on the existing novak/eeg_decode_update branch because this experiment
  depends on the user's uncommitted prior analysis and expressly named output
  location. No existing changes will be reverted, committed, or moved.
- Tests added first for optional diagnostics, native-result identity, fixed
  coordinate indexing, metric definitions, failure flags, and example selection.
- Shared fitter edits will only add opt-in diagnostics; objectives, optimizer
  arguments, defaults, and default return values remain unchanged.
- Important distinction: the package's one-torus model is an annular band,
  not a thin ellipse. Its native R-squared and coverage ignore plane offset;
  its native MSE includes that offset. The two-torus metrics measure containment.

## Completion

1. Diagnostics and fitting runner: complete.
2. All 396 trial/model fits: complete; 396 returned, no exceptions.
3. Plot-only report, all-trial atlas, tables, and provenance: complete, including
   the final review-driven diagnostic annotations.
4. Final export inspection and regression tests: complete; unrelated baseline
   poster failures remain unchanged. No decoding was run.

## Progress

- Diagnostics and core tests: RED -> GREEN, 8 passed. Default versus diagnostic
  native outputs are bitwise identical on synthetic test clouds.
- Runner tests: RED -> GREEN, 3 passed, including exact raw-epoch verification.
  The trusted legacy source NPZ requires pickle for its object-array delay labels;
  generated arrays and checkpoints do not use pickle.
- Repository-wide baseline: 120 passed, 3 pre-existing poster artifact tests failed
  (test_powerpoint_is_single_editable_poster, test_required_sources_and_metric_caveats,
  test_rendered_text_stays_inside_its_editable_box). Poster files are out of scope.
- Four-trial benchmark started; all prior output files are hashed read-only.
- Benchmark complete: 8/8 fits returned in 0.03-0.25 seconds each. Full 198-trial
  run launched with the same immutable configuration, reusing those checkpoints.
- Plot-only report added with full per-trial atlas, separate metric definitions,
  source tables, and independent reproduction of each native distance calculation.
- All 396 fits complete: one-torus converged 198/198; two-torus 197/198.
  Two-torus collapsed-hole diagnostic triggered on 99/198; all were retained.
  Median normalized 3D set errors: one-torus 0.244798, two-torus 0.001245.
  This is not a topology comparison with equal model flexibility.
- Fitting/input hashes and 24,303 previous output files verified unchanged.
  Six actual trial/model checks exactly reproduce original committed fitters.
- Rendering QA caught a long diagnostic label crossing columns. Regression test
  failed first, then passed after wrapping and explicit column spacing.
- Independent read-only review confirmed the fitting/data invariants and raised
  three P2 reporting/verification findings, now addressed with failing-then-passing
  regression tests: parametric sweep labels for thick/self-intersecting tube
  overlays; per-example collapse/convergence/bound warnings; full exported
  identity, geometry, and diagnostic checks against native checkpoint results.
- Final annotated exports checked: 215 nonblank PDF pages, no text outside page
  boundaries, all PNGs 600 dpi, SVG text editable. Overview, compact examples,
  and individual paired overlays were inspected visually.
- Final repository-wide tests: 132 passed, 3 failed (the same pre-existing
  poster artifact failures listed above). All 20 new tests passed.
- Final saved verification reproduces metrics, trial identities, parameters,
  and diagnostics for all 396 results; all 24,303 prior output files and the
  10 hashed source files remain unchanged. No commit or push was performed.
