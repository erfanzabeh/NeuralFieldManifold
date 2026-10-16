# Validation of the Two Requested Figures

- The six representations have dimensions 1, 15, 1, 15, 5, and 14, in the
  requested order. The beta scalar is the saved all-band vector's 13-30 Hz
  component. Combined inputs concatenate, rather than average, their features.
- All 198 trials and saved fold assignments are preserved. The 30 fitted
  method/fold pipelines use training-only median imputation, standardization,
  and automatic-shrinkage LDA. No geometry fit, lag choice, or width feature
  was added or changed.
- All six pooled macro-F1 scores and accuracies were independently reproduced
  using scikit-learn. Every one of the 12,000 bootstrap macro-F1 values
  (2,000 resamples times six methods) was checked against saved predictions.
- Geometry-only and all-band predictions, resample indices, and bootstrap
  scores match the preceding experiment exactly. All 326 files in that
  experiment passed unchanged-hash checks.
- The visualization uses one common LDA transformation learned only from
  the 158 training trials of saved fold 0. Its 40 held-out points are
  transformed without refitting. Training and held-out roles are saved
  alongside every plotted trial identity.
- The plotted LDA coordinates reproduce the saved model and training-only
  centering/scaling. Regression tests confirm that altering held-out data
  or labels cannot change training axes, and held-out positions cannot
  change the KDE grids or densities.
- The two panels were visually inspected. All PDF labels fit within the
  page and are at least 8 pt. Both SVGs contain editable text. PNG sizes:
  4260 x 2250 for the comparison; 3600 x 3150 for the LDA projection, at
  600 dpi (metadata rounding to 599.9988 dpi).
- Automated plotting tests confirm six bars, no gridlines, and exactly
  one dashed horizontal reference at 1/6. There are no shuffled-label markers
  or significance stars. The reference is nominal, not an exact empirical
  macro-F1 null or a significance threshold.
- All five new tests passed. Full suite: 162 passed and 10 subtests passed;
  the same three unrelated pre-existing poster failures remain:
  `test_powerpoint_is_single_editable_poster`,
  `test_required_sources_and_metric_caveats`, and
  `test_rendered_text_stays_inside_its_editable_box`.

These comparisons remain exploratory single-recording results. Training
contours are supervised descriptive summaries, not evidence of held-out
separation. No paired significance tests or favorable-result retuning were
performed for this update.
