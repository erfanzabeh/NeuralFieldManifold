# Final Quality Checks

## Numerical verification

- All 198 saved trials and five folds match their original identities exactly.
  Each trial contributes 494 unprojected, original-coordinate lag points.
- All 198 fitted parameter/diagnostic checkpoints and all 22 arrays in clouds,
  features, held-out predictions, permutations, and bootstrap archives exactly
  reproduce the first complete execution. Only fit wall-clock durations differ.
  The repeated execution followed a resume-environment safeguard correction;
  no analytical parameter, representation, trial, or random seed changed.
- Native fit distances and quality measures were independently recomputed from
  saved clouds and fitted parameters for all 198 trials. All five methods'
  macro-F1, accuracy, and class F1 values were independently reproduced using
  scikit-learn metrics.
- The one nonconverged fit is row 69, original trial 128, direction 3. It reached
  the unchanged 6,000-evaluation limit. Its 14 geometric features are missing;
  training-fold imputation retains its held-out prediction and the full cohort.
- 12 fits have active-bound flags, 13 have near-bound flags; these overlapping
  flags did not exclude trials or trigger retuning.
- The saved validation report verifies training-only imputation/scaling,
  absence of PCA in every fitted pipeline, confusion normalization, all 1,000
  null scores, all 2,000 bootstrap scores, source hashes, and preservation of
  all 25,312 files in earlier experiment outputs.

## Independent review

- A read-only reviewer found a resume guard that did not freeze library
  versions. Python, NumPy, SciPy, scikit-learn, joblib versions and machine
  architecture are now frozen and checked before checkpoint reuse.
- The added regression test first failed and then passed after the correction.
- A second read-only review confirmed the correction and repeatability. Its
  minor caption finding was corrected: observed-score confidence intervals
  use bootstrapping, while shuffled-label reference intervals use permutations.
- All 19 new experiment tests pass. Final repository test run: 157 passed,
  10 subtests passed, and the same three pre-existing poster-artifact failures:
  `test_powerpoint_is_single_editable_poster`,
  `test_required_sources_and_metric_caveats`, and
  `test_rendered_text_stays_inside_its_editable_box`.
  Poster files were not modified.

## Figure inspection

- Six independently movable panels are exported as PDF, SVG, and PNG, with a
  matching CSV and caption for each. All PNGs are 600 dpi (metadata rounding:
  599.9988 dpi). All SVGs retain editable text.
- All PDF text bounds lie within the page; the minimum figure text size is
  8 pt. Both confusion matrices share the full 0-1 scale and direction order.
- Every panel was visually inspected at reading scale for clipping and
  overlapping labels. The zero-F1 marker remains fully visible at the baseline.
- Feature-comparison and radius plot values reproduce their source tables.
  Geometry distributions show finite observed measurements, never imputed ones.

## Interpretation limits

Geometry macro-F1 is 0.227921, with a conditional 95% interval of
0.176038-0.280514, and a one-sided permutation p of 0.016983. All-band power
macro-F1 is 0.225966. This does not establish superiority to the all-band
representation. Geometry has zero correct predictions for direction 4;
its degenerate conditional bootstrap interval is not zero uncertainty about
future performance. This remains an exploratory result from one selected
recording and a visually selected delay, not a confirmatory or population result.
