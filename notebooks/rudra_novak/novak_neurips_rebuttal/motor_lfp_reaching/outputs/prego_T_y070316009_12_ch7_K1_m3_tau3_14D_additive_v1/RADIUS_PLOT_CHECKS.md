# R1-R2 Plot Checks

- All 197 plotted pairs match the saved R1 and R2 columns, without projection,
  scaling, imputation, or refitting. Direction counts are 33, 33, 32, 33, 33, 33.
- Saved row 69 (original trial 128) is omitted because its fit did not converge.
  This does not change the 198-trial decoder or any previous plot.
- Both focused tests pass: exact valid-pair selection and exact scatter
  coordinates with six direction colors, marginal densities, and no gridlines.
- The figure was visually inspected. All PDF text fits on the page and is at
  least 8 pt. PNG is 600 dpi; SVG contains editable text. CSV and caption are
  included beside the figure, with a separate source/output hash manifest.
- Full test suite: 164 passed, 10 subtests passed. The same three unrelated
  pre-existing poster failures remain: `test_powerpoint_is_single_editable_poster`,
  `test_required_sources_and_metric_caveats`, and
  `test_rendered_text_stays_inside_its_editable_box`.
