# Left/right radius plot checks

- Dataset README: direction codes 101-106 run clockwise from the upper-right target.
- Dataset Setup&Task.pdf: targets are upper right, right, lower right, lower left, left, upper left. There are no vertical-midline targets.
- Mapping: Rightward = 1, 2, 3; Leftward = 4, 5, 6. This is target position in the task display, not arm identity or hemisphere.
- Sources inspected: https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/README.md and https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/Setup%26Task.pdf
- All 198 saved original trial IDs and direction labels match the full converted recording; all have short-delay code 91 and matching condition codes.
- Each group has 99 selected trials. The plotted counts are 99 leftward and 98 rightward. Original trial 128 (direction 3) is the previously nonconverged fit; no imputed dot.
- All 197 R1/R2 pairs are unchanged in the plot. Exported CSV values match the source within 1e-14 absolute tolerance; one CSV floating-point round-trip differs by 8.9e-16.
- No jitter, coordinate transformation, fitting, feature extraction, or decoding. Descriptive KDE contours only.
- Same axis limits and density grid as the previous six-direction plot.
- Export inspection: no visible clipping; PDF text within page bounds; minimum text 8 pt; editable SVG contains 24 text elements; PNG 3600 x 3150 at 600 dpi.
- Output hashes verified. Rendering checked 409 prior files for unchanged hashes, including frozen decoding outputs and the existing six-direction plot.
- Focused tests: 9 passed. Full repository tests: 171 passed, 10 subtests passed, 3 existing unrelated poster failures; 2 existing matplotlib deprecation warnings.

## Existing unrelated test failures

All in `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/sfn_poster/test_poster.py`:

- `test_powerpoint_is_single_editable_poster`: existing poster has 36 text shapes; test expects more than 50.
- `test_required_sources_and_metric_caveats`: existing poster text lacks expected `0.082`.
- `test_rendered_text_stays_inside_its_editable_box`: existing `authors_002` box does not match the rendered PDF.

Poster files were left unchanged.
