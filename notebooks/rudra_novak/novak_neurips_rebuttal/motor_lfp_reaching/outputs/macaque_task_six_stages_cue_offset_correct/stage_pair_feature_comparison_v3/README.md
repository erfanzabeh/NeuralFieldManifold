# Side-by-side geometric feature comparisons

Monkey T, session y070316009-12, channel 7. The corrected six-stage analysis
used 198 short-delay trials, 300-ms windows, m=3 and tau=3 ms. Post-SC starts
after the spatial instruction ends (event 206). Only saved feature values and
clouds were read; no fitter or decoder was run.

Each figure has SC in the left column and GO in the right column. Within each
column, pre and post are paired by original trial number. The five figure
groups contain all 15 features: `radii_width`, `plane_normal`, `major_axis`,
`minor_axis`, and `fit_quality`. For each group, `box_*_SC_GO` shows matched
trials as points with median/IQR boxes and 5th-95th percentile whiskers;
`density_*_SC_GO` shows probability densities, not cumulative curves.
Each figure is exported as a 600-dpi PNG and a PDF.

There are 193 trials with both SC fits usable and 194 with both GO fits
usable. The same y-axis limits are used across the two columns for each boxplot
feature. Density plots use the same x/y ranges and a common Gaussian-kernel
bandwidth across all four stages for each feature; plotted coordinates are in
`density_curves.csv`.

The boxplot annotations are two-sided paired Wilcoxon signed-rank p-values,
Holm-corrected across all 30 prespecified tests (15 features x 2 comparisons).
Exact raw/adjusted p-values, matched counts, and median within-trial changes are
in `paired_wilcoxon_holm.csv`. Nonsignificant comparisons are reported too.
These are exploratory, within-session tests; they do not establish an effect
across animals or separate cue effects from the passage of trial time.
Each window was normalized independently, so radii are not raw LFP amplitudes.
The prior v1/v2 plots remain unchanged.
