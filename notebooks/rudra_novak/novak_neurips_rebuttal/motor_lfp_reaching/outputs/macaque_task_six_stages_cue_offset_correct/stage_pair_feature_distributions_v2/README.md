# Corrected pre/post stage feature comparisons

Monkey T, session y070316009-12, channel 7. These are plot-only summaries of
the saved six-stage tau=3 ms, m=3, K=1 geometric fits. Post-SC starts after
the spatial instruction ends (event 206), not at spatial-cue onset.

Each comparison includes only trials with usable fits in both stages:
193 of 198 for pre/post-SC and 194 of 198 for pre/post-GO.
All 15 fitted features are included. No fitting, decoding, significance testing,
or feature redefinition was performed.

## Figure files

For each event prefix, `sc_` or `go_`:

| File suffix | Content |
| --- | --- |
| `box_radii_width` | R1, R2, and band half-width boxplots with all matched-trial points. |
| `box_orientation` | Nine plane-normal, major-axis, and minor-axis components. |
| `box_fit_quality` | MSE, mean in-plane error, and fraction inside. |
| `density_radii_width` | Probability-density curves for R1, R2, and width. |
| `density_orientation` | Probability-density curves for the nine orientation components. |
| `density_fit_quality` | Probability-density curves for the three fit-quality measures. |

Every figure is provided as a 600-dpi PNG and a PDF. The word `density`
means a probability-density estimate, not a cumulative distribution. For each
feature, pre/post curves use the same Gaussian-kernel bandwidth calculated from
their pooled matched-trial values. Curves are normalized over the displayed
range; `density_curves.csv` records all plotted coordinates and bandwidths.
Box centers are medians, boxes are interquartile ranges, and whiskers mark the
5th and 95th percentiles. `paired_feature_values.csv` retains every trial and
its within-trial difference; `feature_summary.csv` gives matched medians.

Each 300-ms window was independently normalized before fitting. Radii and width
are therefore in normalized signal units, not raw LFP amplitude. These are
descriptive comparisons from one recording. The earlier cloud examples remain
in `../stage_pair_geometry_v1/` and were not changed here.
