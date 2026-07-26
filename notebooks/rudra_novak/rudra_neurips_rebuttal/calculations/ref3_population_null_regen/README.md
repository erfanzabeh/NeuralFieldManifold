# ref3_population_null_regen

## Method

- Regenerate the population-summary and existing null-distribution components shown in `../guide/ref3.png`.
- Use `../guide/population_summaries.ipynb` as provenance for the original AR(2) null, torus score, torus error, R2, tau sweep, and window-size sweep.
- Extract runnable code into this unit before recomputing; do not edit the guide notebook.
- Preserve the original synthetic AR(2) null as a separate torus-fit-metric control.
- Distinguish this AR(2) torus-fit null from the Betti shuffle table now kept in `../ref1_topology_assessment_regen/table2_compact.md`.
- Write regenerated component plots and summary tables into `plots/`.

## Variables

- Data/input: monkey LFP source data and existing NeuralFieldManifold helpers read from outside the rebuttal folder.
- Sessions/groups: real monkey LFP windows and synthetic AR(2) null windows.
- Labels/targets: no behavioral labels.
- Signals/features/measures: torus score, torus error, R2, tau-dependent summaries, and window-size-dependent summaries.
- Parameters/thresholds: AR(2) null with real poles from the original notebook, matched signal length, same rescaling, matched windows, lag embedding, and torus-fitting settings.
- Outputs: one regenerated `3x3` panel in `plots/`.

## Statistics

- Tests/models: synthetic AR(2) null generation, lag embedding, two-torus fitting, and distribution comparisons between real and null metrics.
- Null hypothesis: real LFP torus-fit metrics are no different from metrics produced by a same-length synthetic AR(2) autocorrelated signal.
- Alternative hypothesis: real LFP torus-fit metrics show stronger torus fit than the AR(2) null.
- Thresholds/decision rule: use the original notebook's distribution comparison unless changed and documented here.
- What the statistic means: torus score, torus error, and R2 quantify geometric fit quality, not Betti-category frequency.
- Why this statistic is appropriate here: this null tests whether the torus-fit metric pipeline produces strong scores on generic autocorrelated signals.

## Legends

- X axis: torus score, torus error, R2, tau, or window size depending on the panel.
- Y axis: probability, metric value, or error value depending on the panel.
- Color/value: real data and AR(2) null colors should be clear and consistent.
- Grouping: real monkey LFP versus synthetic AR(2) null.
- Ordering/sorting: preserve original panel order where possible.
- Lines/markers/labels: histogram bars, mean markers, mean-plus-SEM sweep lines.
- Panels: metric histograms, tau sweeps, and window-size sweeps.

## Interpretation

- This unit preserves the original AR(2) null story: real LFP is compared against a simple autocorrelated synthetic signal on torus-fit metrics.
- This does not replace the Ref1 compact table, which answers a different question about Betti-category distributions under data-matched shuffles.

## Notes

- Status: AR(2) torus-metric smoke regeneration completed on `80` matched windows.
- Generated output: `plots/ref3_panel_d_3x3.png`.
- Cached fits: `cache/ref3_ar2_metric_summary_n80.pkl`.
- The `3x3` panel follows the original notebook structure: rows are torus score, torus error, and `R2`; columns are metric histograms, tau sweeps, and window-size sweeps.
- This is a controlled smoke run, not the original full `1000`-window run.
- Smoke-run means: real torus score/R2 are higher than AR(2), and real torus error is lower than AR(2), matching the intended direction of the original panel.
- Keep all new outputs inside this unit.
- Keep `../guide/ref3.png` as the visual reference, not an editable target.

## References

- `../guide/ref3.png`
- `../guide/population_summaries.ipynb`
- `../ref1_topology_assessment_regen/table2_compact.md`
