# ref1_topology_assessment_regen

## Method

- Regenerate the real-data topology-assessment components associated with `../../guide/ref1.png`.
- Use `../guide/monkey_torus_fit copy.ipynb` as provenance for the original barcode, Betti sweep, and Betti-space scatter code.
- Extract only the needed runnable logic into this unit before recomputing; do not edit the guide notebook.
- Recreate the barcode panel from a cached full-trace ripser diagram.
- Recreate the Betti-space scatter and Betti-category distribution from the cached Betti sweep.
- Compare the real Betti-signature distribution to the five cached shuffle ripser Betti signatures.
- Do not crop or paste pieces of `../../guide/ref1.png` into generated figures.
- Keep the compact Table 2 Betti summary in `table2_compact.md`.
- Write regenerated component plots into `plots/`.

## Variables

- Data/input: monkey V1 LFP data and existing NeuralFieldManifold helpers read from outside the rebuttal folder.
- Sessions/groups: real monkey V1 windows and five matched 200-window shuffle conditions.
- Labels/targets: Betti categories `(1,2,1)`, `(1,1,1)`, and pooled other.
- Signals/features/measures: lag embedding, persistent homology bars, Betti numbers, and Betti-category percentages.
- Parameters/thresholds: inherit the original topology notebook choices until deliberately changed and documented here.
- Outputs: regenerated topology component plots in `plots/`; optional caches in `cache/`.

## Statistics

- Tests/models: Betti-category classification from the cached real-data sweep.
- Null hypothesis: not tested in this unit.
- Alternative hypothesis: not tested in this unit.
- Thresholds/decision rule: use the same persistence threshold and category pooling rule later when comparing real and shuffled windows.
- What the statistic means: Betti category percentages measure how often each window shows the expected torus signature, partial signature, or other topology.
- Why this statistic is appropriate here: the reviewer concern targets the `46% / 40% / 14%` Betti split, so the improved figure should evaluate that split directly.

## Legends

- X axis: radius or embedding coordinate, depending on the component plot.
- Y axis: homology dimension, Betti coordinate, or embedding coordinate, depending on the component plot.
- Color/value: dark red highlights real data; non-red colors mark individual shuffle conditions in the grouped comparison.
- Grouping: windows grouped by observed Betti signature.
- Ordering/sorting: preserve original panel order when regenerating components.
- Lines/markers/labels: barcode intervals, Betti scatter points, and category labels.
- Panels: barcode, Betti-space scatter, expanded Betti-signature bar chart, and grouped real-vs-shuffle Betti bar chart.

## Interpretation

- This unit exposes the real Betti split without hiding the component counts inside the pooled `Other` category.
- The goal is not to recompose the final Illustrator figure; the goal is to regenerate clean source components from data.

## Notes

- Status: real Betti summary regenerated from the existing cached `betti_sweep.npy`.
- Generated outputs: `plots/ref1_1x3_regenerated_components.png`, `plots/ref1_real_vs_shuffle_betti_grouped_bar.png`, `plots/ref1_real_betti_compact_summary.csv`, `plots/ref1_real_betti_expanded_counts.csv`, `plots/ref1_real_vs_shuffle_betti_grouped_bars.csv`, and `table2_compact.md`.
- The third panel shows every observed Betti signature (`121`, `111`, `131`, `110`, `141`, `120`) sorted by frequency.
- The grouped bar plot includes shuffle-only signatures such as `101`, `151`, and `161`, and sorts signatures by total observed frequency across real plus shuffle windows.
- The grouped bar uses within-condition percentages; real and each shuffle are all 200-window Betti distributions computed with the same ripser recipe.
- The compact regenerated split is `46.5%` full `T2 (1,2,1)`, `39.5%` partial `(1,1,1)`, and `14.0%` other.
- The full-trace barcode diagram is cached in `cache/full_trace_ripser_dgms.npz`; recomputing that cache requires `ripser`.
- Keep all new outputs inside this unit.
- Keep `../../guide/ref1.png` as the visual reference, not an editable target.

## References

- `../../guide/ref1.png`
- `../../guide/monkey_torus_fit copy.ipynb`
- `../nulls/cache/betti_sweeps/`
