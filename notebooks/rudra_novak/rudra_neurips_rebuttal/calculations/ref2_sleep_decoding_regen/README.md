# ref2_sleep_decoding_regen

## Method

- Regenerate the sleep-decoding components shown in `../guide/ref2.png`.
- Use `../guide/behavior_final.ipynb` as provenance for the original torus-feature extraction, density plots, confusion matrix, LDA projection, and F1 comparison.
- Extract runnable code into this unit before recomputing; do not edit the guide notebook.
- Improve the decoding analysis with an explicit spectral baseline, held-out validation split, and error bars if the source data supports them.
- Write regenerated component plots and any table summaries into `plots/`.

## Variables

- Data/input: mouse sleep EEG/LFP data and labels read from the existing NeuralFieldManifold project.
- Sessions/groups: sleep states Wake, NREM, and REM; session or subject groups if available for held-out splitting.
- Labels/targets: sleep-stage label per window.
- Signals/features/measures: torus features `R1`, `R2`, and `r`; spectral baseline features as explicitly defined in the regenerated analysis.
- Parameters/thresholds: windowing and feature extraction settings inherited from `behavior_final.ipynb` unless changed and documented here.
- Outputs: regenerated density panels, confusion matrix, LDA projection, F1 comparison, and summary tables in `plots/`.

## Statistics

- Tests/models: classifier for sleep-state decoding, baseline classifier using named spectral features, cross-validation or held-out evaluation.
- Null hypothesis: torus features do not improve held-out sleep-state decoding over the named spectral baseline.
- Alternative hypothesis: torus features improve held-out sleep-state decoding over the named spectral baseline.
- Thresholds/decision rule: report mean and spread across folds or held-out groups; do not claim improvement from a single split without error bars.
- What the statistic means: F1 summarizes balanced decoding performance across sleep states.
- Why this statistic is appropriate here: reviewers directly questioned the original `74% vs 60%` result because the baseline and validation protocol were underspecified.

## Legends

- X axis: feature value, predicted label, LDA coordinate, or sleep state depending on the panel.
- Y axis: density, true label, LDA coordinate, or F1 score depending on the panel.
- Color/value: consistent Wake, NREM, and REM colors across panels.
- Grouping: sleep-stage labels and feature set conditions.
- Ordering/sorting: use Wake, NREM, REM order unless the original panel requires otherwise.
- Lines/markers/labels: KDE curves, heatmap cell values, contour lines, and bar heights.
- Panels: feature distributions, confusion matrix, LDA projection, and F1 comparison.

## Interpretation

- This unit should clarify whether torus geometry features carry sleep-state information under a defensible validation protocol.
- It should also make explicit what the spectral baseline contains.

## Notes

- Status: blocked for faithful rerun because `state_sec3.npy` is missing.
- Copied available EEG inputs into `../../data/`: `eeg_signal_sec3.npy`, `eeg_time_sec3.npy`, and `eeg_processed.npz`.
- See `REGENERATION_STATUS.md` for exact missing-file details and recoverable notebook-output metrics.
- Keep all new outputs inside this unit.
- Keep `../guide/ref2.png` as the visual reference, not an editable target.

## References

- `../guide/ref2.png`
- `../guide/behavior_final.ipynb`
