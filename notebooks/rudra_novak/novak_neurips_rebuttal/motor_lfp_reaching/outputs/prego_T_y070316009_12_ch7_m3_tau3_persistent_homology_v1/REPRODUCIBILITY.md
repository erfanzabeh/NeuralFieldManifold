# Reproduction and boundaries

This is a separate descriptive experiment, not a decoding update. The observed
cloud for each trial contains `[x(t), x(t-3), x(t-6)]`, `t=6,...,499`, copied by
reading the frozen tau-3 cloud cache. No filtering or embedding is repeated.
Signal amplitude units remain those of the existing normalized epochs.

## Entry points

From the repository root, using the existing `neuralmanifold` environment:

```bash
python -B notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_prego_persistent_homology.py --phase pilot
python -B notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_prego_persistent_homology.py --phase all
python -B notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_prego_persistent_homology.py --verify-only
python -B notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/plot_prego_persistent_homology.py
```

Successful and failed trial checkpoints are reused, not silently retried.
Every trial has a 600-second worker timeout. A failed pilot stops the full run.
Do not delete checkpoints to improve unfavorable results.

## Internal records

- `diagrams/`: complete H0/H1/H2 birth/death arrays, including infinite intervals.
- `checkpoints/`: trial status, runtime, input hash and diagram hash.
- `tables/trial_measurements.csv`: one row per eligible trial, explicit failures,
  longest finite H1 lifetime, its RMS-normalized value, and saved radii/validity.
- `tables/persistence_intervals.csv`: every interval; `inf` is preserved.
- `tables/betti_curves.csv`: actual per-trial Betti counts on shared grids.
- `tables/betti_summary.csv`: means and interquartile ranges by direction.
- `tables/distribution_summary.csv`: trial distribution summaries, not tests.
- `tables/plot_*.csv`: exact plotting inputs, including display-only jitter.
- `captions/`: scientific definitions and qualifications outside the PNG artwork.
- `protected_manifest.json`: before/after protection of all previous experiment
  files and the existing analysis/fitter source files.

Only PNGs in `figures/` are final artwork. `previews/` is visual QA, not an extra
scientific result. PNGs are exported at 600 dpi; designed widths are 3.3-3.6 inches,
with at least 8-point text at that size. No PDF or SVG export is produced.

Betti counts use `birth <= distance < death`. They are not counts exceeding a
chosen lifetime threshold. The centered RMS normalization is applied only to
the lifetime summary, not to the clouds before persistence. Direction labels
are used for grouping and color only. No parameter or example is selected for
class separation, and no test of between-direction significance is performed.

Validation is restricted to the new experiment. Existing decoder/fitter test
suites are not executed, since the approved task explicitly excludes those
model calls. Synthetic persistence checks are separate from the 198-trial run.
