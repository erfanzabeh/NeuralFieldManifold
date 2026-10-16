# Betti Analysis: Corrected Six Stages

This folder contains topology-only results, not a new decoder.
All source geometry fits and decoding outputs remain unchanged.

## Main Panels
- [Loop-count curves](01_betti_h1.png)
- [Size-normalized loop-count curves](02_betti_h1_rms_normalized.png)
- [Longest loop lifetime](03_longest_h1_lifetime.png)
- [Size-normalized longest lifetime](04_longest_h1_lifetime_rms_normalized.png)

## All Standalone PNGs
- [01_betti_h1.png](01_betti_h1.png)
- [02_betti_h1_rms_normalized.png](02_betti_h1_rms_normalized.png)
- [03_longest_h1_lifetime.png](03_longest_h1_lifetime.png)
- [04_longest_h1_lifetime_rms_normalized.png](04_longest_h1_lifetime_rms_normalized.png)
- [05_betti_h0.png](05_betti_h0.png)
- [06_betti_h2.png](06_betti_h2.png)
- [07_persistence_trial6_post_GO.png](07_persistence_trial6_post_GO.png)
- [07_persistence_trial6_post_SC.png](07_persistence_trial6_post_SC.png)
- [07_persistence_trial6_post_TC.png](07_persistence_trial6_post_TC.png)
- [07_persistence_trial6_pre_GO.png](07_persistence_trial6_pre_GO.png)
- [07_persistence_trial6_pre_SC.png](07_persistence_trial6_pre_SC.png)
- [07_persistence_trial6_pre_TC.png](07_persistence_trial6_pre_TC.png)
- [08_h1_barcode_trial6_post_GO.png](08_h1_barcode_trial6_post_GO.png)
- [08_h1_barcode_trial6_post_SC.png](08_h1_barcode_trial6_post_SC.png)
- [08_h1_barcode_trial6_post_TC.png](08_h1_barcode_trial6_post_TC.png)
- [08_h1_barcode_trial6_pre_GO.png](08_h1_barcode_trial6_pre_GO.png)
- [08_h1_barcode_trial6_pre_SC.png](08_h1_barcode_trial6_pre_SC.png)
- [08_h1_barcode_trial6_pre_TC.png](08_h1_barcode_trial6_pre_TC.png)

## Documentation
- [Short report](REPORT.md)
- [Calculation validation](validation/numerical_checks.json)
- [Plot validation](validation/plot_checks.json)
- [Per-window measurements](tables/window_measurements.csv)
- [Stage lifetime summaries](tables/lifetime_summary.csv)
- [Stage Betti-curve summaries](tables/betti_summary.csv)

`diagrams/` stores full H0/H1/H2 intervals, including infinity. `tables/` contains
measurements, compressed CSVs of intervals and per-window curves, shared grids,
and the exact values used in each PNG. `captions/` provides methods outside the artwork.
`inputs/` preserves trial/stage metadata; the original frozen clouds remain in the
source experiment identified in `config.json`. All raw-distance axes use the existing
preprocessed-coordinate units, not raw voltage. RMS-normalized axes are dimensionless.

Examples always use original trial 6; H1 barcode rows are ranked by lifetime within
each stage, not matched loop identities across stages. All its H1 intervals are shown.
H0's essential infinite interval is displayed separately above the finite diagram scale.

Regeneration uses the saved tables and intervals only:
`python plot_stage_persistent_homology.py` from the motor-LFP analysis directory.
The independent runner supports `--phase pilot`, `--phase all --resume`, and
`--phase verify --resume`. Neither entrypoint calls a geometric fitter or decoder.
