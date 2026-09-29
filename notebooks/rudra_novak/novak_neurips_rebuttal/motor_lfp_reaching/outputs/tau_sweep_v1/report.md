# Six-stage tau sweep: Monkey T, channel 7

The same 198 short-delay trials and six corrected 300-ms windows were analyzed at each delay from 1 to 20 ms. All six windows from a trial remained in the same held-out fold. The native sweep used 300 - 2*tau points per cloud; the matched sweep used the same final 260 time positions at every delay. Both used the same 1-torus fitter, 15 features, and shrinkage LDA.

| Sweep | Observed highest macro-F1 | Tau | Usable fits | Tau=3 macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Native points | 0.344453 | 9 ms | 1,187 / 1,188 | 0.292841 |
| Matched 260 points | 0.339168 | 15 ms | 1,187 / 1,188 | 0.310112 |

Native tau=3 reproduced the previous 15-feature table exactly and its earlier macro-F1 to full saved precision. The observed winner changes when point count is held fixed, so these results do not support a unique robust delay. Post-GO has the strongest winner-stage F1 (0.611 native; 0.616 matched), while several other stages remain weak or confused. Confusion-matrix diagonals are class recall, not F1.

These are descriptive held-out results for one recording. Choosing the largest score after comparing 20 delays introduces selection optimism; neither winner is an independently validated optimum. No new preprocessing, topology calculation, PCA, or spectral comparator was run.

The four requested 600-dpi PNGs are in `figures/`. Full per-delay fit diagnostics, features, models, and predictions are retained under `checkpoints/`, `tables/`, `models/`, and `predictions/`. Plot-only regeneration uses `plot_stage_tau_sweep.py` from the experiment's parent directory.
