# Stage-window AMI diagnostic

All 198 short-delay trials contributed six corrected 300-ms windows (1,188 total). The pooled primary AMI curve suggests **tau = 25 ms**, using the first interior local minimum after sigma=1-ms smoothing. This selection uses no stage labels or decoding scores.

Saved outer-training-fold choices (folds 0-4): **[25, 25, 24, 25, 25] ms**. Histogram-bin sensitivity: 32 bins: 26 ms; 48 bins: 25 ms; 64 bins: 24 ms. Stage-specific descriptive choices: pre_TC: 21 ms; post_TC: 17 ms; pre_SC: 18 ms; post_SC: 30 ms; pre_GO: 16 ms; post_GO: 46 ms.

AMI was computed between scalar signal values separated by each lag from 1-100 ms, pooling within-window pairs without crossing boundaries. The existing histogram estimator uses uniform bins between signal percentiles 1 and 99; excluded-pair fractions are saved. All available in-range pairs were used, with no random subsampling. Stage-specific delays are diagnostics, not six separate applied embedding parameters.

The result is a delay heuristic, not proof of correct topology, fit quality, or optimal decoding. Short, filtered windows and histogram/smoothing choices limit interpretation. No geometry, feature, decoder, or persistence result was recalculated. Existing native and matched F1 scores are copied only for context where the suggested delay was already swept; the AMI diagnostic does not independently validate those scores. The currently adopted tau remains 3 ms.

Computation/validation wall time: 10.06 seconds. Protected source-result hashes were unchanged.
