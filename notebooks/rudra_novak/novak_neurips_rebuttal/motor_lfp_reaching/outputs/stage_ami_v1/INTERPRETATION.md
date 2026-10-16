# Interpretation

The selected pooled delay is 25 ms. The five training-only choices are 25, 25, 24, 25, and 25 ms, and the 32/48/64-bin choices are 26/25/24 ms. These checks show stability under the tested estimator and fold variations, not a confidence interval or universally optimal delay.

The first minimum is shallow: the curve falls rapidly at short delays and then remains near zero over a broad region. The selected integer should be treated as an approximate decorrelation scale, not a uniquely optimal 25-ms embedding. Pooled MI is measured on the combined distribution of within-window lagged pairs; it is not the mean of 1,188 separately estimated trial-window MI curves. Stage-specific curves illustrate that the signals have different delay-dependence, but do not authorize separate stage-specific embeddings or establish stage separation.

The saved decoding sweep stopped at 20 ms, so it contains no F1 result for the pooled 25-ms suggestion. The existing 3-ms clouds, fits, features, and decoding remain unchanged. A 3D embedding at 25 ms would retain 250 points per 300-sample window, compared with 294 at 3 ms. No such cloud or fit was produced here.

`ami_pooled_detail.png` magnifies 10-40 ms and `ami_by_stage_detail.png` magnifies 10-70 ms. Their explicit, linear axis limits preserve the small absolute AMI values. Full-range plots remain alongside them. All curves and selections can be reproduced from the saved CSVs; the detail plotting script performs no estimation, fitting, or decoding.

Methodological reference: [Fraser and Swinney, Physical Review A (1986)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.33.1134).
