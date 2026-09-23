# Panel captions

Geometry: Original trials 6, 9, and 10 were selected by increasing original trial ID, independently of fit quality or classification. Every cloud contains 294 observed delay vectors from its 300-ms window, with m=3 and tau=3 ms. All examples use identical original-coordinate limits and viewpoints. Fitted panels overlay the outer solid and inner dashed annular boundaries only for usable fits; these outlines do not prove topology. No projection or dimensionality reduction is applied.

Feature distributions: Each dot is one usable trial-window fit. White-centered markers show medians; thick vertical segments show interquartile ranges. Stage colors are fixed, not assigned by performance. Radii and band half-width are in normalized signal units. Band half-width is (R1_out-R1_in+R2_out-R2_in)/4, not R1-R2. MSE has squared normalized units; mean error has normalized units; fraction inside is unitless. Unusable fits are absent only from descriptive distributions; decoding retains their windows using training-only imputation. Counts are in fit_accounting.csv.

Orientation: Components refer to the lag-coordinate system, not anatomical or reaching directions. Normals and ellipse axes use the same canonical ordering/sign convention as the prior decoder. Signed components may be discontinuous at canonical sign boundaries, and nearly circular fits have uncertain in-plane axes; no claim of stage separation should be based on these conventions alone.

Confusion: Row-normalized held-out prediction fractions from five folds, with all six stages of each original trial kept together. Diagonal values are stage recall, not F1. Each row contains 198 windows before normalization. The color scale is fixed to 0-1.

Per-stage F1: Pooled held-out per-class F1, with 95% conditional intervals from 2,000 whole-trial bootstrap resamples stratified by original reach direction. Every resampled trial contributes all six windows. Folds and windows are not independent experimental replicates. Overall macro-F1, accuracy, and the within-trial stage-label permutation test are in report.md.

Event windows: Pre is [-300,0) ms and post is [0,300) ms relative to actual recorded TC onset, SC onset, or GO. Post-TC/SC include subsequent delay activity; post-GO is a reaction/movement window rather than a pure reaching epoch. Filtering is zero-phase and local to each short window, so this is offline classification with potential edge effects, not causal online forecasting.
