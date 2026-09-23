# Six-stage geometric decoding

Monkey T, session y070316009-12, channel 7: 198 short-delay trials, 33 per reach direction, yielded 1,188 windows (198 per stage). Each 300-ms window was independently detrended, median/MAD-normalized, and zero-phase filtered at 2-55 Hz. Direct embedding used m=3 and tau=3 ms, producing 294 observed points. No PCA reduction, AR signal replacement, PINN, or persistent homology was run.

The fixed annular-band fitter provided 15 features, including band half-width. 1177/1,188 fits were usable; unusable counts: pre_TC: 1, post_TC: 3, pre_SC: 2, post_SC: 0, pre_GO: 2, post_GO: 3. Boundary flags were retained; 0 fits had nearly circular, potentially unstable in-plane orientations. Missing feature rows were training-median imputed.

Five-fold LDA kept all six windows from each trial together. Held-out macro-F1 was **0.293** (95% conditional interval 0.268-0.318); accuracy was **0.310** (0.284-0.335). Intervals used 2,000 whole-trial bootstraps stratified by reach direction. The 1,000 within-trial stage permutations gave macro-F1 p=0.0010; null median 0.162. Complete-fit sensitivity retained 188 trials: macro-F1 0.304, accuracy 0.320.

Examples are original trials 6, 9, and 10, chosen by ID, not separation. Radii and half-width are normalized, not absolute voltage measures. Post-cue windows include delay activity; post-GO includes reaction and potentially movement. Short-window filtering, fit quality, temporal drift, and this previously inspected single recording limit interpretation. This is offline within-recording stage prediction, not cross-animal validation, causal decoding, or proof of distinct topologies. The selected annular model is a representation, not a verified topology for every stage.
