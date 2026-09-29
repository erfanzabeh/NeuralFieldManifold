# Observed stage geometry at 25-ms delay

Trials 6, 9, and 10 are the previously established examples. Each `tau25_trial_*.png` shows the six corrected windows; columns pair before/after TC, SC, and GO. Each `tau3_vs_tau25_trial_*.png` compares the same six windows at 3 ms (top) and 25 ms (bottom), in chronological stage order.

The frozen 300-ms processed signals were embedded directly as [x(t), x(t-tau), x(t-2*tau)]: 250 observed points at 25 ms and 294 at 3 ms. Lines connect actual samples in temporal order without smoothing, resampling, PCA, or AR prediction. No fitted ellipse or torus is shown; no geometric fitting or decoding was run. Post-TC and post-SC begin at the recorded cue offsets.

Both delays and all stages share limits and camera within each trial, including across its two panels. Limits vary between trials and are saved in `view_scales.csv`. Units are normalized signal units. Inputs and previous results were hash-verified unchanged. PNGs are 600 dpi.

Saved coordinates: `clouds.npz`; window identities: `windows.csv`. Regenerate only plots with `plot_stage_tau25_geometry_examples.py --plot-only`.
