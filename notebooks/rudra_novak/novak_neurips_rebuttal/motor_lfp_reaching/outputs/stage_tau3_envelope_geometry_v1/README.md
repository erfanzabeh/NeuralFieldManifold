# Stage geometry: alternative normalization

Same Monkey T recording y070316009-12, channel 7, short-delay trials 6, 9 and 10. Each `envelope_geometry_trial_*.png` shows the six corrected 300-ms windows. Each `mad_vs_envelope_trial_*.png` pairs the exact saved MAD geometry (top) with the alternative geometry (bottom). Post-TC/SC remain cue-offset aligned.

Window detrending, median centering, fourth-order 2-55 Hz zero-phase filtering, m=3 and tau=3 ms are unchanged. Instead of dividing each window by its own MAD, we divide by a Hilbert amplitude envelope estimated from the entire individual 6301-sample trial, smoothed with a second-order 3-Hz zero-phase filter, and multiply by that trial's median envelope. No trials or stage windows are concatenated. The envelope uses full-trial context; this is an offline visualization, not a causal or stage-local decoder preprocessing proposal. This is the older mouse envelope normalization family, with the macaque band and original window filters retained.

The 294 observed delayed samples are connected in time without smoothing, PCA, AR prediction or geometric fitting. No decoder or persistent homology was run. PNG exports are 600 dpi. All six stages share a camera and scale within each trial and method. MAD and envelope rows use different numeric limits because their units differ; absolute sizes cannot be compared between rows.

Saved signals, envelopes and clouds are in `inputs_and_clouds.npz`; configuration, metadata, diagnostics and protected-file hashes support reproduction. Existing geometry, decoding and topology files were hash-checked unchanged. Plot-only regeneration: `plot_stage_normalization_geometry.py --plot-only`.
