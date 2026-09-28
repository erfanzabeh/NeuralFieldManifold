# Frozen Inputs

`windows.npz` contains the original raw windows, new float64 envelope-normalized windows, full raw trials, trial envelopes, and window envelopes. Nonfinite normalized windows are explicit failures, not substituted signals.

`window_metadata.csv` is the original corrected-stage metadata copied from the source experiment. Its `cloud_sha256` column refers to the **original MAD-normalized tau=3 clouds**, not this experiment's new clouds.

After validation, `envelope_window_metadata.csv` preserves the same identities, anchors, and folds, explicitly renames that original hash, and adds hashes for the new envelope-normalized signals and tau=3 clouds. Delay-specific cloud hashes are saved in every fit checkpoint. `clouds_tau_XX.npz` stores each complete native cloud array after collection.

`event_samples.csv` preserves recorded event timestamps. Post-TC starts at tone offset; post-SC starts at the end of the spatial instruction. Every window has 300 samples, and embedding uses only those samples.
