# Torus Feature Refit Interpretation

This rerun replaces the earlier PCA torus proxy with torus features from an elliptical torus least-squares fit for movement-aligned LFP epochs.
The delay cloud uses per-LFP tau and embedding dimension from notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/tables/lfp_embedding_params_pertrace_tau_dim.csv; dimensions above 3 are projected to their leading three principal coordinates before applying the same 3D elliptical torus fit.

For all unique LFPs, torus features reached F1=0.288 +/- 0.169 across 341 LFPs.
The average-PSD baseline reached F1=0.270 +/- 0.178.
The all-band spectral baseline reached F1=0.321 +/- 0.177.
For full six-direction LFPs, torus features reached F1=0.181 +/- 0.032 across 237 LFPs.
The average-PSD baseline reached F1=0.159 +/- 0.030.
The all-band spectral baseline reached F1=0.212 +/- 0.049.
