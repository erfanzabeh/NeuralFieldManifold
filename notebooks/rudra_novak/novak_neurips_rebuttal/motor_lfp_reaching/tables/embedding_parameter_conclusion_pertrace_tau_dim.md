# Per-Trace Macaque LFP Embedding Parameters

Selected tau and embedding dimension independently for 341 unique LFP traces using movement-aligned LFP signal structure only.
Tau was chosen from average mutual information with autocorrelation fallbacks; embedding dimension was chosen from robust per-trace PSD peaks using dimension 2K+1.
Median tau was 17.0 ms (IQR 14.0-23.0 ms).
Embedding-dimension counts: 3D=34, 5D=103, 7D=123, 9D=81.

These parameters should be used for torus features in the per-trace rerun, while spectral baselines remain unchanged.
