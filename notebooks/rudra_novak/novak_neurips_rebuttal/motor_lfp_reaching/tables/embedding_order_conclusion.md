# Macaque LFP Lag-Embedding Order Selection

Movement-aligned PSDs were aggregated across 341 unique LFPs.
For each LFP, trial PSDs were median-aggregated, log-transformed, and corrected for a linear 1/f background in log-frequency space.
Peaks within 48-52 Hz were excluded as line noise before counting biological oscillatory modes.

Robust biological PSD peaks were found at: 19.0 Hz, 32.0 Hz.
This gives K=2 oscillatory modes, corresponding to AR order 2K=4 and recommended lag-embedding dimension 2K+1=5.

Conclusion: use one dataset-level lag-embedding dimension of 5 for macaque LFP analyses if the embedding order is chosen from PSD peak count.
