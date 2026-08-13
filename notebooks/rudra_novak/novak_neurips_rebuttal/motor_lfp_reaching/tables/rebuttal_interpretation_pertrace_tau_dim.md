# Motor-Cortex LFP Reaching Interpretation

This analysis tests the reviewer-suggested macaque reaching setting directly, using the published motor-cortex LFP dataset rather than adding an unrelated grid-cell discussion.
The torus-feature row uses per-LFP tau and embedding dimension from notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/tables/lfp_embedding_params_pertrace_tau_dim.csv.

For movement-aligned reach-direction decoding, torus geometry features reached F1=0.288 +/- 0.171 across 341 unique LFP recordings.
The multi-band spectral baseline reached F1=0.322 +/- 0.178, with single-band baselines reported separately in the accompanying CSV and bar plots.

Interpretation for the rebuttal should emphasize that this is a motor-cortex reaching analysis with the dataset's documented six reach directions and short/long delays. The result is intended as an empirical stress test of whether delay-geometry features carry task information beyond conventional spectral power, not as a claim about grid-cell attractor topology.

Large converted arrays and feature caches are stored locally under git-ignored folders; the committed artifacts are scripts, notebooks, compact tables, and final plots.
