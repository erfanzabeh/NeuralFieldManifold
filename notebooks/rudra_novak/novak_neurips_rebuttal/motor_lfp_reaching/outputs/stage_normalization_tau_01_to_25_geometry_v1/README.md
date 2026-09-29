# MAD versus envelope geometry: tau 1-25 ms

Three PDFs, one each for the same short-delay trials 6, 9 and 10. Each PDF contains 25 bookmarked vector pages. Page number equals tau in milliseconds. MAD geometry is in the left two columns, Hilbert-envelope geometry in the right two columns. Rows pair pre/post-TC, pre/post-SC and pre/post-GO.

Uses the exact saved 300-sample signals from stage_tau3_envelope_geometry_v1. m=3; all native points are retained (298 at tau=1, 250 at tau=25). Each point is an observed delayed sample; lines connect samples without smoothing. No normalization, filtering, geometric fitting, decoding or persistent homology was rerun. The tau=3 clouds reproduce the previous comparison exactly.

The camera and coordinate limits are fixed across all stages and all 25 pages within each trial and method. MAD and envelope have different units and separate numeric limits, printed on each page; absolute sizes cannot be compared across methods. All earlier input/results were hash-checked unchanged.

Saved clouds, window identities, page index, scales and hashes support reproduction. Plot-only regeneration: plot_stage_normalization_tau_atlas.py --plot-only.
