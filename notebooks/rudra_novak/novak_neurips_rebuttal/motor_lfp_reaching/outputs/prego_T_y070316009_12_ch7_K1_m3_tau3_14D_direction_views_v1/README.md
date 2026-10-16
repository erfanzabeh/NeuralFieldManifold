# Plot-only direction views

Monkey T, session y070316009-12, LFP channel 7; 198 short-delay trials, 33 per direction; final 500 ms before GO. K=1, m=3, tau=3 ms. Exploratory single-recording results. 

These are visualizations of frozen tables, not a new experiment. No decoder evaluations, torus fits, feature extraction, bootstrap, permutation tests, or tuning models were run.

## Included

- Target-space held-out predictions: geometry and all-band powers, standalone plus comparison.
- R1 and R2 direction profiles: raw trial dots and arithmetic mean +/- sample SD.
- All nine orientation components: mean values on a fixed -1 to 1 scale.

## Not run

- Feature-family ablation: requires new decoder evaluations.
- Time-resolved decoding: requires new windows, features, fits, and decoder evaluations.

## Accounting

Prediction maps retain all 198 trials (33 per direction). Feature plots use 197 usable fits; original trial 128 in direction 3 remains omitted from these descriptive feature views only. Unusable values were not imputed for plotting. Boundary-flagged but usable fits remain included.

Direction order: 1 upper right, 2 right, 3 lower right, 4 lower left, 5 left, 6 upper left. Target positions are schematic. Source definitions: https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/README.md and https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/Setup%26Task.pdf.

PDF and SVG preserve vector artwork and text; PNGs are 600 dpi. Small PNGs in previews/ are for screen viewing. Every panel has its own caption and underlying display table; raw plotted features are also saved in tables/plotted_geometry_trials.csv.

Verified 420 existing source/plot files unchanged.
