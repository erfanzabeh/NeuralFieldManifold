# Six-Stage Persistent Homology

Completed **1188/1,188** clouds from **198 short-delay trials** in Monkey T,
session y070316009-12, channel 7. Successful windows by stage:
Pre-TC: 198; Post-TC: 198; Pre-SC: 198; Post-SC: 198; Pre-GO: 198; Post-GO: 198.
All calculations used the saved 294 observed points, m=3, tau=3 ms, and corrected
300-ms windows. Post-TC starts at tone offset; post-SC starts at spatial-instruction
end (distractor onset). No signals, geometric fits, or decoding results changed.

Ripser 0.6.15 computed full H0/H1/H2 persistence with Euclidean distances and
coefficients in F2, without subsampling or an assumed topology.
**1188** successful clouds contain H1 intervals; **1148** contain H2 intervals.
Stage median longest H1 lifetimes range from **0.715 to 0.904**
in preprocessed-coordinate distance units. RMS-normalized medians are: Pre-TC 0.452; Post-TC 0.450; Pre-SC 0.455; Post-SC 0.475; Pre-GO 0.448; Post-GO 0.444.
Stage distributions overlap substantially; size-normalized Betti curves are visually
similar. Betti counts are not the oscillatory-mode count K.

Curves show stage means and interquartile bands, not confidence intervals.
Distribution dots are individual trial windows; markers show median/IQR.
Normalized summaries divide diagram distances by each cloud's centered RMS radius;
they do not require transformed clouds or another Ripser run. Trial 6 supplies all
six example diagrams and barcodes, without selection for separation.

Failures: **0**. Undefined RMS normalizations:
**0**. Clouds with unusable geometry fits
remain included (11 successful PH calculations).
The six stages are repeated measurements from the same trials, not independent recordings.

These are descriptive, scale-dependent measurements from one selected recording.
Short filtered windows, finite sampling, and noise can affect persistence. H1/H2
intervals alone do not establish clean torus topology or stage-specific separation.
No threshold optimization, significance testing, or topology decoding was performed.
