# Corrected stage-pair geometry: Monkey T, channel 7

Plot-only comparisons from the saved 198 short-delay trials of session y070316009-12.
Each stage uses its corrected 300-ms window, m=3, tau=3 ms, and 294 observed points.
Post-SC begins after the spatial instruction ends (event 206), not at SC onset.
The existing 1-torus fits and 15-feature table are reused without refitting or decoding.
Band half-width is `(R1 - R1_inner + R2 - R2_inner) / 4` in normalized signal units.

## Files

| File | Content |
| --- | --- |
| `sc_example_clouds_trials_6_9_10.png` | Pre-SC versus post-SC observed clouds and saved annular fit outlines, for original trials 6, 9, 10. |
| `go_example_clouds_trials_6_9_10.png` | Pre-GO versus post-GO for the same trials. |
| `sc_radii_width.png` / `go_radii_width.png` | R1, R2, and band half-width: matched-trial distributions above; within-trial post-minus-pre values below. |
| `sc_orientation.png` / `go_orientation.png` | Empirical cumulative distributions of all nine saved orientation components. |
| `sc_fit_quality.png` / `go_fit_quality.png` | Empirical cumulative distributions of MSE, mean in-plane error, and fraction inside. |
| `feature_summary.csv` | Matched-trial medians and quartiles of each post-minus-pre change. |
| `paired_trial_features.csv` | All 198 trial identities and their pre/post features, usability, and differences for each comparison. |
| `example_index.json` | Example trial and saved-window identities, with 3D axis limits. |
| `source_hashes.json` / `figure_manifest.json` | Source and figure integrity records. |

SC has 193 matched trials with both fits usable; GO has 194. Unusable fits remain
in `paired_trial_features.csv` but are absent from plotted geometric distributions.
The example trial numbers were fixed by the earlier six-stage figure; they were not
selected for visually large differences. Outer fit boundary is solid charcoal;
inner fit boundary is dashed. Identical coordinate limits and viewpoints are used
within each two-stage example figure. Plots are descriptive, from one channel and
one recording; no significance testing or decoder rerun was performed. Every
300-ms window was normalized separately before embedding, so a radius change
cannot be interpreted as a change in raw LFP amplitude.
