# Six-Stage Betti Heatmaps

- [betti_h1_stage_deviation_raw.png](betti_h1_stage_deviation_raw.png)
- [betti_h1_stage_deviation_raw_zoom.png](betti_h1_stage_deviation_raw_zoom.png)
- [betti_h1_stage_deviation_rms_normalized.png](betti_h1_stage_deviation_rms_normalized.png)
- [betti_h1_stage_deviation_rms_normalized_zoom.png](betti_h1_stage_deviation_rms_normalized_zoom.png)

Rows are task stages, not reach directions. All 198 trials contribute to every row.
Color is stage mean beta1 minus the pooled mean across all 1,188 windows at the
same distance: red = more loops; blue = fewer. The reference is not a per-stage
mean across distances. Values are count differences, not z-scores or percentages.

Raw and RMS-normalized panels share a +/-3-count scale. The zoom panels
show 0-1 on the corresponding distance axis, matching the earlier rectangular
display; no distance was selected to maximize stage contrast. Full-range panels
retain all 512 saved distances. There is no interpolation or smoothing.

These descriptive mean contrasts do not show variability or establish stage
separation. The original trial distributions overlap. Post-TC starts after tone
offset; post-SC starts after the spatial instruction ends.

Saved results only: no Ripser, geometry fitting, preprocessing, or decoding rerun.
All previous results remain unchanged. Supporting CSVs reproduce every cell;
captions, source, and validation are in their respective subfolders.
