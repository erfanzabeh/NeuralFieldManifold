# Stage Betti and Geometry Mean/SEM Bars

Monkey T, session y070316009-12, channel 7. All 198 short-delay trials were
included in topology; 1,177 of 1,188 geometric fits were usable. Post-TC starts
at tone offset and post-SC at spatial-instruction end. The existing stage colors
are reused in every panel. All exports are standalone 600-dpi PNGs.

- [betti_0_1_2_mean_sem_raw.png](png/betti_0_1_2_mean_sem_raw.png)
- [betti_0_1_2_mean_sem_rms_normalized.png](png/betti_0_1_2_mean_sem_rms_normalized.png)
- [geometry_mean_sem_R1.png](png/geometry_mean_sem_R1.png)
- [geometry_mean_sem_R2.png](png/geometry_mean_sem_R2.png)
- [geometry_mean_sem_band_half_width.png](png/geometry_mean_sem_band_half_width.png)
- [geometry_mean_sem_normal_x.png](png/geometry_mean_sem_normal_x.png)
- [geometry_mean_sem_normal_y.png](png/geometry_mean_sem_normal_y.png)
- [geometry_mean_sem_normal_z.png](png/geometry_mean_sem_normal_z.png)
- [geometry_mean_sem_u_x.png](png/geometry_mean_sem_u_x.png)
- [geometry_mean_sem_u_y.png](png/geometry_mean_sem_u_y.png)
- [geometry_mean_sem_u_z.png](png/geometry_mean_sem_u_z.png)
- [geometry_mean_sem_v_x.png](png/geometry_mean_sem_v_x.png)
- [geometry_mean_sem_v_y.png](png/geometry_mean_sem_v_y.png)
- [geometry_mean_sem_v_z.png](png/geometry_mean_sem_v_z.png)
- [geometry_mean_sem_mse.png](png/geometry_mean_sem_mse.png)
- [geometry_mean_sem_mean_error.png](png/geometry_mean_sem_mean_error.png)
- [geometry_mean_sem_frac_inside.png](png/geometry_mean_sem_frac_inside.png)

## Betti definition

A Betti number changes with distance. Each trial's plotted value is the exact
average Betti count from zero to a **common endpoint** (the maximum finite H0
death over all saved clouds): raw distance 1.31857, or RMS-normalized
distance 0.553385. The three dimensions share the same
endpoint within a plot. Saved birth/death intervals are integrated exactly,
including each H0 infinite interval. The y axis uses a zero-preserving symmetric
log scale so beta0, beta1, and beta2 are visible together.

SEM is SD/sqrt(n) across trials within a stage, not across animals. The stages
reuse the same trials; these error bars do not test paired stage differences.
The geometric panels use only valid saved 15-dimensional feature vectors and
exclude the 11 unusable fits without imputation. Radii and width use per-window
normalized signal units; orientation vectors are in lag coordinates. The 15
features differ in scale, so each panel has its own y axis.

No persistence calculation, geometric fitting, preprocessing, or decoding was
run. See `tables/` for every per-trial value and the exact means, SDs, and SEMs;
see `captions/`, `source/`, and `validation/` for details.
