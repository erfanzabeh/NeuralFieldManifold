# T channel 7: fixed-delay geometric fitting

No decoding was run.

- Recording: `monkeyT_session-y070316009-12_lfp-7`.
- Exactly 198 saved short-delay trials, 33 per direction; original trial IDs retained.
- Audited 500-ms epochs [4000:4500] at 1000 Hz; all samples strictly before GO.
- Existing per-trial detrend, median/MAD normalization, zero-phase 2-55 Hz filter.
- m=3, tau=16 ms for EVERY trial; 468 points per cloud, no coordinate reduction.
- Original package geometric fitters: nonlinear least squares, Huber loss, max_nfev=6000.
- Native SVD is used for initialization and extents only; all original coordinates enter the objectives.
- Saved PSD peak near 22 Hz is context only. Both candidate shapes are imposed and tested, not inferred from fitting scores.

## Read first

[Compact fitting report](Fitting_Report.pdf)

[All 198 trial pairs](All_198_Trials_Atlas.pdf)

## Results

Model | Returned | Converged | Flagged | Collapsed hole | Median normalized 3D error | IQR
--- | ---: | ---: | ---: | ---: | ---: | ---
1-torus: planar band | 198/198 | 198/198 | 91/198 | 0/198 | 0.244798 | 0.198023-0.281141
2-torus: donut volume | 198/198 | 197/198 | 183/198 | 99/198 | 0.001245 | 0.000995-0.001602

All finite returned fits, including flagged/nonconverged fits, contribute to distributions.
Exceptions remain explicit rows with missing numerical metrics. No favorable-result filtering.

## Metric definitions and interpretation

- 1-torus: a planar annular band, not a thin ellipse. Native coverage and mean_error use in-plane distances only. Native R-squared uses in-plane squared error divided by full-cloud variance. Native MSE includes normal-plane offset.
- 2-torus: containment in a solid elliptical tube. Native errors are zero inside the volume; native R-squared uses outside-only squared distance divided by full-cloud variance. They do not measure distance to the donut surface.
- Shared normalized_3d_set_error = sum(native 3D distances squared) / sum((points - cloud mean) squared). For the band, include out-of-plane offset; for the donut, use outside-only tube distance.
- The native four-step nearest-ellipse calculation is an approximation, preserved unchanged. Shared normalization does not make the geometric model classes equally flexible.
- A lower donut containment error is not proof of K=2. High coverage, high native R-squared, and optimizer success are not evidence of genuine torus topology.
- Tube overlays depict the native parametric normal sweep, not the boundary of the solid fitted volume. Thick-tube sweeps can self-intersect and include internal surfaces; collapsed-hole flags are displayed and retained.
- Do not compare these numbers to the old 0.80 time-trajectory SSE/TSS. The residual definitions and fitted models differ.
- One channel, one session/day: 198 trial fits are not 198 independent recordings or animals.
- Flags: optimizer nonconvergence, active/near parameter bounds (relative tolerance 1e-5), nonfinite/degenerate geometry, collapsed holes, invalid band radii, and nearly circular orientation (axis ratio <1.05). Flags do not cause trial removal.

## Saved artifacts

- `clouds.npz`: processed epochs, complete original-coordinate clouds, labels, original trial IDs, fixed embedding metadata.
- `checkpoints/`: 396 trial/model JSON files with native fit parameters, optimizer diagnostics, per-point residuals, and exact input-cloud hash.
- `tables/trial_fits.csv`: all results/failures; `fit_summary.csv`: medians/IQR and counts; `paired_metrics.csv`: matched trial values; `flag_counts.csv`: diagnostics.
- `tables/example_trials.csv`: label-blind lowest-ID example selection; `metric_verification.csv`: independent distance/metric reproduction.
- `figures/`: independent editable SVG, vector PDF, 600-dpi PNG and captions. Numerical source for summary figures is in `tables/`; example clouds and meshes reproduce from `clouds.npz` and the corresponding checkpoints.
- `source/`, `provenance.json`, `config.json`, `prior_outputs_manifest.json`: code/input/config tracking and prior-output protection.

## Reproduce

Run from the repository root using the neuralmanifold environment:

```bash
python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_prego_geometric_fits.py --resume --jobs 4
python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/report_prego_geometric_fits.py
```

The report entrypoint only reads frozen results; it cannot trigger fitting. Existing output directories require matching configuration and source/input hashes to resume.
Verified unchanged: 24303 previous output files.

**Stop point: fitting inspection only. No feature matrix, LDA, F1, or behavioral model selection.**
