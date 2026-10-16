# Corrected Six-Stage Decoding

Post-TC starts at recorded tone offset. Post-SC starts when the spatial instruction ends,
marked by distractor onset. All six windows last 300 ms. This replaces the old onset-aligned run.

## Figures

- [Corrected task cartoon](../macaque_task_six_stages_cue_offset_v2/macaque_task_six_stages.png)
- [Held-out confusion matrix](png/stage_confusion.png): diagonal entries are recall.
- [Per-stage F1 heatmap](f1_heatmap_v1/stage_f1_heatmap.png)
- [Per-stage F1 with intervals](png/stage_f1.png)
- [All 15 feature distributions](feature_distributions_v1/png/all_geometric_feature_distributions.png)
- [Circular radius/width distributions](radial_feature_distributions_v1/png/radial_stage_distributions.png)
- [Trial 6, trajectories and fitted outlines](trajectory_style_v1/png/geometry_trial_6_fitted.png)
- [All 198 trials with fitted outlines](all_trials_pdf_v1/all_198_trials_with_fits.pdf)
- [All 198 trials, observed trajectories only](all_trials_pdf_v1/all_198_trials_observed.pdf)

Individual panels are also saved in the corresponding figure folders. PDF pages are labeled
by original trial ID and retain matching page numbers between the two atlases.

## Results and Checks

[Short report](report.md), [numerical validation](validation.json),
[figure validation](figure_validation.json), [stage scores](tables/per_stage_f1.csv),
[fit accounting](tables/fit_accounting.csv), and [exact window shifts](inputs/correction_audit.csv).

198 trials, 1,188 windows, 15 features, the same five trial-grouped folds, m=3, and tau=3 ms.
Only 396 post-cue fits were recomputed; 792 unchanged fits were reused after exact input checks.
All classifier fits, permutations, bootstrap intervals, and complete-fit sensitivity were regenerated.
Results remain exploratory within one recording; no settings were retuned after inspection.

## Historical Recovery

The old tracked outputs remain in pushed commit `57d5b2a`. A complete local archive additionally
preserves Git-ignored caches. See [archive manifest](superseded_archive.json) and
[cleanup record](cleanup.json). The raw data, earlier direction-decoding results, and persistent
homology analysis were not removed or changed. The archive is local, not an off-machine backup.
