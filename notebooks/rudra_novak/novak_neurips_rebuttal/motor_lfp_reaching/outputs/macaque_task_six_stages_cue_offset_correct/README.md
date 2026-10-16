# Corrected Stage Decoding: Start Here

These are the corrected cue-offset results for Monkey T, session y070316009-12,
channel 7: 198 short-delay trials, six 300-ms windows per trial.

## Main Figures

- [01: Task timeline](01_task_timeline.png)
- [02: Confusion matrix](02_stage_confusion.png). Diagonal values are class recall, not F1.
- [03: Per-stage F1 heatmap](03_stage_f1_heatmap.png)
- [04: Per-stage F1 with intervals](04_stage_f1_intervals.png)
- [05: All 15 geometric feature distributions](05_all_15_feature_distributions.png)
- [06: Circular radius and width distributions](06_circular_radius_width_distributions.png)
- [07: Trial 6 geometry and fitted outlines](07_trial_6_geometry_with_fits.png)
- [08: All 198 trials with fits](08_all_198_trials_with_fits.pdf)
- [09: All 198 trials, observed trajectories](09_all_198_trials_observed.pdf)

Individual feature and geometry PNGs are in `individual_panels/`.
The labeled PDF atlases retain the same trial/page ordering; `tables/trial_page_index.csv`
provides the index. Editable timeline artwork is in `cartoon_editable/`.

## What Is Corrected

| Stage | Window |
| --- | --- |
| Pre-TC | 300 ms immediately before TC onset |
| Post-TC | 300 ms immediately after TC ends |
| Pre-SC | 300 ms immediately before SC onset |
| Post-SC | 300 ms immediately after the spatial instruction ends |
| Pre-GO | 300 ms immediately before GO |
| Post-GO | 300 ms immediately after GO |

TC end uses recorded code 203. SC instruction end uses code 206, when distractors
appear; this is not the later all-stimuli-off event. Analysis uses each trial's
recorded times, not the nominal cue durations drawn in the cartoon.

## Results and Verification

Macro-F1: **0.29284**. Accuracy: **30.387%**. These are exploratory single-recording
results; the corrected timing did not improve overall decoding performance.

- [Short report](REPORT.md)
- [Fresh collection checks](verification/collection_check.json): all 1,188 window
  boundaries checked against saved event timestamps; scores and confusion matrix
  reproduced from held-out predictions; copied files checked byte-for-byte.
- [Original numerical validation](verification/validation.json)
- [Figure validation](verification/figure_validation.json)
- [File manifest](verification/file_manifest.json): exact source and SHA256 for every copy.
- [Exact trial window boundaries](verification/window_metadata.csv)

This is a convenient collection of existing corrected outputs, not a new analysis.
No refitting, decoding, or plot generation is performed when collecting it.
The underlying experiment and its caches remain unchanged at:

`notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2/`

This collection is a snapshot. Later analysis changes will not silently update these copies.
