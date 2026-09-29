# Six-stage cue-offset correction

The active experiment is `outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2/`.
The older `300ms_15D_v1` run used cue-onset-aligned post windows and does not answer the intended question.

| Class | Half-open window | Recorded anchor |
|---|---|---|
| Pre-TC | onset - 300 ms to onset | TCon, code 202 |
| Post-TC | offset to offset + 300 ms | TCoff, code 203 |
| Pre-SC | onset - 300 ms to onset | SCon, code 204 |
| Post-SC | instruction end to end + 300 ms | DistrOn, code 206 |
| Pre-GO | GO - 300 ms to GO | code 207 |
| Post-GO | GO to GO + 300 ms | code 207 |

The dataset [README](https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/src/master/README.md)
defines these codes and a 1-kHz time base. Its [task description](https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/raw/master/Setup%26Task.pdf)
describes the 200-ms tone and 55-ms spatial instruction. Distractor onset ends the spatial instruction;
it is not an all-screen-stimuli-off event. Recorded, per-trial event times determine every slice;
nominal durations are never added to onsets to infer the ends. Actual TC durations are 194-205 ms,
and spatial-instruction durations are 53-54 ms in this cohort.

## What Changes

Only 396 post-cue windows are reprocessed and refitted. The 792 other raw windows, processed
signals, point clouds, and fit checkpoints are checked for exact agreement before reuse.
All decoding, 1,000 within-trial label permutations, 2,000 trial-cluster bootstrap samples,
complete-fit sensitivity, and downstream figures are recomputed. Even unchanged stages can
have different predictions because the decoder is trained against changed competing classes.
No cohorts, folds, geometry parameters, feature definitions, or classifier settings are retuned.

## Entry Points

- `run_stage_geometry_decoding.py --phase all --resume --jobs 6`: resume the corrected analysis.
- `regenerate_stage_cue_offset_figures.py`: regenerate plots only after numerical validation.
- `stage_cue_offset_correction.remove_superseded(OUTPUT)`: remove only the archived old stage and
  six-stage cartoon folders, after both numerical and figure validation succeed.

## Preservation

Commit `57d5b2a90a7430234e1275ca6ea178a3d943a4fa` was checked against the remote branch before correction.
It preserves the tracked old files, but not ignored numerical caches. A separate, fully verified
archive also preserves those caches at
`/home/nochen/code/NeuralFieldManifold_archives/stage_onset_57d5b2a/superseded_stage_and_cartoon.tar.gz`.
Its member hashes and checksum are in the corrected run's `superseded_archive.json`.
The backup is local, not an off-machine backup. Raw data and earlier direction-decoding/topology
experiments are not removed. The corrected run's `cleanup.json` records any completed removals.

For full historical restoration, extract that archive into `outputs/` and use source files from
the historical commit. Do not run current cue-offset scripts against restored onset-aligned results.
