# ref2_sleep_decoding_regen status

Ref2 cannot be faithfully regenerated from files currently present in the repository.

## Blocking issue

- The source notebook loads `state_sec3.npy` from an absolute local path:
  `/Users/novak/Documents/UTSW/erfan/deeplagfield/state_sec3.npy`.
- The repository contains `eeg_signal_sec3.npy`, `eeg_time_sec3.npy`, and `eeg_processed.npz`, but no sleep-state label file was found.
- Without the 1800 window labels, the Wake/NREM/REM feature densities, confusion matrix, LDA projection, and F1 bars cannot be recomputed.

## What is recoverable from notebook outputs

- The copied notebook records that labels had shape `(1800,)` and unique values `[0, 1, 2]`.
- The old balanced split used 109 windows per class, 327 windows total.
- Old printed results include:
  - Band power accuracy: `0.599`
  - Lag embedding statistics accuracy: `0.716`
  - Torus geometry accuracy: `0.725`
  - Radii-only accuracy: `0.731`
  - Orientation-only accuracy: `0.379`

## Next step

Copy `state_sec3.npy` into `../../data/` when available, then regenerate this unit from real data.
