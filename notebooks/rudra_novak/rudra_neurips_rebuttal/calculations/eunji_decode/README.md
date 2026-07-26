# eunji_decode

## Goal

Investigate whether torus-derived LFP dynamics help decode Eunji EKEZ behavioral state labels: Mobile, Immobile, and Sleep.

The rigorous dashboard outputs are the main analysis. The old Novak-style rerun is kept only as a reference artifact.

## Inputs

- `../../data/ekez/LFP0_*.dat`: six raw `int16` LFP recordings.
- `../../data/ekez/dat_file_check.ipynb`: tutorial/provenance for reading the `.dat` files.
- `../../data/ekez/timestamp.xlsx`: mouse ID, group, date, phase, good channels, and labeled windows.

## Parameters from the tutorial/reference

- `.dat` layout: 32 interleaved `int16` channels.
- Raw sampling rate: `20000 Hz`.
- Tutorial probe order: `[18, 19, 12, 13]`.
- Good channels are used as direct NumPy columns, matching the tutorial notebook.
- Rigorous decode window length: `2 s`, with a 2 s label-edge buffer.
- Native PSD features are computed from the original 20 kHz LFP.
- Torus features use anti-aliased 2 kHz traces over 0.5-200 Hz with a 2 ms delay.
- Feature sets: native spectrum, high-rate dynamics, torus shape, spectrum+torus, and all features.
- Primary classifier: LDA with held-out session/channel/day/mouse splits.

## Label handling

The spreadsheet phases are `mobile`, `immobile`, and `sleep`, not Wake/NREM/REM. One date has overlapping intervals, so decode windows are made exclusive before feature extraction: `sleep` and `immobile` intervals are kept first, then removed from the broader `mobile` interval.

## Commands

```bash
/home/dev/miniconda3/envs/manifold/bin/python inventory_ekez_data.py
/home/dev/miniconda3/envs/manifold/bin/python regen_eunji_decode.py --force --n-jobs -1
/home/dev/miniconda3/envs/manifold/bin/python rigorous_sleep_state_investigation.py --force --n-jobs -1 --n-null 100 --max-per-cell 24 --torus-fs 2000
/home/dev/miniconda3/envs/manifold/bin/python torus_advantage_analyses.py --force --n-jobs -1 --n-null-reps 5 --n-perm 100 --torus-fs 2000
```

## Outputs

- `cache/ekez_file_inventory.csv`: raw file sizes and durations.
- `cache/ekez_label_windows.csv`: parsed spreadsheet windows.
- `cache/ekez_overlap_report.csv`: overlapping label intervals.
- `cache/eunji_all_windows.csv`: every extracted 2 s LFP window.
- `cache/eunji_balanced_windows.csv`: balanced windows used for decoding.
- `cache/eunji_decode_features.npz`: cached signal windows, features, predictions, confusion matrices, and LDA projection.
- `cache/eunji_decode_scores.csv`: accuracy and per-state F1 for each feature set.
- `cache/eunji_decode_summary.json`: compact run summary and parameters.
- `plots/01_data_qc_label_coverage.png`: label intervals, sampled windows, amplitude QC, and torus fit failures.
- `plots/02_native_psd_state_atlas.png`: native 20 kHz PSD by state and channel.
- `plots/03_feature_state_atlas.png`: torus, spectral, and high-rate dynamics feature summaries.
- `plots/04_decode_generalization_matrix.png`: macro-F1 by feature set and held-out scale.
- `plots/05_null_distributions.png`: leave-session real scores against label-permutation and circular-shift nulls.
- `plots/06_session_robustness.png`: held-out session scores for spectrum and torus.
- `plots/07_sampling_rate_sensitivity.png`: torus decoding at 400, 1000, and 2000 Hz analysis rates.
- `plots/08_incremental_torus_gain.png`: paired macro-F1 gain from adding torus features to native spectrum.
- `plots/09_state_r2_not_f1.png`: held-out state variance explained using one-hot R2 instead of F1.
- `plots/10_residual_torus_after_spectrum.png`: torus residual structure after predicting torus variables from spectrum.
- `plots/11_psd_matched_torus_nulls.png`: paired real torus metrics vs PSD-matched phase-randomized nulls.
- `plots/12_rms_matched_state_structure.png`: state R2 after matching raw amplitude distributions.
- `plots/13_conditional_r2_importance.png`: R2 drop when spectrum or torus variables are permuted inside the combined model.
- `plots/14_domain_stability_map.png`: state separation relative to session/channel/mouse shifts.
- `plots/15_temporal_torus_trajectories.png`: residual torus trajectories against theta power and raw SD over time.
- `plots/eunji_decode_ref2_style.png`: old ref2-style reference figure, not the main analysis.
- `plots/eunji_representative_lag_embeddings.png`: old representative lag-embedding reference figure.

## Current run

- Structural inventory: six `.dat` files, all divisible by 32 `int16` channels; every spreadsheet window falls inside its matching recording.
- Parsed labels: 18 rows; 19 exclusive intervals after splitting one overlapped mobile interval.
- Extracted windows: 5554 total 2 s windows; balanced decode set is 256 each for Mobile, Immobile, and Sleep.
- Overlaps removed from broad mobile windows: 120 s overlapping sleep and 27 s overlapping immobile on `ekez004/250828`.
- Accuracy: band power `0.642`, lag-embedding stats `0.552`, torus 6-feature `0.560`, all torus 15-feature `0.562`.
- Per-state F1 for the plotted comparison: band power `[0.734, 0.559, 0.635]`; all torus `[0.772, 0.345, 0.557]` for `[Mobile, Immobile, Sleep]`.

## Rigorous run

- Analysis table: 800 windows after a 2 s label-edge buffer, capped at 24 windows per mouse/date/channel/state.
- Native spectral analysis stays at the original 20 kHz sampling rate. The torus fit uses anti-aliased 2 kHz traces over 0.5-200 Hz with a 2 ms delay; this is the highest practical point-cloud rate used here, and the sensitivity plot checks 400/1000/2000 Hz.
- Primary generalization test: leave one recording session out. Macro-F1 is native spectrum `0.643 +/- 0.047`, torus alone `0.597 +/- 0.058`, and spectrum+torus `0.677 +/- 0.044`.
- Incremental torus gain over native spectrum on leave-session folds: `+0.034` macro-F1 on average. The gain is positive on most held-out folds but negative on one session.
- Hardest generalization test: leave one mouse out. Macro-F1 is native spectrum `0.525 +/- 0.068`, torus alone `0.546 +/- 0.030`, and spectrum+torus `0.552 +/- 0.068`; with only two mice this is informative but not a stable population claim.
- Nulls: 100 label permutations and 100 session/channel circular shifts per feature set for the leave-session test. Real scores are well to the right of both nulls for native spectrum, torus alone, and spectrum+torus.
- Caveat: torus fitting failed for 13.6% of the 2 kHz windows, mostly Mobile windows. Failed torus rows are median-imputed for decoding and the failure fraction is plotted, so this result should be treated as promising but not final.

## Additional torus-advantage run

- State variance by held-out one-hot R2: leave-session native spectrum `0.372`, torus shape `0.230`, spectrum+torus `0.372`, and all non-F1 features `0.384`.
- Conditional information: in the combined spectrum+torus model, permuting spectrum drops held-out state R2 by `0.546 +/- 0.006`; permuting torus drops it by `0.204 +/- 0.004`.
- PSD-matched nulls: real torus metrics differ from phase-randomized PSD-preserving nulls, strongest for `frac_inside` with paired effect size `0.54`; not every torus variable changes in the favorable direction.
- RMS-matched subset: after matching raw amplitude distributions, native spectrum state R2 drops to `-0.094`, while torus shape remains positive at `0.094` on 117 matched windows.
- Residual torus after spectrum: some residual torus variables retain state structure, but the residual effect sizes are small; the strongest is `frac_inside` with residual state eta2 `0.034`.
- Domain stability: torus shape has stronger state/mouse separation ratio than spectrum (`3.32` vs `1.95`), but spectrum is stronger for session and channel ratios.
