# nulls

## Method

- Load the copied topology-ready monkey LFP trace from `../../data/monkey_lfp.npz`.
- Generate five shuffled traces that each destroy or preserve a different part of the real signal structure.
- Save each shuffle as a compressed `.npz` file in `cache/`.
- Cache a windowed ripser Betti sweep for each shuffle in `cache/betti_sweeps/`.
- Save a diagnostic plot for each shuffle in `plots/`.
- Diagnostics are one PNG per shuffle in a `2x3` grid: the shared trace/PSD/histogram comparison stays in row 1, and row 2 shows shuffle-specific checks.
- These traces and cached Betti sweeps are inputs for later Betti and torus-score reruns.
- `notebook_lfp_psd_check.py` separately recreates the manuscript-style raw LFP PSD panel from the raw `.mat` channel after the notebook's notch filter only. That PSD display trace is not the shuffle input.

## Variables

- Data/input for shuffles: `../../data/monkey_lfp.npz`, arrays `xs`, `Fs`, and `tau`. The time vector is reconstructed from `Fs`.
- Data/input for the PSD check: original raw `NSP8_array16_LFP.mat`, global channel `1001` / local channel `41`, first `10000` samples.
- Sessions/groups: one real monkey V1/LFP trace and five shuffle conditions.
- Labels/targets: no behavioral labels.
- Signals/features/measures: time-domain LFP trace, Welch power spectrum, amplitude histogram, lag-embedded point clouds, ripser persistence diagrams, and Betti signatures.
- Parameters/thresholds: default random seed `20260725`; shuffle-specific parameters are recorded in each `.npz` metadata JSON. PSD diagnostics use the MonkeyData notebook settings: MNE Welch, `fmin=0.5`, `fmax=200`, `n_fft=4096`, `n_overlap=2048`, Gaussian smoothing `sigma=2`. Betti sweeps use the same recipe as the real-data notebook: lag embedding dimension `3`, tau `40`, window `2000`, `200` windows, `300` sampled cloud points per window, seed `42`, `ripser(maxdim=2)`, and Betti threshold `0.30`.
- Outputs: one shuffle `.npz`, one windowed Betti-sweep `.npz`, and one `2x3` diagnostic `.png` per shuffle.

## Statistics

- Tests/models: no hypothesis test is run in this unit.
- Null hypothesis: not evaluated here; the generated traces define future null conditions.
- Alternative hypothesis: not evaluated here.
- Thresholds/decision rule: diagnostic plots are sanity checks only.
- What the statistic means: none; spectra and histograms are descriptive summaries.
- Why this statistic is appropriate here: before running expensive topology, each shuffle must be visually checked to make sure it preserves and destroys the intended signal features.

## Legends

- X axis: row 1 uses time, frequency, and signal value; row 2 uses frequency, phase, percentile, lag, or lag-embedding coordinate depending on the shuffle-specific diagnostic.
- Y axis: row 1 uses standardized signal, PSD, and density; row 2 uses Fourier amplitude, phase density, signal value, autocorrelation, PSD, ratio, or lag-embedding coordinate depending on the panel.
- Color/value: black is real monkey LFP; dark red is the shuffle trace.
- Grouping: one diagnostic per shuffle condition.
- Ordering/sorting: scripts are ordered by the page-2 table: Fourier phase shuffle, IAAFT, aperiodic-only 1/f, envelope-preserving phase reset, one-mode ablation.
- Lines/markers/labels: line traces for time and spectrum; filled or step histograms for amplitude distributions.
- Panels: each diagnostic has six panels arranged as `2x3`.

## Interpretation

- Fourier phase shuffle preserves frequency power but scrambles phase timing.
- IAAFT preserves frequency power and amplitude distribution while scrambling phase timing.
- Aperiodic-only 1/f creates broadband autocorrelated signal without narrow oscillatory peaks.
- Envelope-preserving phase reset keeps slow amplitude modulation but breaks fast phase continuity.
- One-mode ablation smooths the marked 10-32 Hz oscillatory mode into the local PSD background while leaving the rest of the real trace intact.

## Notes

- Status: complete for trace generation and diagnostics.
- Generated cache files: `fourier_phase_shuffle.npz`, `iaaft_shuffle.npz`, `aperiodic_1f.npz`, `envelope_phase_reset.npz`, and `one_mode_ablation.npz`.
- Generated Betti-sweep caches: one `*_betti_sweep.npz` per shuffle in `cache/betti_sweeps/`.
- Generated diagnostics: one `2x3` PNG per shuffle in `plots/`; PDFs are intentionally not generated for now.
- One-mode ablation records candidate peaks in the notebook-style 10-32 Hz PSD and smooths that full band with local Fourier-amplitude interpolation, rather than zeroing power.
- `notebook_style_lfp_psd_check.png` should resemble the pasted manuscript PSD panel because it uses the raw notched channel, not the envelope-normalized topology trace.
- Downstream Betti/torus analyses should read the saved `.npz` files.
- The old synthetic AR(2) null is not one of these five table-2 shuffle rows.
- Keep this unit independent from `ref1_topology_assessment_regen` until the shuffles have passed diagnostic inspection.

## References

- `../../data/monkey_lfp.npz`
- `../../data/monkey_chan1001_notched_raw_20s.npz`
- `plot_null_diagnostics.py`
- `cache_shuffle_betti_sweeps.py`
- `../ref1_topology_assessment_regen/table2_compact.md`
- `../../guide/Figure4_Surrogate_Betti_Tables.pdf`
