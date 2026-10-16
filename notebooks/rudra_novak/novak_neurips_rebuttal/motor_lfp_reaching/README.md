# Macaque Motor-Cortex LFP Reaching Analysis

This folder contains the NeurIPS rebuttal analysis for the public Confais/Kilavik/Riehle macaque motor-cortex LFP dataset. The analysis uses the dataset's documented labels: six reach directions crossed with short/long delay trials, with two-direction subset sessions retained but treated according to their available labels.

Run everything with the project conda environment:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python -m pip install h5py

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/download_motor_lfp_data.py

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/convert_motor_lfp.py

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_motor_lfp_decoding.py

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/select_lag_embedding_order.py

# Per-trace macaque rerun: each unique LFP gets its own tau and embedding dimension.
/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/select_trace_embedding_parameters.py \
  --n-jobs 12 --n-bootstraps 120 --output-suffix pertrace_tau_dim

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_motor_lfp_decoding.py \
  --torus-params-csv notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/tables/lfp_embedding_params_pertrace_tau_dim.csv \
  --output-suffix pertrace_tau_dim

# Slower reviewer-facing torus refit. This replaces the quick PCA/SVD torus
# proxy with a nonlinear elliptical-torus least-squares fit per trial.
/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_motor_lfp_nonlinear_refit.py \
  --torus-params-csv notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/tables/lfp_embedding_params_pertrace_tau_dim.csv \
  --output-suffix pertrace_tau_dim --n-jobs 12 --n-torus-points 300 --max-nfev 1200
```

Use `--force` on `run_motor_lfp_nonlinear_refit.py` only when intentionally recomputing all torus fits.

Large files are intentionally local-only and git-ignored:

- `raw_data/MonkeyT.mat`, `raw_data/MonkeyM.mat`
- `lfp_converted_data/*.npz`
- `cache/features/*.npz`
- `cache/nonlinear_refit/*.npz`

Tracked, reviewer-facing outputs:

- `tables/lfp_manifest.csv`: one row per unique `(monkey, session_ID, LFP_ID)`.
- `tables/condition_counts.csv`: trial counts for each direction x delay label.
- `tables/single_band_decode_scores.csv`: one-band LDA decoders.
- `tables/average_psd_decode_scores.csv`: one-scalar average PSD decoder.
- `tables/multiband_decode_scores.csv`: all-band power baseline.
- `tables/torus_decode_scores.csv`: delay-geometry feature decoder.
- `tables/summary_decode_scores.csv`: mean and standard deviation across unique LFPs.
- `tables/nonlinear_refit_direction_movement_scores.csv`: per-LFP movement-direction results from the slower nonlinear torus refit.
- `tables/nonlinear_refit_direction_movement_summary.csv`: mean and standard deviation for single-band, all-band, and nonlinear torus decoders.
- `tables/relevant_band_selection.csv`: best single-band selection for the full-six-direction and per-monkey comparisons.
- `tables/torus_avgpsd_bestband_f1_scores.csv`: matched LFP-level F1 values for torus features, average PSD, and best single band.
- `tables/torus_avgpsd_bestband_significance.csv`: paired Wilcoxon tests with Holm correction.
- `tables/torus_avgpsd_bestband_session_skips.csv`: sessions skipped because they have fewer than 5 full-six-direction LFPs.
- `tables/embedding_order_selection.csv`: dataset-level PSD peak count and recommended lag-embedding dimension.
- `tables/embedding_order_psd_peaks.csv`: detected PSD peaks, bootstrap support, and line-noise exclusion.
- `tables/embedding_order_conclusion.md`: concise written conclusion for choosing the macaque lag-embedding order.
- `tables/lfp_embedding_params_pertrace_tau_dim.csv`: per-LFP tau and embedding dimension chosen from AMI and PSD peaks.
- `tables/lfp_embedding_peak_table_pertrace_tau_dim.csv`: per-LFP detected PSD peaks and support values.
- `tables/summary_decode_scores_pertrace_tau_dim.csv`: broad decoding summary with per-LFP torus embedding parameters.
- `tables/nonlinear_refit_direction_movement_summary_pertrace_tau_dim.csv`: nonlinear movement-direction summary with per-LFP torus embedding parameters.
- `tables/torus_avgpsd_bestband_f1_scores_pertrace_tau_dim.csv`: matched LFP-level comparison for torus features, average PSD, and best single band under the per-trace rerun.
- `tables/torus_avgpsd_bestband_significance_pertrace_tau_dim.csv`: paired Wilcoxon tests for the per-trace rerun.
- `tables/nonlinear_refit_interpretation.md`: short interpretation of the nonlinear torus refit result.
- `tables/rebuttal_interpretation.md`: short neuroscientist-facing interpretation.
- `plots/summary/*`: task sanity, bar plots, heatmaps, and mean confusion matrices.

The main comparison plot is:

```text
plots/summary/direction_movement_feature_f1_barplot.png
```

It compares single bands, all-band power, and torus geometry for movement-aligned reach-direction decoding.

For the slower torus result, use:

```text
plots/summary/direction_movement_feature_f1_barplot_nonlinear_torus_full_6_direction_lfps.png
plots/summary/direction_movement_feature_f1_barplot_nonlinear_torus_all_unique_lfps.png
plots/summary/direction_movement_torus_nonlinear_mean_confusion.png
plots/summary/direction_movement_torus_avgpsd_bestband_f1_boxplot.png
plots/summary/direction_movement_torus_avgpsd_bestband_f1_boxplot_by_monkey.png
plots/summary/macaque_lfp_embedding_order_figure4c_style.png
plots/summary/macaque_lfp_embedding_order_raw_psd.png
plots/summary/macaque_lfp_embedding_order_residual_psd.png
plots/summary/macaque_lfp_embedding_order_peak_support.png
plots/summary/lfp_embedding_parameter_summary_pertrace_tau_dim.png
plots/summary/macaque_lfp_pertrace_embedding_figure4c_style_pertrace_tau_dim.png
plots/summary/direction_epoch_feature_f1_heatmap_pertrace_tau_dim.png
plots/summary/direction_movement_feature_f1_barplot_pertrace_tau_dim.png
plots/summary/direction_movement_feature_f1_barplot_nonlinear_torus_full_6_direction_lfps_pertrace_tau_dim.png
plots/summary/direction_movement_torus_avgpsd_bestband_f1_boxplot_pertrace_tau_dim.png
plots/summary/direction_movement_torus_avgpsd_bestband_f1_boxplot_by_monkey_pertrace_tau_dim.png
```

In the nonlinear run, `torus_nonlinear_15` is the reviewer-facing torus feature set. The older `torus_geometry_15` output is a fast screening estimate based on PCA/SVD geometry and should not be described as the final torus refit.

The dataset-level PSD-based embedding-order diagnostic is retained as a sanity check, but the current reviewer-facing rerun uses per-trace parameters instead of one global order. For each unique LFP trace, tau is chosen from the first local minimum of average mutual information, with autocorrelation fallbacks available if AMI is flat. Embedding dimension is chosen from robust per-trace PSD peaks in 2-55 Hz after 1/f correction and 48-52 Hz line-band exclusion, using `dimension = 2K + 1` capped at 9D.

For the `pertrace_tau_dim` rerun, each LFP epoch is delay-embedded with that LFP's selected tau and dimension. The same 15 torus geometry features are retained by projecting delay clouds above 3D onto their leading three principal coordinates before applying the existing 3D elliptical torus fit. This keeps the plots comparable while avoiding a single dataset-wide lag-embedding choice.
