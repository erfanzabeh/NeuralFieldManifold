# Novak NeurIPS rebuttal EEG sleep decoding

This folder contains the per-session conversion and decoding rerun for the
`EEG_EEG1.1A-B_EMG_EMG.1` mouse cortical EEG dataset.

## Environment

Use the project conda environment for every command:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python -m pip install -e ".[dev]"
```

## Commands

```bash
cd /home/nochen/code/NeuralFieldManifold

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/convert_mat_to_npy.py --force

/home/nochen/miniconda3/envs/neuralmanifold/bin/python \
  notebooks/rudra_novak/novak_neurips_rebuttal/run_sleep_decoding.py --force --n-jobs -1

/home/nochen/miniconda3/envs/neuralmanifold/bin/python -m pip install ripser persim
```

## Outputs

- `eeg_npy_data/`: converted per-session `signal.npy`, `time.npy`, and
  aligned `state.npy` files, plus `session_manifest.csv`.
- `cache/`: per-session cached feature arrays, predictions, and status files.
- `plots/session_mSTART_END/ref2_style.png`: per-session figure.
- `plots/summary/session_f1_heatmap.png`: summary heatmap.
- `plots/summary/session_f1_barplot.png`: collapsed cross-session F1 bar plot
  with session standard-deviation error bars.
- `plots/summary/session_accuracy_heatmap.png`: summary decoding-accuracy
  heatmap.
- `plots/summary/session_accuracy_barplot.png`: collapsed cross-session
  decoding-accuracy bar plot with session standard-deviation error bars.
- `tables/session_class_counts.csv`: all converted sessions and labels.
- `tables/session_decode_scores.csv`: per-session decoder metrics.
- `tables/session_feature_f1_summary.csv`: mean and standard deviation F1
  values used in the collapsed summary bar plot.
- `tables/session_feature_accuracy_summary.csv`: mean and standard deviation
  accuracy values used in the collapsed accuracy bar plot.
- `tables/session_summary.md`: compact F1 table.
- `notebooks/inspect_results.ipynb`: quick notebook for browsing results.
- `eeg_torus_fit.ipynb`: one-session persistent-homology notebook for
  testing the EEG torus Betti signature hypothesis.
- `cache/eeg_torus_fit_session_m0900_0960.npz`: Betti sweep cache from the
  topology notebook.
- `tables/eeg_torus_betti_summary.csv`: topology summary for the selected EEG
  session.
- `plots/topology/session_m0900_0960/`: topology diagnostic plots.
- `motor_lfp_reaching/`: public macaque motor-cortex LFP reaching analysis for
  the reviewer-requested repeated movement setting, including single-band power,
  all-band power, torus geometry decoding comparisons, and a PSD-based
  dataset-level lag-embedding order diagnostic.
