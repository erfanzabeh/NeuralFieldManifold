"""Independent AMI diagnostic on frozen cue-offset stage windows; no refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d

from select_trace_embedding_parameters import average_mutual_information


UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
ROOT = UNIT / "outputs/stage_ami_v1"
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")
LAGS = np.arange(1, 101)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configuration():
    return dict(recording="monkeyT_session-y070316009-12_lfp-7", trials=198,
                windows=1188, window_samples=300, sampling_rate_hz=1000,
                stage_names=list(STAGES), stage_labels=list(LABELS), stage_colors=list(COLORS),
                input="frozen processed signals; no preprocessing or cloud reconstruction",
                lags_ms=LAGS.tolist(), primary_bins=48, sensitivity_bins=[32, 64],
                histogram_edges="uniform bins between scope-specific signal percentiles 1 and 99",
                out_of_range_pairs="excluded by the existing estimator; fractions recorded",
                pair_subsampling=False, max_pairs=1000000, seed=42,
                smoothing_sigma_ms=1., smoothing_mode="reflect",
                selection="first interior local minimum of smoothed AMI; no fallback",
                primary_scope="pooled all six stages, equal windows per stage; descriptive",
                training_selection="separate choice using each saved outer training fold only",
                stage_specific="descriptive, not separate applied embedding delays",
                units="nats", refitting=False, decoding=False, ripser=False)


def protected_paths():
    return [SOURCE / "inputs/windows.npz", SOURCE / "inputs/window_metadata.csv",
            SOURCE / "inputs/event_samples.csv", SOURCE / "features.npz",
            SOURCE / "heldout_predictions.npz",
            UNIT / "outputs/tau_sweep_v1/config.json",
            UNIT / "outputs/tau_sweep_v1/tables/overall_f1.csv",
            UNIT / "outputs/tau_sweep_v1/tables/winners.csv"]


def hashes(paths):
    return {str(p.resolve()): sha256(p) for p in paths}


def first_minimum(values, sigma=1.):
    values = np.asarray(values, dtype=float)
    if values.shape != LAGS.shape or not np.isfinite(values).all():
        raise ValueError("AMI curve is incomplete or has wrong lag grid")
    smooth = gaussian_filter1d(values, sigma=sigma, mode="reflect")
    # A flat curve must not produce an artificial delay choice.
    minima = np.flatnonzero((smooth[1:-1] < smooth[:-2]) &
                            (smooth[1:-1] <= smooth[2:])) + 1
    selected = None if not len(minima) else int(LAGS[minima[0]])
    return selected, smooth


def training_segments(processed, metadata, fold):
    return processed[metadata.heldout_fold_zero_based.to_numpy() != fold]


def load_inputs():
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv", keep_default_na=False)
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        processed, clouds = saved["processed"], saved["clouds"]
    assert processed.shape == (1188, 300) and clouds.shape == (1188, 294, 3)
    assert np.isfinite(processed).all() and np.isfinite(clouds).all()
    np.testing.assert_array_equal(metadata.window_index, np.arange(1188))
    assert len(metadata) == 1188 and (metadata.delay == "short").all()
    assert (metadata.end_sample_exclusive - metadata.start_sample == 300).all()
    assert not metadata.preprocessing_error.astype(bool).any()
    grouped = metadata.groupby("trial_index")
    assert len(grouped) == 198 and grouped.size().eq(6).all()
    assert grouped.stage_name.nunique().eq(6).all()
    assert grouped.heldout_fold_zero_based.nunique().eq(1).all()
    assert sorted(metadata.heldout_fold_zero_based.unique()) == list(range(5))
    anchors = ("TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO")
    codes = (202, 203, 204, 206, 207, 207)
    for stage, anchor, code in zip(STAGES, anchors, codes):
        rows = metadata[metadata.stage_name == stage]
        assert len(rows) == 198 and (rows.anchor_name == anchor).all()
        assert (rows.anchor_code == code).all()
        bound = rows.start_sample if stage.startswith("post_") else rows.end_sample_exclusive
        np.testing.assert_array_equal(bound, rows.event_sample)
    for _, rows in grouped:
        ordered = rows.sort_values("stage")
        assert (ordered.start_sample.to_numpy()[1:] >=
                ordered.end_sample_exclusive.to_numpy()[:-1]).all()
    expected = np.stack((processed[:, 6:], processed[:, 3:-3], processed[:, :-6]), axis=-1)
    np.testing.assert_array_equal(clouds, expected)
    return metadata, processed


def calculate_scope(segments, bins):
    values = average_mutual_information(segments, LAGS, bins, 1000000, 42)
    selected, smooth = first_minimum(values)
    low, high = np.percentile(segments, [1, 99])
    records = []
    for lag, raw, filtered in zip(LAGS, values, smooth):
        x, y = segments[:, :-lag], segments[:, lag:]
        retained = ((x >= low) & (x <= high) & (y >= low) & (y <= high)).sum()
        records.append(dict(tau_ms=int(lag), ami_nats=float(raw), smoothed_ami_nats=float(filtered),
                            total_pairs=int(x.size), retained_pairs=int(retained),
                            retained_fraction=float(retained / x.size)))
    return pd.DataFrame(records), selected, float(low), float(high)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    begin = time.perf_counter()
    protected_before = hashes(protected_paths())
    metadata, processed = load_inputs()
    if args.validate_only:
        print("Validated 198 trials, 1,188 corrected windows, grouped folds, and saved tau=3 clouds.")
        return
    sources = hashes([Path(__file__), UNIT / "select_trace_embedding_parameters.py",
                     UNIT / "motor_lfp_utils.py", UNIT / "plot_stage_ami.py"])
    config = configuration()
    if (ROOT / "config.json").exists():
        assert json.loads((ROOT / "config.json").read_text()) == config
        assert json.loads((ROOT / "protected_input_hashes.json").read_text()) == protected_before
        assert json.loads((ROOT / "source_hashes.json").read_text()) == sources
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "checkpoints").mkdir(exist_ok=True)
        (ROOT / "tables").mkdir(exist_ok=True)
        write_json(ROOT / "config.json", config)
        write_json(ROOT / "protected_input_hashes.json", protected_before)
        write_json(ROOT / "source_hashes.json", sources)
        write_json(ROOT / "environment.json", dict(python=platform.python_version(),
                   numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__))
        metadata.to_csv(ROOT / "tables/window_identities.csv", index=False)

    scopes = [("pooled", processed, 48, "descriptive")]
    scopes += [(stage, processed[metadata.stage_name == stage], 48, "stage_descriptive")
               for stage in STAGES]
    scopes += [(f"training_fold_{fold}", training_segments(processed, metadata, fold),
                48, "training_only") for fold in range(5)]
    scopes += [("pooled", processed, bins, "bin_sensitivity") for bins in (32, 64)]
    curves, choices = [], []
    for name, segments, bins, role in scopes:
        path = ROOT / "checkpoints" / f"{name}_bins{bins}.csv"
        record_path = path.with_suffix(".json")
        reused = path.exists() and record_path.exists()
        if reused:
            curve = pd.read_csv(path)
            record = json.loads(record_path.read_text())
        else:
            curve, selected, low, high = calculate_scope(segments, bins)
            record = dict(scope=name, bins=bins, role=role, n_windows=len(segments),
                          selected_tau_ms=selected, lower_bin_edge=low, upper_bin_edge=high,
                          status="ok" if selected is not None else "no_interior_local_minimum")
            curve.to_csv(path, index=False)
            write_json(record_path, record)
        assert len(curve) == 100 and np.isfinite(curve.ami_nats).all()
        selected, smoothed = first_minimum(curve.ami_nats)
        assert selected == record["selected_tau_ms"]
        np.testing.assert_allclose(curve.smoothed_ami_nats, smoothed, atol=1e-12)
        curve["scope"], curve["bins"] = name, bins
        curves.append(curve)
        choices.append(record)
        print(f"{name}, {bins} bins: tau={selected} ms ({len(segments)} windows; "
              f"{'reused' if reused else 'calculated'})", flush=True)

    combined, selections = pd.concat(curves, ignore_index=True), pd.DataFrame(choices)
    combined.to_csv(ROOT / "tables/ami_curves.csv", index=False)
    selections.to_csv(ROOT / "tables/selected_delays.csv", index=False)
    overall = pd.read_csv(UNIT / "outputs/tau_sweep_v1/tables/overall_f1.csv")
    pooled = selections[(selections.scope == "pooled") & (selections.bins == 48)].iloc[0]
    tau = pooled.selected_tau_ms
    comparison = overall[overall.tau_ms.isin([3, 9, 15] + ([] if pd.isna(tau) else [int(tau)]))]
    comparison.to_csv(ROOT / "tables/existing_f1_context.csv", index=False)
    assert hashes(protected_paths()) == protected_before
    write_json(ROOT / "validation.json", dict(trials=198, windows=1188, curves=len(scopes),
               samples_per_window=300, anchors_verified=True, native_tau3_clouds_exact=True,
               trial_grouped_folds=True, protected_hashes_unchanged=True,
               full_pair_counts=True, geometric_fits_called=False, decoder_called=False,
               runtime_seconds=time.perf_counter() - begin))
    chosen = "none within 1-100 ms" if pd.isna(tau) else f"{int(tau)} ms"
    folds = selections[selections.role == "training_only"].selected_tau_ms.tolist()
    stage_choices = selections[selections.role == "stage_descriptive"]
    stages_text = "; ".join(f"{r.scope}: {r.selected_tau_ms:g} ms" for r in stage_choices.itertuples())
    bin_text = "; ".join(f"{r.bins} bins: {r.selected_tau_ms:g} ms" for r in
                         selections[selections.scope == "pooled"].sort_values("bins").itertuples())
    (ROOT / "REPORT.md").write_text(
        "# Stage-window AMI diagnostic\n\n"
        f"All 198 short-delay trials contributed six corrected 300-ms windows (1,188 total). "
        f"The pooled primary AMI curve suggests **tau = {chosen}**, using the first interior "
        "local minimum after sigma=1-ms smoothing. This selection uses no stage labels or decoding scores.\n\n"
        f"Saved outer-training-fold choices (folds 0-4): **{folds} ms**. "
        f"Histogram-bin sensitivity: {bin_text}. Stage-specific descriptive choices: {stages_text}.\n\n"
        "AMI was computed between scalar signal values separated by each lag from 1-100 ms, "
        "pooling within-window pairs without crossing boundaries. The existing histogram estimator "
        "uses uniform bins between signal percentiles 1 and 99; excluded-pair fractions are saved. "
        "All available in-range pairs were used, with no random subsampling. Stage-specific delays "
        "are diagnostics, not six separate applied embedding parameters.\n\n"
        "The result is a delay heuristic, not proof of correct topology, fit quality, or optimal decoding. "
        "Short, filtered windows and histogram/smoothing choices limit interpretation. No geometry, "
        "feature, decoder, or persistence result was recalculated. Existing native and matched F1 "
        "scores are copied only for context where the suggested delay was already swept; the AMI "
        "diagnostic does not independently validate those scores. The currently adopted tau remains 3 ms.\n\n"
        f"Computation/validation wall time: {time.perf_counter()-begin:.2f} seconds. "
        "Protected source-result hashes were unchanged.\n")


if __name__ == "__main__":
    main()
