"""Compare saved MAD clouds with trial-envelope normalization; geometry only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal

from motor_lfp_utils import bandpass_filter, detrend_zscore, lag_embed


UNIT = Path(__file__).resolve().parent
RECORDING = "monkeyT_session-y070316009-12_lfp-7"
SOURCE = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
RAW_SOURCE = UNIT / "lfp_converted_data" / f"{RECORDING}.npz"
OUTPUT = UNIT / "outputs/stage_tau3_envelope_geometry_v1"
TRIALS = (6, 9, 10)
FS, DIMENSION, TAU, WINDOW = 1000, 3, 3, 300
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protected_hashes():
    paths = [RAW_SOURCE, SOURCE / "inputs/windows.npz",
             SOURCE / "inputs/window_metadata.csv", SOURCE / "config.json",
             SOURCE / "features.npz", SOURCE / "heldout_predictions.npz",
             SOURCE / "bootstrap.npz", UNIT / "motor_lfp_utils.py"]
    paths.extend((UNIT / "outputs/tau_sweep_v1").glob("*.npz"))
    paths.extend((UNIT / "outputs/tau_sweep_v1").glob("tables/*.csv"))
    betti = UNIT / "outputs/macaque_task_six_stages_cue_offset_correct/BETTI_STAGE_v1"
    paths.extend(betti.glob("tables/*.csv"))
    return {str(path.resolve()): sha256(path) for path in sorted(set(paths))}


def filtered_without_scaling(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
        raise ValueError("Expected a finite nonconstant signal")
    y = signal.detrend(x, type="linear")
    return bandpass_filter(y - np.median(y), fs=FS, low=2., high=55., order=4)


def trial_envelope(raw_trial):
    full_filtered = filtered_without_scaling(raw_trial)
    unsmoothed = np.abs(signal.hilbert(full_filtered))
    b, a = signal.butter(2, 3. / (FS / 2), btype="lowpass")
    envelope = signal.filtfilt(b, a, unsmoothed)
    if not np.isfinite(envelope).all() or np.any(envelope <= 0):
        raise ValueError("Nonpositive/nonfinite trial envelope; no clipping or fallback")
    return full_filtered, envelope, float(np.median(envelope))


def envelope_window(raw_window, envelope, shared_scale):
    if raw_window.shape != (WINDOW,) or envelope.shape != (WINDOW,):
        raise ValueError("Expected matching 300-sample windows")
    return filtered_without_scaling(raw_window) / (envelope + 1e-6) * shared_scale


def prepare():
    previous = json.loads((SOURCE / "config.json").read_text())
    assert previous["recording"] == RECORDING
    assert previous["m"] == DIMENSION and previous["tau_samples"] == TAU
    assert tuple(previous["stage_names"]) == STAGES
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv", keep_default_na=False)
    selected = metadata[metadata.original_trial_number.isin(TRIALS)].sort_values(
        ["original_trial_number", "stage"]).reset_index(drop=True)
    if len(selected) != 18 or not selected.delay.eq("short").all():
        raise ValueError("Expected 18 saved short-delay windows")
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        raw_windows = saved["raw"][selected.window_index.to_numpy()].copy()
        old_signals = saved["processed"][selected.window_index.to_numpy()].copy()
        old_clouds = saved["clouds"][selected.window_index.to_numpy()].copy()
    new_signals, envelopes, new_clouds, diagnostics = [], [], [], []
    trials, full_filtered, full_envelopes = [], [], []
    with np.load(RAW_SOURCE, allow_pickle=True) as data:
        assert int(data["sampling_rate_hz"]) == FS
        for trial in TRIALS:
            rows = selected[selected.original_trial_number == trial]
            assert tuple(rows.stage_name) == STAGES
            assert rows.stage.tolist() == list(range(6))
            assert rows.raw_trial_index.nunique() == rows.heldout_fold_zero_based.nunique() == 1
            raw_index = int(rows.iloc[0].raw_trial_index)
            assert int(data["original_trial_number"][raw_index]) == trial
            assert str(data["delay_label"][raw_index]) == "short"
            assert int(data["direction"][raw_index]) == int(rows.iloc[0].direction)
            raw_trial = data["lfp"][raw_index].copy()
            filtered, envelope, scale = trial_envelope(raw_trial)
            trials.append(raw_trial)
            full_filtered.append(filtered)
            full_envelopes.append(envelope)
            for row in rows.itertuples():
                i = row.Index
                start, end = int(row.start_sample), int(row.end_sample_exclusive)
                assert end - start == WINDOW and start >= 0 and end <= len(raw_trial)
                np.testing.assert_array_equal(raw_trial[start:end], raw_windows[i])
                if row.stage in (1, 3):
                    assert row.anchor_name == ("TC_off" if row.stage == 1 else "SC_off")
                    assert row.anchor_code == (203 if row.stage == 1 else 206)
                    assert start == row.event_sample
                elif row.stage % 2 == 0:
                    assert end == row.event_sample
                else:
                    assert start == row.event_sample
                x = raw_windows[i]
                reproduction = bandpass_filter(detrend_zscore(x), FS, 2., 55., 4).astype(np.float32)
                np.testing.assert_array_equal(reproduction, old_signals[i])
                np.testing.assert_array_equal(lag_embed(reproduction, DIMENSION, TAU), old_clouds[i])
                y = signal.detrend(x.astype(np.float64), type="linear")
                mad_scale = 1.4826 * np.median(np.abs(y - np.median(y)))
                np.testing.assert_allclose(filtered_without_scaling(x) / mad_scale,
                                           old_signals[i], rtol=1e-6, atol=1e-7)
                env = envelope[start:end]
                normalized = envelope_window(x, env, scale)
                cloud = lag_embed(normalized, DIMENSION, TAU)
                np.testing.assert_array_equal(cloud, np.column_stack(
                    (normalized[6:], normalized[3:-3], normalized[:-6])))
                assert cloud.shape == (294, 3) and np.isfinite(cloud).all()
                new_signals.append(normalized)
                envelopes.append(env)
                new_clouds.append(cloud)
                diagnostics.append(dict(window_index=int(row.window_index),
                                        original_trial_number=trial, stage=row.stage_name,
                                        mad_scale=float(mad_scale),
                                        trial_envelope_median=scale,
                                        window_envelope_min=float(env.min()),
                                        window_envelope_max=float(env.max())))
    assert len(new_clouds) == 18
    np.savez_compressed(OUTPUT / "inputs_and_clouds.npz", raw=raw_windows,
                        old_processed=old_signals, old_clouds=old_clouds,
                        envelope_processed=np.stack(new_signals),
                        envelope_clouds=np.stack(new_clouds),
                        window_envelopes=np.stack(envelopes),
                        raw_trials=np.stack(trials), full_filtered=np.stack(full_filtered),
                        full_envelopes=np.stack(full_envelopes))
    selected.to_csv(OUTPUT / "windows.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "normalization_diagnostics.csv", index=False)
    write_json(OUTPUT / "config.json", dict(
        recording=RECORDING, source=str(SOURCE), original_trial_numbers=list(TRIALS),
        sampling_rate_hz=FS, delay="short", dimension=DIMENSION, delay_ms=TAU,
        window_samples=WINDOW, points_per_cloud=294,
        stage_names=list(STAGES), stage_labels=list(LABELS),
        stage_colors=list(COLORS), selection="previously established trials 6, 9, 10",
        normalization="full-trial Hilbert envelope; second-order 3-Hz zero-phase smoothing; "
                      "multiply by one full-trial median envelope",
        normalization_formula="window_bandpassed / (trial_envelope_window + 1e-6) * trial_median_envelope",
        envelope_context="each complete 6301-sample trial; never concatenated",
        window_preprocessing="same per-window linear detrend, median centering and fourth-order "
                             "2-55 Hz zero-phase filtering; no per-window amplitude division",
        full_trial_preprocessing="linear detrend, median centering, fourth-order 2-55 Hz zero-phase filter",
        difference_from_legacy_mouse="same envelope division/rescaling family, but fixed macaque band "
                                     "and existing per-window filtering retained; envelope estimated on full trial",
        display="fixed camera; common limits across six stages within each normalization/trial; "
                "MAD and envelope rows have separate limits because their units differ",
        camera=dict(elevation=24, azimuth=-58),
        geometric_fitting=False, decoding=False, pca=False, persistent_homology=False,
        versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                      matplotlib=matplotlib.__version__, pandas=pd.__version__)))


def normalization_checks():
    t = np.arange(6301) / FS
    x = (1.2 + .3 * np.sin(2 * np.pi * .4 * t)) * np.sin(2 * np.pi * 22 * t)
    _, env1, scale1 = trial_envelope(x)
    _, env2, scale2 = trial_envelope(3 * x)
    np.testing.assert_allclose(env2, 3 * env1, rtol=1e-8, atol=1e-9)
    a = envelope_window(x[2500:2800], env1[2500:2800], scale1)
    b = envelope_window(3 * x[2500:2800], env2[2500:2800], scale2)
    np.testing.assert_allclose(b, 3 * a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(detrend_zscore(3 * x[2500:2800]),
                               detrend_zscore(x[2500:2800]), rtol=1e-10, atol=1e-10)


def geometry_axis(ax, points, color, limit):
    line, = ax.plot(*points.T, color=color, lw=.8, alpha=1., antialiased=True,
                    solid_capstyle="round", solid_joinstyle="round")
    np.testing.assert_array_equal(np.column_stack(line.get_data_3d()), points)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1), zoom=1.16)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()


def export_png(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *(ax.title for ax in fig.axes)]:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
    destination = OUTPUT / name
    fig.savefig(destination, dpi=600, facecolor="white")
    plt.close(fig)
    print(destination, flush=True)
    return dict(name=name, sha256=sha256(destination))


def render():
    config = json.loads((OUTPUT / "config.json").read_text())
    metadata = pd.read_csv(OUTPUT / "windows.csv")
    with np.load(OUTPUT / "inputs_and_clouds.npz") as data:
        old, new = data["old_clouds"], data["envelope_clouds"]
    assert old.shape == new.shape == (18, 294, 3)
    assert np.isfinite(old).all() and np.isfinite(new).all()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#263238", "figure.facecolor": "white"})
    exports, scales = [], []
    for trial in TRIALS:
        indices = np.flatnonzero(metadata.original_trial_number.to_numpy() == trial)
        assert metadata.iloc[indices].stage.tolist() == list(range(6))
        limits = [float(np.max(np.abs(clouds[indices]))) * 1.08 for clouds in (old, new)]
        scales.append(dict(original_trial_number=trial, mad_limit=limits[0], envelope_limit=limits[1]))
        fig = plt.figure(figsize=(9.2, 6.2), facecolor="white")
        fig.text(.04, .963, f"Trial {trial} | Hilbert-envelope normalization", fontsize=13, weight="bold")
        fig.text(.04, .92, "Monkey T | channel 7 | short delay | m = 3 | tau = 3 ms", fontsize=10)
        for stage, index in enumerate(indices):
            position = (stage % 2) * 3 + stage // 2 + 1
            ax = fig.add_subplot(2, 3, position, projection="3d", proj_type="ortho")
            geometry_axis(ax, new[index], config["stage_colors"][stage], limits[1])
            ax.set_title(config["stage_labels"][stage], fontsize=11, pad=0)
        fig.subplots_adjust(left=.035, right=.965, top=.85, bottom=.025, wspace=.02, hspace=.08)
        exports.append(export_png(fig, f"envelope_geometry_trial_{trial}.png"))

        fig = plt.figure(figsize=(14.4, 5.4), facecolor="white")
        fig.text(.07, .954, f"Trial {trial} | m = 3 | tau = 3 ms", fontsize=13, weight="bold")
        for row, (clouds, label, unit) in enumerate(((old, "Window MAD", "Normalized units"),
                                                  (new, "Hilbert envelope", "Signal units"))):
            fig.text(.012, .66 - row * .41, f"{label}\n{unit}", fontsize=10,
                     ha="left", va="center", rotation=90)
            for stage, index in enumerate(indices):
                ax = fig.add_subplot(2, 6, row * 6 + stage + 1, projection="3d", proj_type="ortho")
                geometry_axis(ax, clouds[index], config["stage_colors"][stage], limits[row])
                if row == 0:
                    ax.set_title(config["stage_labels"][stage], fontsize=10, pad=0)
        fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.02, wspace=.012, hspace=.06)
        exports.append(export_png(fig, f"mad_vs_envelope_trial_{trial}.png"))
    pd.DataFrame(scales).to_csv(OUTPUT / "display_scales.csv", index=False)
    return exports


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    before = protected_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        assert json.loads((OUTPUT / "protected_hashes.json").read_text()) == before
    else:
        if (OUTPUT / "protected_hashes.json").exists():
            raise FileExistsError("Saved clouds already exist; use --plot-only")
        normalization_checks()
        prepare()
        write_json(OUTPUT / "protected_hashes.json", before)
    exports = render()
    assert protected_hashes() == before
    write_json(OUTPUT / "validation.json", dict(
        trials=list(TRIALS), stages_per_trial=6, windows=18, points_per_cloud=294,
        saved_mad_signals_and_clouds_exact=True, raw_window_identity_verified=True,
        cue_offset_anchors_verified=True, no_envelope_clipping_or_fallback=True,
        whole_trial_envelope_positive=True, amplitude_scaling_test_passed=True,
        previous_results_unchanged=True, fitter_called=False, decoder_called=False,
        ripser_called=False, pca_called=False, exports=exports,
        source_sha256=sha256(Path(__file__))))
    (OUTPUT / "README.md").write_text(
        "# Stage geometry: alternative normalization\n\n"
        "Same Monkey T recording y070316009-12, channel 7, short-delay trials 6, 9 and 10. "
        "Each `envelope_geometry_trial_*.png` shows the six corrected 300-ms windows. "
        "Each `mad_vs_envelope_trial_*.png` pairs the exact saved MAD geometry (top) "
        "with the alternative geometry (bottom). Post-TC/SC remain cue-offset aligned.\n\n"
        "Window detrending, median centering, fourth-order 2-55 Hz zero-phase filtering, "
        "m=3 and tau=3 ms are unchanged. Instead of dividing each window by its own MAD, "
        "we divide by a Hilbert amplitude envelope estimated from the entire individual "
        "6301-sample trial, smoothed with a second-order 3-Hz zero-phase filter, and "
        "multiply by that trial's median envelope. No trials or stage windows are concatenated. "
        "The envelope uses full-trial context; this is an offline visualization, not a causal "
        "or stage-local decoder preprocessing proposal. This is the older mouse envelope "
        "normalization family, with the macaque band and original window filters retained.\n\n"
        "The 294 observed delayed samples are connected in time without smoothing, PCA, "
        "AR prediction or geometric fitting. No decoder or persistent homology was run. "
        "PNG exports are 600 dpi. All six stages share a camera and scale within each "
        "trial and method. MAD and envelope rows use different numeric limits because "
        "their units differ; absolute sizes cannot be compared between rows.\n\n"
        "Saved signals, envelopes and clouds are in `inputs_and_clouds.npz`; configuration, "
        "metadata, diagnostics and protected-file hashes support reproduction. Existing "
        "geometry, decoding and topology files were hash-checked unchanged. Plot-only "
        "regeneration: `plot_stage_normalization_geometry.py --plot-only`.\n")


if __name__ == "__main__":
    main()
