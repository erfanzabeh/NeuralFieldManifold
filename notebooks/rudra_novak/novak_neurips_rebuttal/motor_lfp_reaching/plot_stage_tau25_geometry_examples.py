"""Observed delay trajectories from frozen stage signals; no fitting or decoding."""
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

from motor_lfp_utils import lag_embed


UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
AMI = UNIT / "outputs/stage_ami_v1"
ROOT = UNIT / "outputs/stage_tau25_geometry_examples_v1"
TRIALS = (6, 9, 10)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_hashes(paths):
    return {str(path.resolve()): sha256(path) for path in paths}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protected_paths():
    return [SOURCE / "inputs/windows.npz", SOURCE / "inputs/window_metadata.csv",
            SOURCE / "config.json", SOURCE / "features.npz",
            SOURCE / "heldout_predictions.npz", AMI / "config.json",
            AMI / "tables/selected_delays.csv"]


def prepare():
    config = json.loads((AMI / "config.json").read_text())
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv", keep_default_na=False)
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        signals, reference = saved["processed"], saved["clouds"]
    assert config["sampling_rate_hz"] == 1000
    assert signals.shape == (1188, 300) and reference.shape == (1188, 294, 3)
    assert np.isfinite(signals).all() and np.isfinite(reference).all()
    assert len(metadata) == 1188 and metadata.delay.eq("short").all()
    selected = metadata[metadata.original_trial_number.isin(TRIALS)].copy()
    selected = selected.sort_values(["original_trial_number", "stage"]).reset_index(drop=True)
    assert len(selected) == 18
    clouds3, clouds25 = [], []
    for trial in TRIALS:
        rows = selected[selected.original_trial_number == trial]
        assert rows.stage.tolist() == list(range(6))
        assert rows.stage_name.tolist() == config["stage_names"]
        assert rows.heldout_fold_zero_based.nunique() == 1
    for row in selected.itertuples():
        signal = signals[row.window_index]
        cloud3, cloud25 = (lag_embed(signal, dim=3, tau=tau) for tau in (3, 25))
        np.testing.assert_array_equal(cloud3, reference[row.window_index])
        np.testing.assert_array_equal(cloud25, np.column_stack(
            (signal[50:], signal[25:-25], signal[:-50])))
        assert cloud25.shape == (250, 3) and np.isfinite(cloud25).all()
        assert row.end_sample_exclusive - row.start_sample == 300
        if row.stage in (1, 3):
            assert row.anchor_name == ("TC_off" if row.stage == 1 else "SC_off")
            assert row.anchor_code == (203 if row.stage == 1 else 206)
            assert row.start_sample == row.event_sample
        clouds3.append(cloud3)
        clouds25.append(cloud25)
    np.savez_compressed(ROOT / "clouds.npz", tau3=np.stack(clouds3), tau25=np.stack(clouds25))
    selected.to_csv(ROOT / "windows.csv", index=False)
    write_json(ROOT / "config.json", dict(
        source=str(SOURCE), original_trial_numbers=list(TRIALS), dimension=3,
        delays_ms=[3, 25], samples_per_window=300, points_per_window={"3": 294, "25": 250},
        selection="previously established original trial IDs; no new appearance-based selection",
        stage_names=config["stage_names"], stage_labels=config["stage_labels"],
        stage_colors=config["stage_colors"], coordinates="x(t), x(t-tau), x(t-2*tau)",
        camera=dict(elevation=24, azimuth=-58),
        scale="common centered limits for both delays and all six stages within each trial",
        geometry_refitting=False, decoding=False, preprocessing=False, pca=False))


def trajectory_axis(ax, points, color, limit):
    line, = ax.plot(*points.T, color=color, lw=.8, alpha=1.,
                    solid_capstyle="round", solid_joinstyle="round", antialiased=True)
    np.testing.assert_array_equal(np.column_stack(line.get_data_3d()), points)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *(ax.title for ax in fig.axes)]:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
    path = ROOT / name
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)
    print(path, flush=True)
    return dict(path=name, sha256=sha256(path))


def render():
    config = json.loads((ROOT / "config.json").read_text())
    metadata = pd.read_csv(ROOT / "windows.csv")
    with np.load(ROOT / "clouds.npz") as saved:
        clouds3, clouds25 = saved["tau3"], saved["tau25"]
    assert clouds3.shape == (18, 294, 3) and clouds25.shape == (18, 250, 3)
    assert np.isfinite(clouds3).all() and np.isfinite(clouds25).all()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#263238", "figure.facecolor": "white"})
    exports, scales = [], []
    for trial in TRIALS:
        indices = np.flatnonzero(metadata.original_trial_number.to_numpy() == trial)
        assert metadata.iloc[indices].stage.tolist() == list(range(6))
        limit = max(float(np.max(np.abs(clouds3[indices]))),
                    float(np.max(np.abs(clouds25[indices])))) * 1.08
        scales.append(dict(original_trial_number=trial, minimum=-limit, maximum=limit))
        fig = plt.figure(figsize=(9., 6.1), facecolor="white")
        fig.suptitle(f"Original trial {trial} | m = 3 | delay = 25 ms", y=.98, fontsize=13)
        for stage, index in enumerate(indices):
            # Columns pair the before/after windows for TC, SC, and GO.
            position = (stage % 2) * 3 + stage // 2 + 1
            ax = fig.add_subplot(2, 3, position, projection="3d", proj_type="ortho")
            trajectory_axis(ax, clouds25[index], config["stage_colors"][stage], limit)
            ax.set_title(config["stage_labels"][stage], fontsize=11, pad=0)
        fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.03, wspace=.02, hspace=.08)
        exports.append(save(fig, f"tau25_trial_{trial}.png"))

        fig = plt.figure(figsize=(13.2, 4.7), facecolor="white")
        fig.suptitle(f"Original trial {trial} | m = 3", y=.99, fontsize=13)
        for row, (tau, clouds) in enumerate(((3, clouds3), (25, clouds25))):
            fig.text(.022, .72 - row * .43, f"{tau} ms", rotation=90, ha="center", va="center",
                     fontsize=11, weight="bold")
            for stage, index in enumerate(indices):
                ax = fig.add_subplot(2, 6, row * 6 + stage + 1, projection="3d", proj_type="ortho")
                trajectory_axis(ax, clouds[index], config["stage_colors"][stage], limit)
                if row == 0:
                    ax.set_title(config["stage_labels"][stage], fontsize=10, pad=0)
        fig.subplots_adjust(left=.05, right=.985, top=.87, bottom=.03, wspace=.015, hspace=.08)
        exports.append(save(fig, f"tau3_vs_tau25_trial_{trial}.png"))
    pd.DataFrame(scales).to_csv(ROOT / "view_scales.csv", index=False)
    return exports


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    before = file_hashes(protected_paths())
    ROOT.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        if (ROOT / "clouds.npz").exists():
            raise FileExistsError("Saved examples already exist; use --plot-only")
        prepare()
        write_json(ROOT / "protected_hashes.json", before)
    assert json.loads((ROOT / "protected_hashes.json").read_text()) == before
    exports = render()
    assert file_hashes(protected_paths()) == before
    write_json(ROOT / "validation.json", dict(
        original_trial_numbers=list(TRIALS), windows=18, saved_tau3_clouds_exact=True,
        tau25_coordinates_verified=True, cue_offset_anchors_verified=True,
        no_smoothing_or_resampling=True, previous_results_unchanged=True,
        fitter_called=False, decoder_called=False, exports=exports,
        sources=file_hashes([Path(__file__), UNIT / "motor_lfp_utils.py"])))
    (ROOT / "README.md").write_text(
        "# Observed stage geometry at 25-ms delay\n\n"
        "Trials 6, 9, and 10 are the previously established examples. Each `tau25_trial_*.png` "
        "shows the six corrected windows; columns pair before/after TC, SC, and GO. "
        "Each `tau3_vs_tau25_trial_*.png` compares the same six windows at 3 ms (top) and "
        "25 ms (bottom), in chronological stage order.\n\n"
        "The frozen 300-ms processed signals were embedded directly as "
        "[x(t), x(t-tau), x(t-2*tau)]: 250 observed points at 25 ms and 294 at 3 ms. "
        "Lines connect actual samples in temporal order without smoothing, resampling, "
        "PCA, or AR prediction. No fitted ellipse or torus is shown; no geometric fitting "
        "or decoding was run. Post-TC and post-SC begin at the recorded cue offsets.\n\n"
        "Both delays and all stages share limits and camera within each trial, "
        "including across its two panels. Limits vary between trials and are saved "
        "in `view_scales.csv`. Units are normalized signal units. Inputs and previous "
        "results were hash-verified unchanged. PNGs are 600 dpi.\n\n"
        "Saved coordinates: `clouds.npz`; window identities: `windows.csv`. "
        "Regenerate only plots with `plot_stage_tau25_geometry_examples.py --plot-only`.\n")


if __name__ == "__main__":
    main()
