"""One long-delay trial: observed six-stage lag trajectories, tau 1-20 ms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from convert_motor_lfp import h5_cell_dataset, h5_cell_array, trial_column
from motor_lfp_utils import bandpass_filter, detrend_zscore, lag_embed


UNIT = Path(__file__).resolve().parent
RECORDING = "monkeyT_session-y070316009-12_lfp-7"
SOURCE = UNIT / "lfp_converted_data" / f"{RECORDING}.npz"
DESTINATION = UNIT / "outputs/long_delay_trial_v1"
CORRECTED = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
AMI_CONFIG = UNIT / "outputs/stage_ami_v1/config.json"
TAUS = tuple(range(1, 21))
EVENT_NAMES = ("TC_on", "TC_off", "SC_on", "SC_off", "GO")
EVENT_CODES = (202, 203, 204, 206, 207)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_trial():
    with np.load(SOURCE, allow_pickle=True) as saved:
        delays = saved["delay_label"].astype(str)
        original_ids = saved["original_trial_number"]
        candidates = np.flatnonzero(delays == "long")
        index = int(candidates[np.argmin(original_ids[candidates])])
        trace = saved["lfp"][index]
        fs = int(saved["sampling_rate_hz"])
        assert fs == 1000
        info = dict(recording=RECORDING, original_trial_number=int(original_ids[index]),
                    raw_trial_index=index, direction=int(saved["direction"][index]),
                    delay="long", sampling_rate_hz=fs,
                    source_pair_index=int(saved["source_pair_index"]),
                    source_condition_index=int(saved["source_condition_index"][index]),
                    source_trial_index=int(saved["source_trial_index"][index]),
                    selection="lowest original trial ID among long-delay trials")
    with h5py.File(UNIT / "raw_data/MonkeyT.mat", "r") as handle:
        group = handle["Monkey"]
        codes_ds = h5_cell_dataset(handle, group, "TrialCodesCorr", info["source_pair_index"])
        times_ds = h5_cell_dataset(handle, group, "TrialTimesCorr", info["source_pair_index"])
        condition, trial = info["source_condition_index"] - 1, info["source_trial_index"] - 1
        codes = trial_column(h5_cell_array(handle, codes_ds, condition), trial)
        times = trial_column(h5_cell_array(handle, times_ds, condition), trial)
        assert 92 in codes and 91 not in codes
        events = []
        for code in EVENT_CODES:
            hits = np.flatnonzero(codes == code)
            assert len(hits) == 1
            value = float(times[hits[0]])
            assert np.isfinite(value) and value.is_integer()
            events.append(int(value) - 1)
    assert np.all(np.diff(events) > 0)
    tc_on, tc_off, sc_on, sc_off, go = events
    intervals = [(tc_on - 300, tc_on), (tc_off, tc_off + 300),
                 (sc_on - 300, sc_on), (sc_off, sc_off + 300),
                 (go - 300, go), (go, go + 300)]
    assert all(intervals[i][1] <= intervals[i + 1][0] for i in range(5))
    raw, processed = [], []
    for start, end in intervals:
        window = trace[start:end]
        assert start >= 0 and end <= len(trace) and window.shape == (300,)
        assert np.isfinite(window).all() and np.ptp(window) > 1e-12
        normalized = detrend_zscore(window)
        filtered = bandpass_filter(normalized, fs=fs, low=2., high=55., order=4).astype(np.float32)
        assert np.isfinite(filtered).all() and np.ptp(filtered) > 1e-12
        raw.append(window)
        processed.append(filtered)
    info.update(event_samples_zero_based=dict(zip(EVENT_NAMES, events)),
                window_intervals_zero_based_half_open=intervals,
                raw_trial_codes=np.asarray(codes).astype(int).tolist(),
                raw_trial_times=np.asarray(times).tolist(),
                processed_windows_sha256=hashlib.sha256(np.stack(processed).tobytes()).hexdigest())
    return info, np.stack(processed)


def check_labels(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *(axis.title for axis in fig.axes)]:
        if not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1


def main():
    protected = [SOURCE, AMI_CONFIG, CORRECTED / "inputs/windows.npz",
                 CORRECTED / "features.npz", CORRECTED / "heldout_predictions.npz"]
    before = {str(path): sha256(path) for path in protected}
    palette = json.loads(AMI_CONFIG.read_text())
    info, signals = load_trial()
    trial, direction = info["original_trial_number"], info["direction"]
    clouds = {}
    for tau in TAUS:
        clouds[tau] = [lag_embed(signal, dim=3, tau=tau) for signal in signals]
        for signal, cloud in zip(signals, clouds[tau]):
            assert cloud.shape == (300 - 2 * tau, 3)
            np.testing.assert_array_equal(cloud, np.column_stack(
                (signal[2 * tau:], signal[tau:-tau], signal[:-2 * tau])))
    limit = max(float(np.max(np.abs(cloud))) for row in clouds.values() for cloud in row) * 1.08
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / f"long_delay_trial_{trial}_tau_01_to_20_geometries.pdf"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#263238", "pdf.fonttype": 42, "pdf.compression": 6})
    metadata = dict(Title=f"Monkey T long-delay trial {trial}: delay sweep 1-20 ms",
                    Subject="Observed six-stage delay trajectories; m=3; no fitting or decoding",
                    Author="NeuralFieldManifold")
    with PdfPages(path, metadata=metadata) as pdf:
        for tau in TAUS:
            fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
            fig.text(.045, .957, f"Long-delay trial {trial} | tau = {tau} ms", fontsize=17,
                     weight="bold", va="top")
            fig.text(.045, .913,
                     f"Monkey T | y070316009-12 | Channel 7 | Reach direction {direction}",
                     fontsize=10, va="top")
            fig.text(.955, .957, f"{tau} / 20", fontsize=10, ha="right", va="top")
            for stage, points in enumerate(clouds[tau]):
                position = (stage % 2) * 3 + stage // 2 + 1
                axis = fig.add_subplot(2, 3, position, projection="3d", proj_type="ortho")
                line, = axis.plot(*points.T, lw=.85, color=palette["stage_colors"][stage],
                                  solid_capstyle="round", solid_joinstyle="round", antialiased=True)
                np.testing.assert_array_equal(np.column_stack(line.get_data_3d()), points)
                axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
                axis.set_box_aspect((1, 1, 1), zoom=1.18)
                axis.view_init(elev=24, azim=-58)
                axis.set_axis_off()
                anchor = "Before onset" if stage % 2 == 0 else "After cue offset" if stage in (1, 3) else "After GO"
                axis.set_title(f"{palette['stage_labels'][stage]}\n{anchor}", fontsize=11, pad=1)
            fig.subplots_adjust(left=.03, right=.97, top=.82, bottom=.12, wspace=.02, hspace=.09)
            fig.text(.045, .055,
                     f"m = 3 | 300-ms windows | {300 - 2 * tau} observed points per stage | No fitting",
                     fontsize=9)
            fig.text(.045, .027,
                     f"Coordinates: x(t), x(t - {tau} ms), x(t - {2 * tau} ms) | "
                     f"Common limits across all pages: +/-{limit:.2f} normalized signal units",
                     fontsize=8.5)
            check_labels(fig)
            pdf.savefig(fig)
            plt.close(fig)
    with fitz.open(path) as document:
        assert len(document) == 20
        for tau, page in enumerate(document, start=1):
            text = page.get_text()
            assert f"Long-delay trial {trial}" in text and f"tau = {tau} ms" in text
            assert all(label in text for label in palette["stage_labels"])
            assert f"{300 - 2 * tau} observed points per stage" in text
            assert not page.get_images()
        document.set_toc([[1, f"tau = {tau} ms", tau] for tau in TAUS])
        document.saveIncr()
    assert before == {str(path): sha256(path) for path in protected}
    info.update(delays_ms=list(TAUS), dimension=3, pages=20, cloud_count=120,
                common_axis_limits=[-limit, limit], camera=dict(elevation=24, azimuth=-58),
                preprocessing="per-window linear detrend, median/MAD normalization, fourth-order 2-55 Hz sosfiltfilt; float32",
                preprocessing_applied_once_per_window=True, post_cues_at_offsets=True,
                fitting=False, decoding=False, pca=False, smoothing=False,
                pdf=str(path), pdf_sha256=sha256(path), protected_file_hashes=before,
                previous_results_unchanged=True, source_sha256=sha256(Path(__file__)))
    (DESTINATION / "provenance.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print(path, flush=True)
    print(f"Verified 20 vector PDF pages, 120 clouds, corrected cue-offset anchors; trial {trial} is long delay.")


if __name__ == "__main__":
    main()
