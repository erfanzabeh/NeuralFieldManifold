"""Paper-style, SC-aligned spectrograms of the frozen decoded LFP cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
RECORDING = "monkeyT_session-y070316009-12_lfp-7"
SOURCE = HERE / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2/inputs"
RAW = HERE / "lfp_converted_data" / f"{RECORDING}.npz"
OUT = HERE / "outputs/stage_T_y070316009_12_ch7_spectrogram_v1"
START_MS, END_MS = -1300, 1500
EVENTS = (("TC_on_sample", "TC"), ("SC_on_sample", "SC"), ("GO_sample", "GO"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrograms():
    trials = pd.read_csv(SOURCE / "selected_trials.csv")
    events = pd.read_csv(SOURCE / "event_samples.csv")
    assert len(trials) == len(events) == 198
    assert trials.original_trial_number.nunique() == 198
    np.testing.assert_array_equal(trials.original_trial_number, events.original_trial_number)
    np.testing.assert_array_equal(trials.groupby("direction").size(), np.full(6, 33))
    with np.load(RAW, allow_pickle=True) as saved:
        assert int(saved["sampling_rate_hz"]) == 1000
        indices = trials.raw_trial_index.to_numpy(dtype=int)
        np.testing.assert_array_equal(saved["original_trial_number"][indices], trials.original_trial_number)
        np.testing.assert_array_equal(saved["direction"][indices], trials.direction)
        assert np.all(saved["delay_label"][indices] == "short")
        lfp = saved["lfp"][indices].astype(np.float64)
    assert lfp.shape == (198, 6301) and np.isfinite(lfp).all()
    sc = events.SC_on_sample.to_numpy(dtype=int)
    starts, ends = sc + START_MS, sc + END_MS
    assert starts.min() >= 0 and ends.max() <= lfp.shape[1]
    # The paper's spectrogram uses a 2-Hz fourth-order high-pass before PSD.
    sos = signal.butter(4, 2, btype="highpass", fs=1000, output="sos")
    filtered = signal.sosfiltfilt(sos, lfp, axis=1)
    aligned = np.stack([filtered[i, a:b] for i, (a, b) in enumerate(zip(starts, ends))])
    assert aligned.shape == (198, END_MS-START_MS)
    freq, sec, power = signal.spectrogram(
        aligned, fs=1000, window="hann", nperseg=300, noverlap=250,
        nfft=1000, detrend=False, scaling="density", mode="psd", axis=1)
    assert power.shape == (198, len(freq), len(sec))
    time_from_sc = START_MS/1000 + sec
    return trials, events, freq, time_from_sc, power


def render(freq, time, power, event_positions, title, path, limits):
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    mask = (freq >= 2) & (freq <= 55)
    artist = ax.pcolormesh(time, freq[mask], 10*np.log10(np.maximum(power[mask], 1e-30)),
                           cmap="magma", vmin=limits[0], vmax=limits[1], shading="auto")
    for label, seconds in event_positions.items():
        ax.axvline(seconds, color="white", linewidth=.8, alpha=.9)
        ax.text(seconds+.015, 52, label, color="white", fontsize=8, ha="left", va="top")
    ax.set(xlim=(time[0], time[-1]), ylim=(2, 55),
           xlabel="Time from spatial cue (s)", ylabel="Frequency (Hz)", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(artist, ax=ax, label="Power spectral density (dB/Hz)", shrink=.9)
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    before = {str(p):sha256(p) for p in (RAW, SOURCE / "selected_trials.csv", SOURCE / "event_samples.csv")}
    trials, events, freq, time, power = spectrograms()
    assert np.isfinite(power).all() and (power >= 0).all()
    average = power.mean(axis=0)
    example = int(np.argmin(trials.original_trial_number.to_numpy()))
    event_positions = {label:float(np.median((events[col]-events.SC_on_sample)/1000))
                       for col, label in EVENTS}
    example_positions = {label:float((events.iloc[example][col]-events.iloc[example].SC_on_sample)/1000)
                         for col, label in EVENTS}
    both = np.concatenate((average[(freq>=2)&(freq<=55)].ravel(),
                           power[example, (freq>=2)&(freq<=55)].ravel()))
    db = 10*np.log10(np.maximum(both, 1e-30))
    limits = np.quantile(db, [.02, .995])
    render(freq, time, average, event_positions,
           "Trial-averaged spectrogram | Monkey T, channel 7 | 198 short trials",
           OUT / "trial_averaged_198.png", limits)
    render(freq, time, power[example], example_positions,
           f"Single-trial spectrogram | Original trial {int(trials.iloc[example].original_trial_number)}",
           OUT / "single_trial_example.png", limits)
    np.savez_compressed(OUT / "spectrograms.npz", frequency_hz=freq, time_from_sc_s=time,
                        power_trials=power.astype(np.float32), power_mean=average,
                        original_trial_number=trials.original_trial_number.to_numpy())
    (OUT / "provenance.json").write_text(json.dumps({
        "recording": RECORDING, "trial_count":198, "delay":"short",
        "direction_counts":trials.groupby("direction").size().to_dict(),
        "single_trial_original_number":int(trials.iloc[example].original_trial_number),
        "preprocessing":"fourth-order 2-Hz zero-phase high-pass on complete trial",
        "spectrogram":"Hann 300 ms, 50 ms shift, FFT interpolated to 1 Hz; PSD averaged in linear units across trials",
        "shared_db_color_limits":limits.tolist(), "average_event_positions_seconds":event_positions,
        "input_sha256":before, "fitting_or_decoding_performed":False}, indent=2)+"\n")
    after = {p:sha256(Path(p)) for p in before}
    assert before == after, "Protected inputs changed"
    print(OUT)


if __name__ == "__main__":
    main()
