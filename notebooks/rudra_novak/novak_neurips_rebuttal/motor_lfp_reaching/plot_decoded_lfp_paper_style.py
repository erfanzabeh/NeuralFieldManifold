"""Render the frozen 198-trial decoded cohort with the paper-example style."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from plot_paper_example_spectrograms import (
    START_MS, END_MS, EVENT_CODES, digest, panel,
)

UNIT = Path(__file__).resolve().parent
RECORDING = "monkeyT_session-y070316009-12_lfp-7"
RAW = UNIT / "lfp_converted_data" / f"{RECORDING}.npz"
INPUTS = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2/inputs"
OUT = UNIT / "outputs/stage_T_y070316009_12_ch7_paper_style_spectrogram_v1"
EVENT_COLUMNS = {"TC on":"TC_on_sample", "TC off":"TC_off_sample",
                 "SC on":"SC_on_sample", "SC off":"SC_off_sample", "GO":"GO_sample"}


def calculate():
    trials = pd.read_csv(INPUTS / "selected_trials.csv")
    events = pd.read_csv(INPUTS / "event_samples.csv")
    assert len(trials) == len(events) == 198
    assert trials.original_trial_number.nunique() == 198
    np.testing.assert_array_equal(trials.original_trial_number, events.original_trial_number)
    np.testing.assert_array_equal(trials.groupby("direction").size(), np.full(6, 33))
    assert set(trials.delay) == {"short"}
    with np.load(RAW, allow_pickle=True) as saved:
        assert int(saved["sampling_rate_hz"]) == 1000
        ids = trials.raw_trial_index.to_numpy(dtype=int)
        np.testing.assert_array_equal(saved["original_trial_number"][ids], trials.original_trial_number)
        np.testing.assert_array_equal(saved["direction"][ids], trials.direction)
        assert np.all(saved["delay_label"][ids] == "short")
        lfp = saved["lfp"][ids].astype(np.float64)
    assert lfp.shape == (198, 6301) and np.isfinite(lfp).all()
    sc = events.SC_on_sample.to_numpy(dtype=int)
    starts, ends = sc+START_MS, sc+END_MS
    assert starts.min() >= 0 and ends.max() <= lfp.shape[1]
    assert np.all(events.GO_sample == 4500)
    sos = signal.butter(4, 2, btype="highpass", fs=1000, output="sos")
    filtered = signal.sosfiltfilt(sos, lfp, axis=1)
    aligned = np.stack([filtered[i,a:b] for i,(a,b) in enumerate(zip(starts,ends))])
    freq, seconds, power = signal.spectrogram(
        aligned, fs=1000, window="hann", nperseg=300, noverlap=250,
        nfft=1000, detrend=False, scaling="density", mode="psd", axis=1)
    assert power.shape == (198, 501, 61)
    mean_power = power.mean(axis=0)
    assert np.isfinite(mean_power).all() and (mean_power >= 0).all()
    positions = {label:float(np.median((events[column]-events.SC_on_sample)/1000))
                 for label,column in EVENT_COLUMNS.items()}
    return dict(monkey="T", label="y070316009-12, LFP 7", n=198,
                freq=freq, time=START_MS/1000+seconds, mean_power=mean_power,
                event_positions=positions), trials, power


def main():
    OUT.mkdir(exist_ok=True)
    protected = [RAW, INPUTS/"selected_trials.csv", INPUTS/"event_samples.csv"]
    hashes = {str(path):digest(path) for path in protected}
    data, trials, power = calculate()
    fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
    panel(ax, data)
    fig.savefig(OUT/"decoded_channel_7_paper_style.png", dpi=600, facecolor="white")
    plt.close(fig)
    np.savez_compressed(OUT/"spectrograms.npz", frequency_hz=data["freq"],
                        time_from_sc_s=data["time"], trial_power=power.astype(np.float32),
                        mean_power=data["mean_power"],
                        original_trial_number=trials.original_trial_number.to_numpy(),
                        reach_direction=trials.direction.to_numpy())
    (OUT/"provenance.json").write_text(json.dumps({
        "recording":RECORDING, "trial_count":198, "trials_per_direction":33,
        "delay":"short", "selected_trials_source":str(INPUTS/"selected_trials.csv"),
        "method":"Identical high-pass, SC-aligned 300-ms Hann PSD/50-ms shift, linear trial averaging, and above-10-Hz color scaling as plot_paper_example_spectrograms.py",
        "paper_figure_color_style":"afmhot; color min/max above 10 Hz for this channel",
        "event_positions_seconds":data["event_positions"],
        "input_sha256":hashes, "fitting_or_decoding_performed":False,
    }, indent=2)+"\n")
    assert hashes == {str(path):digest(path) for path in protected}
    print(OUT/"decoded_channel_7_paper_style.png")


if __name__ == "__main__":
    main()
