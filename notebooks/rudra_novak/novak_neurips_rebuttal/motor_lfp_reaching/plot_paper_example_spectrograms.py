"""Recreate the two published example-LFP spectrograms from their saved trials."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from convert_motor_lfp import h5_cell_array, h5_cell_dataset, trial_column

UNIT = Path(__file__).resolve().parent
OUT = UNIT / "outputs/paper_example_lfp_spectrograms_v1"
EXAMPLES = (
    ("T", "monkeyT_session-y070315015-17_lfp-6", "y070315015-17, LFP 6"),
    ("M", "monkeyM_session-o080520001-6_lfp-2", "o080520001-6, LFP 2"),
)
START_MS, END_MS = -1600, 1700
EVENT_CODES = {"TC on":202, "TC off":203, "SC on":204, "SC off":206, "GO":207}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_example(monkey, name):
    converted = UNIT / "lfp_converted_data" / f"{name}.npz"
    raw = UNIT / "raw_data" / f"Monkey{monkey}.mat"
    with np.load(converted, allow_pickle=True) as z, h5py.File(raw, "r") as handle:
        assert int(z["sampling_rate_hz"]) == 1000
        selected = np.flatnonzero(z["delay_label"] == "short")
        lfp = np.asarray(z["lfp"][selected], dtype=np.float64)
        assert lfp.ndim == 2 and lfp.shape[0] == len(selected) and np.isfinite(lfp).all()
        pair = int(z["source_pair_index"])
        group = handle["Monkey"]
        codes_dataset = h5_cell_dataset(handle, group, "TrialCodesCorr", pair)
        times_dataset = h5_cell_dataset(handle, group, "TrialTimesCorr", pair)
        cache, event_rows = {}, []
        for i in selected:
            ci = int(z["source_condition_index"][i])-1
            ti = int(z["source_trial_index"][i])-1
            if ci not in cache:
                cache[ci] = (h5_cell_array(handle, codes_dataset, ci),
                             h5_cell_array(handle, times_dataset, ci))
            codes, times = (trial_column(a, ti) for a in cache[ci])
            assert 91 in codes and 92 not in codes
            events = {}
            for label, code in EVENT_CODES.items():
                hit = np.flatnonzero(codes == code)
                assert len(hit) == 1
                value = times[hit[0]]
                assert np.isfinite(value) and float(value).is_integer()
                events[label] = int(value)-1
            assert list(events.values()) == sorted(events.values())
            event_rows.append(events)
        numbers = z["original_trial_number"][selected].astype(int)
        directions = z["direction"][selected].astype(int)
    sc = np.array([row["SC on"] for row in event_rows])
    starts = sc + START_MS
    ends = sc + END_MS
    assert starts.min() >= 0 and ends.max() <= lfp.shape[1]
    sos = signal.butter(4, 2, btype="highpass", fs=1000, output="sos")
    filtered = signal.sosfiltfilt(sos, lfp, axis=1)
    aligned = np.stack([filtered[i, a:b] for i, (a,b) in enumerate(zip(starts, ends))])
    freq, seconds, power = signal.spectrogram(
        aligned, fs=1000, window="hann", nperseg=300, noverlap=250,
        nfft=1000, detrend=False, scaling="density", mode="psd", axis=1)
    assert power.shape == (len(selected), len(freq), len(seconds))
    average = power.mean(axis=0)
    positions = {label:float(np.median([(row[label]-row["SC on"])/1000 for row in event_rows]))
                 for label in EVENT_CODES}
    return dict(monkey=monkey, name=name, label=name.split("_session-")[1].replace("_lfp-", ", LFP "),
                n=len(selected), numbers=numbers,
                directions=directions, freq=freq, time=START_MS/1000+seconds,
                mean_power=average, event_positions=positions,
                raw_go_range=[min(row["GO"] for row in event_rows),
                              max(row["GO"] for row in event_rows)],
                input_hashes={str(converted):digest(converted), str(raw):digest(raw)})


def panel(ax, data):
    freq = data["freq"]
    shown = (freq >= 2) & (freq <= 50)
    reference = data["mean_power"][(freq >= 10) & (freq <= 50)]
    lo, hi = float(reference.min()), float(reference.max())
    normalized = np.clip((data["mean_power"][shown]-lo)/(hi-lo), 0, 1)
    ax.pcolormesh(data["time"], freq[shown], normalized,
                  cmap="afmhot", vmin=0, vmax=1, shading="auto", rasterized=True)
    for label in EVENT_CODES:
        ax.axvline(data["event_positions"][label], color="#E6E6E6", linewidth=.7)
    ax.set(xlim=(-1.45, 1.55), ylim=(2, 50),
           xlabel="Time from SC (s)", ylabel="Frequency (Hz)",
           title=f"Monkey {data['monkey']} | {data['label']}\n{data['n']} correct short-delay trials")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    OUT.mkdir(exist_ok=True)
    data = [load_example(monkey, name) for monkey, name, _ in EXAMPLES]
    assert data[0]["n"] == 74 and data[1]["n"] == 51
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), layout="constrained")
    for ax, item in zip(axes, data):
        panel(ax, item)
    fig.savefig(OUT / "paper_examples_T_and_M.png", dpi=600, facecolor="white")
    plt.close(fig)
    for item in data:
        fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
        panel(ax, item)
        fig.savefig(OUT / f"paper_example_monkey_{item['monkey']}.png", dpi=600, facecolor="white")
        plt.close(fig)
        np.savez_compressed(OUT / f"paper_example_monkey_{item['monkey']}.npz",
                            frequency_hz=item["freq"], time_from_sc_s=item["time"],
                            mean_power=item["mean_power"], original_trial_number=item["numbers"],
                            direction=item["directions"])
    metadata = {d["monkey"]:{k:v for k,v in d.items()
                            if k not in ("freq", "time", "mean_power", "numbers", "directions")}
                for d in data}
    metadata["method"] = ("All converted correct short-delay trials for each named LFP; "
        "per-trial alignment to raw SC-on event; fourth-order 2-Hz zero-phase high-pass; "
        "300-ms Hann PSD, 50-ms shift, zero-padded 1-Hz frequency grid; "
        "linear power averaged across trials; color limits min/max power above 10 Hz "
        "per monkey, matching the paper's description. Exact source window taper and "
        "colormap were not specified, so the image is a methodological recreation.")
    (OUT / "provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    for item in data:
        for path, expected in item["input_hashes"].items():
            assert digest(Path(path)) == expected
    print(OUT)


if __name__ == "__main__":
    main()
