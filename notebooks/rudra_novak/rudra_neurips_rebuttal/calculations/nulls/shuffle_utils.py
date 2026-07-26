from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


UNIT_DIR = Path(__file__).resolve().parent
REBUTTAL_DIR = UNIT_DIR.parent.parent
DATA_PATH = REBUTTAL_DIR / "data" / "monkey_lfp.npz"
REAL_TRACE_ID = "monkey_lfp_chan1001_20s_torusfit_envnorm"
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--recompute", action="store_true")
    return parser


def load_real_trace() -> tuple[np.ndarray, np.ndarray, float]:
    data = np.load(DATA_PATH)
    x = np.asarray(data["xs"], dtype=np.float64)
    fs = float(np.asarray(data["Fs"]).item())
    if "time" in data.files:
        time = np.asarray(data["time"], dtype=np.float64)
    else:
        time = np.arange(len(x), dtype=np.float64) / fs
    return x, time, fs


def match_mean_std(y: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    scale = np.std(y)
    if scale > 0:
        y = y / scale
    return y * np.std(x_ref) + np.mean(x_ref)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def rfft_phase_shuffle(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(x)
    amplitudes = np.abs(spectrum)
    phases = np.angle(spectrum)
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    random_phases[0] = phases[0]
    if len(random_phases) > 1 and len(x) % 2 == 0:
        random_phases[-1] = phases[-1]
    shuffled = amplitudes * np.exp(1j * random_phases)
    return np.fft.irfft(shuffled, n=len(x))


def notebook_style_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD settings used by the MonkeyData notebook's LFP PSD panel."""
    from mne.time_frequency import psd_array_welch

    n_fft = min(4096, len(x))
    n_overlap = min(2048, n_fft // 2)
    psd, freqs = psd_array_welch(
        np.asarray(x, dtype=np.float64)[None, :],
        sfreq=fs,
        fmin=0.5,
        fmax=min(200.0, fs / 2.0),
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    return freqs, gaussian_filter1d(psd[0], sigma=2)


def welch_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(8192, len(x))
    noverlap = nperseg // 2
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd


def attach_real_trace_metadata(metadata: dict[str, Any], n_samples: int, fs: float) -> dict[str, Any]:
    out = dict(metadata)
    out["real_trace_id"] = REAL_TRACE_ID
    out["real_data_path"] = str(DATA_PATH)
    out["real_n_samples"] = int(n_samples)
    out["real_fs"] = float(fs)
    return out


def save_shuffle(stem: str, x: np.ndarray, time: np.ndarray, fs: float, metadata: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{stem}.npz"
    np.savez_compressed(path, xs=np.asarray(x), time=np.asarray(time), Fs=np.asarray(fs), metadata=json.dumps(metadata, sort_keys=True))
    return path


def load_cached(stem: str) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]] | None:
    path = CACHE_DIR / f"{stem}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    if metadata.get("real_trace_id") != REAL_TRACE_ID:
        return None
    return np.asarray(data["xs"]), np.asarray(data["time"]), float(np.asarray(data["Fs"]).item()), metadata


def plot_diagnostic(stem: str, real: np.ndarray, shuffle: np.ndarray, time: np.ndarray, fs: float, title: str, metadata: dict[str, Any]) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n_trace = min(len(real), int(10 * fs))
    freqs_real, psd_real = notebook_style_psd(real, fs)
    freqs_surr, psd_surr = notebook_style_psd(shuffle, fs)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].plot(time[:n_trace] - time[0], zscore(real[:n_trace]), color="black", lw=0.8, label="real")
    axes[0].plot(time[:n_trace] - time[0], zscore(shuffle[:n_trace]) - 4.0, color="darkred", lw=0.8, label="shuffle")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("z-scored trace")
    axes[0].legend(frameon=False, fontsize=8)

    mask_real = (freqs_real >= 0.5) & (freqs_real <= min(120.0, fs / 2))
    mask_surr = (freqs_surr >= 0.5) & (freqs_surr <= min(120.0, fs / 2))
    axes[1].loglog(freqs_real[mask_real], psd_real[mask_real], color="black", lw=1.0, label="real")
    axes[1].loglog(freqs_surr[mask_surr], psd_surr[mask_surr], color="darkred", lw=1.0, label="shuffle")
    axes[1].axvspan(1.0, 50.0, color="0.7", alpha=0.25, zorder=0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (a.u)")
    axes[1].legend(frameon=False, fontsize=8)

    lo, hi = np.percentile(real, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    axes[2].hist(real, bins=bins, density=True, histtype="step", color="black", lw=1.2, label="real")
    axes[2].hist(shuffle, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color="darkred", label="shuffle")
    axes[2].set_xlabel("Signal value")
    axes[2].set_ylabel("Density")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    meta_text = ", ".join(f"{k}={v}" for k, v in metadata.items() if k in {"seed", "alpha", "iterations", "target_frequency_hz", "bandwidth_hz"})
    if meta_text:
        fig.text(0.01, 0.01, meta_text, ha="left", va="bottom", fontsize=7)
    fig.tight_layout()
    png_path = PLOTS_DIR / f"{stem}_diagnostic.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)
    return png_path


def run_or_load(stem: str, recompute: bool, compute_fn):
    cached = None if recompute else load_cached(stem)
    if cached is not None:
        return cached
    x, time, fs = load_real_trace()
    shuffle, metadata = compute_fn(x, time, fs)
    metadata = attach_real_trace_metadata(metadata, len(x), fs)
    save_shuffle(stem, shuffle, time, fs, metadata)
    return shuffle, time, fs, metadata
