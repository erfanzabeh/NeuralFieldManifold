from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from NeuralFieldManifold.embedders import embed

from shuffle_utils import (
    PLOTS_DIR,
    load_cached,
    load_real_trace,
    notebook_style_psd,
    welch_psd,
    zscore,
)


STEMS = [
    "fourier_phase_shuffle",
    "iaaft_shuffle",
    "aperiodic_1f",
    "envelope_phase_reset",
    "one_mode_ablation",
]

TITLES = {
    "fourier_phase_shuffle": "Fourier phase shuffle",
    "iaaft_shuffle": "IAAFT shuffle",
    "aperiodic_1f": "Aperiodic-only 1/f shuffle",
    "envelope_phase_reset": "Envelope-preserving phase reset",
    "one_mode_ablation": "One-mode ablation",
}

REAL_COLOR = "black"
SHUFFLE_COLOR = "darkred"
NULL_COLOR = "#9a9a9a"


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2: len(corr) // 2 + max_lag + 1]
    return corr / (corr[0] + 1e-12)


def slow_envelope(x: np.ndarray, fs: float, smooth_sec: float = 1.0) -> np.ndarray:
    env = np.abs(signal.hilbert(x - np.mean(x)))
    return gaussian_filter1d(env, sigma=max(1.0, smooth_sec * fs))


def common_row(axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, time: np.ndarray, fs: float) -> None:
    n_trace = min(len(real), int(10 * fs))
    axes[0].plot(time[:n_trace] - time[0], zscore(real[:n_trace]), color=REAL_COLOR, lw=0.8, label="real")
    axes[0].plot(time[:n_trace] - time[0], zscore(shuffle[:n_trace]) - 4.0, color=SHUFFLE_COLOR, lw=0.8, label="shuffle")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("z-scored trace")
    axes[0].legend(frameon=False, fontsize=8)

    freqs_real, psd_real = notebook_style_psd(real, fs)
    freqs_surr, psd_surr = notebook_style_psd(shuffle, fs)
    mask_real = (freqs_real >= 0.5) & (freqs_real <= min(120.0, fs / 2.0))
    mask_surr = (freqs_surr >= 0.5) & (freqs_surr <= min(120.0, fs / 2.0))
    axes[1].loglog(freqs_real[mask_real], psd_real[mask_real], color=REAL_COLOR, lw=1.0, label="real")
    axes[1].loglog(freqs_surr[mask_surr], psd_surr[mask_surr], color=SHUFFLE_COLOR, lw=1.0, label="shuffle")
    axes[1].axvspan(1.0, 50.0, color="0.7", alpha=0.25, zorder=0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (a.u)")
    axes[1].legend(frameon=False, fontsize=8)

    lo, hi = np.percentile(real, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    axes[2].hist(real, bins=bins, density=True, histtype="step", color=REAL_COLOR, lw=1.2, label="real")
    axes[2].hist(shuffle, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=SHUFFLE_COLOR, label="shuffle")
    axes[2].set_xlabel("Signal value")
    axes[2].set_ylabel("Density")
    axes[2].legend(frameon=False, fontsize=8)


def delayed_embedding_panel(ax: plt.Axes, real: np.ndarray, shuffle: np.ndarray, tau: int = 30) -> None:
    n = min(1200, len(real) - tau)
    xr = embed(real[: n + 2 * tau], 3, tau)[:n]
    xs = embed(shuffle[: n + 2 * tau], 3, tau)[:n]
    ax.plot(xr[:, 0], xr[:, 1], color=REAL_COLOR, lw=0.7, alpha=0.8, label="real")
    ax.plot(xs[:, 0], xs[:, 1], color=SHUFFLE_COLOR, lw=0.7, alpha=0.75, label="shuffle")
    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(r"$x(t-\tau)$")
    ax.set_title("Lag plane")
    ax.legend(frameon=False, fontsize=8)


def plot_fourier_specific(axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, fs: float) -> None:
    freqs = np.fft.rfftfreq(len(real), d=1.0 / fs)
    amp_real = np.abs(np.fft.rfft(real))
    amp_surr = np.abs(np.fft.rfft(shuffle))
    mask = (freqs >= 0.5) & (freqs <= 80.0)
    axes[0].loglog(freqs[mask], amp_real[mask], color=REAL_COLOR, lw=1.1, label="real")
    axes[0].loglog(freqs[mask], amp_surr[mask], color=SHUFFLE_COLOR, lw=1.0, ls="--", label="shuffle")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Fourier amplitude")
    axes[0].set_title("Amplitude preserved")
    axes[0].legend(frameon=False, fontsize=8)

    phase_real = np.angle(np.fft.rfft(real)[mask])
    phase_surr = np.angle(np.fft.rfft(shuffle)[mask])
    bins = np.linspace(-np.pi, np.pi, 31)
    axes[1].hist(phase_real, bins=bins, density=True, histtype="step", color=REAL_COLOR, lw=1.2, label="real")
    axes[1].hist(phase_surr, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=SHUFFLE_COLOR, label="shuffle")
    axes[1].set_xlabel("Fourier phase")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Phase reset")
    axes[1].legend(frameon=False, fontsize=8)

    delayed_embedding_panel(axes[2], real, shuffle)


def plot_iaaft_specific(axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, fs: float) -> None:
    ranks = np.linspace(0, 100, len(real))
    axes[0].plot(ranks, np.sort(real), color=REAL_COLOR, lw=1.0, label="real")
    axes[0].plot(ranks, np.sort(shuffle), color=SHUFFLE_COLOR, lw=1.0, ls="--", label="shuffle")
    axes[0].set_xlabel("Percentile")
    axes[0].set_ylabel("Signal value")
    axes[0].set_title("Distribution preserved")
    axes[0].legend(frameon=False, fontsize=8)

    freqs = np.fft.rfftfreq(len(real), d=1.0 / fs)
    amp_real = np.abs(np.fft.rfft(real))
    amp_surr = np.abs(np.fft.rfft(shuffle))
    mask = (freqs >= 0.5) & (freqs <= 80.0)
    axes[1].loglog(freqs[mask], amp_real[mask], color=REAL_COLOR, lw=1.1, label="real")
    axes[1].loglog(freqs[mask], amp_surr[mask], color=SHUFFLE_COLOR, lw=1.0, ls="--", label="shuffle")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Fourier amplitude")
    axes[1].set_title("Spectrum preserved")
    axes[1].legend(frameon=False, fontsize=8)

    delayed_embedding_panel(axes[2], real, shuffle)


def plot_aperiodic_specific(axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, fs: float) -> None:
    freqs_r, psd_r = welch_psd(real, fs)
    freqs_s, psd_s = welch_psd(shuffle, fs)
    mask_r = (freqs_r >= 1.0) & (freqs_r <= 80.0)
    mask_s = (freqs_s >= 1.0) & (freqs_s <= 80.0)
    axes[0].loglog(freqs_r[mask_r], gaussian_filter1d(psd_r[mask_r], 1), color=REAL_COLOR, lw=1.0, label="real")
    axes[0].loglog(freqs_s[mask_s], gaussian_filter1d(psd_s[mask_s], 1), color=SHUFFLE_COLOR, lw=1.0, label="1/f shuffle")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title("No narrow peaks")
    axes[0].legend(frameon=False, fontsize=8)

    lag_sec = np.arange(0, int(2.0 * fs) + 1) / fs
    ac_r = autocorr(real, len(lag_sec) - 1)
    ac_s = autocorr(shuffle, len(lag_sec) - 1)
    axes[1].plot(lag_sec, ac_r, color=REAL_COLOR, lw=1.0, label="real")
    axes[1].plot(lag_sec, ac_s, color=SHUFFLE_COLOR, lw=1.0, label="1/f shuffle")
    axes[1].axhline(0.0, color="0.8", lw=0.8)
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].set_title("Only broad memory")
    axes[1].legend(frameon=False, fontsize=8)

    delayed_embedding_panel(axes[2], real, shuffle)


def plot_envelope_specific(axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, time: np.ndarray, fs: float) -> None:
    env_r = slow_envelope(real, fs)
    env_s = slow_envelope(shuffle, fs)
    n = min(len(real), int(20 * fs))
    axes[0].plot(time[:n], zscore(env_r[:n]), color=REAL_COLOR, lw=1.0, label="real")
    axes[0].plot(time[:n], zscore(env_s[:n]), color=SHUFFLE_COLOR, lw=1.0, label="shuffle")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("z-scored envelope")
    axes[0].set_title("Slow envelope kept")
    axes[0].legend(frameon=False, fontsize=8)

    lag_sec = np.arange(0, int(5.0 * fs) + 1) / fs
    ac_r = autocorr(env_r, len(lag_sec) - 1)
    ac_s = autocorr(env_s, len(lag_sec) - 1)
    axes[1].plot(lag_sec, ac_r, color=REAL_COLOR, lw=1.0, label="real")
    axes[1].plot(lag_sec, ac_s, color=SHUFFLE_COLOR, lw=1.0, label="shuffle")
    axes[1].axhline(0.0, color="0.8", lw=0.8)
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Envelope autocorr.")
    axes[1].set_title("Envelope timescale")
    axes[1].legend(frameon=False, fontsize=8)

    delayed_embedding_panel(axes[2], real, shuffle)


def plot_ablation_specific(axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, fs: float, metadata: dict) -> None:
    if "ablation_band_hz" in metadata:
        band_low, band_high = [float(v) for v in metadata["ablation_band_hz"]]
    else:
        target = float(metadata.get("target_frequency_hz", 0.0))
        bw = float(metadata.get("bandwidth_hz", 2.0))
        band_low, band_high = target - bw, target + bw
    freqs, psd_r = notebook_style_psd(real, fs)
    _, psd_s = notebook_style_psd(shuffle, fs)
    mask = (freqs >= max(0.5, band_low - 6.0)) & (freqs <= band_high + 8.0)
    axes[0].semilogy(freqs[mask], gaussian_filter1d(psd_r[mask], 1), color=REAL_COLOR, lw=1.1, label="real")
    axes[0].semilogy(freqs[mask], gaussian_filter1d(psd_s[mask], 1), color=SHUFFLE_COLOR, lw=1.1, label="shuffle")
    axes[0].axvspan(band_low, band_high, color="0.7", alpha=0.35, zorder=0)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title(f"Removed {band_low:g}-{band_high:g} Hz mode")
    axes[0].legend(frameon=False, fontsize=8)

    ratio = np.log10((psd_s + 1e-18) / (psd_r + 1e-18))
    mask_full = (freqs >= 0.5) & (freqs <= 80.0)
    axes[1].plot(freqs[mask_full], gaussian_filter1d(ratio[mask_full], 1), color=SHUFFLE_COLOR, lw=1.0)
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].axvspan(band_low, band_high, color="0.7", alpha=0.35, zorder=0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel(r"$\log_{10}$ shuffle / real")
    axes[1].set_title("Localized spectral change")

    delayed_embedding_panel(axes[2], real, shuffle)


def specific_row(stem: str, axes: np.ndarray, real: np.ndarray, shuffle: np.ndarray, time: np.ndarray, fs: float, metadata: dict) -> None:
    if stem == "fourier_phase_shuffle":
        plot_fourier_specific(axes, real, shuffle, fs)
    elif stem == "iaaft_shuffle":
        plot_iaaft_specific(axes, real, shuffle, fs)
    elif stem == "aperiodic_1f":
        plot_aperiodic_specific(axes, real, shuffle, fs)
    elif stem == "envelope_phase_reset":
        plot_envelope_specific(axes, real, shuffle, time, fs)
    elif stem == "one_mode_ablation":
        plot_ablation_specific(axes, real, shuffle, fs, metadata)
    else:
        raise ValueError(f"Unknown stem: {stem}")


def plot_one(stem: str) -> None:
    cached = load_cached(stem)
    if cached is None:
        raise FileNotFoundError(f"Missing or stale cache for {stem}; run {stem}.py --recompute first")
    shuffle, time, fs, metadata = cached
    real, _, _ = load_real_trace()

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    fig.suptitle(TITLES[stem], fontsize=13, fontweight="bold")
    common_row(axes[0], real, shuffle, time, fs)
    specific_row(stem, axes[1], real, shuffle, time, fs, metadata)
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{stem}_diagnostic.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        plot_one(stem)
        print(PLOTS_DIR / f"{stem}_diagnostic.png")


if __name__ == "__main__":
    main()
