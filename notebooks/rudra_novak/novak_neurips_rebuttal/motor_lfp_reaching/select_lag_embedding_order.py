#!/usr/bin/env python
"""Select one dataset-level lag-embedding dimension from macaque LFP PSD peaks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from motor_lfp_utils import (
    CONVERTED_DIR,
    FS,
    PLOT_DIR,
    RANDOM_SEED,
    TABLE_DIR,
    detrend_zscore,
    ensure_dirs,
    extract_epoch_segments,
    write_csv,
)


FREQ_LOW = 2.0
FREQ_HIGH = 55.0
FIGURE4C_FREQ_HIGH = 200.0
LINE_FREQ = 50.0
LINE_HALF_WIDTH = 2.0
LINE_HARMONICS = (50.0, 100.0, 150.0)
PEAK_MARKER_HALF_WIDTH_HZ = 1.2
SMOOTH_SIGMA_BINS = 1.2
MIN_PEAK_DISTANCE_HZ = 3.0
PEAK_SUPPORT_TOLERANCE_HZ = 2.0
BOOTSTRAPS = 500
ROBUST_SUPPORT_THRESHOLD = 0.50
MIN_PROMINENCE = 0.02
PROMINENCE_STD_FRACTION = 0.20


def load_paths(max_lfps: int | None = None) -> list[Path]:
    paths = sorted(CONVERTED_DIR.glob("*.npz"))
    return paths[:max_lfps] if max_lfps is not None else paths


def band_label(freq_hz: float) -> str:
    if 2.0 <= freq_hz < 4.0:
        return "delta"
    if 4.0 <= freq_hz < 8.0:
        return "theta"
    if 8.0 <= freq_hz < 13.0:
        return "alpha"
    if 13.0 <= freq_hz < 30.0:
        return "beta"
    if 30.0 <= freq_hz <= 55.0:
        return "low_gamma"
    return "outside_band"


def line_mask(freqs: np.ndarray) -> np.ndarray:
    return np.abs(freqs - LINE_FREQ) <= LINE_HALF_WIDTH


def line_harmonic_mask(freqs: np.ndarray) -> np.ndarray:
    return np.any(
        np.abs(freqs[:, None] - np.asarray(LINE_HARMONICS)[None, :]) <= LINE_HALF_WIDTH,
        axis=1,
    )


def interpolate_masked_curve(freqs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return y
    keep = ~mask & np.isfinite(y)
    if np.sum(keep) < 2:
        return y
    out = y.copy()
    out[mask] = np.interp(freqs[mask], freqs[keep], y[keep])
    return out


def compute_lfp_log_psds(
    paths: list[Path],
    epoch: str,
    freq_low: float = FREQ_LOW,
    freq_high: float = FREQ_HIGH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    freq_ref: np.ndarray | None = None
    log_psds: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for path in tqdm(paths, desc="Computing LFP PSDs"):
        with np.load(path, allow_pickle=True) as data:
            segments, keep = extract_epoch_segments(data, epoch)
            if len(segments) == 0:
                continue
            trial_psds = []
            for segment in segments:
                x = detrend_zscore(segment)
                nperseg = min(1024, len(x))
                noverlap = min(nperseg // 2, max(0, nperseg - 1))
                freqs, psd = signal.welch(x, fs=FS, nperseg=nperseg, noverlap=noverlap)
                mask = (freqs >= freq_low) & (freqs <= freq_high)
                if freq_ref is None:
                    freq_ref = freqs[mask]
                trial_psds.append(psd[mask])

            log_psds.append(np.log10(np.nanmedian(np.vstack(trial_psds), axis=0) + 1e-12))
            rows.append(
                {
                    "lfp_uid": path.stem,
                    "epoch": epoch,
                    "n_trials": int(len(segments)),
                    "n_samples_per_epoch": int(segments.shape[1]),
                }
            )

    if freq_ref is None or not log_psds:
        raise RuntimeError(f"No usable {epoch} LFP epochs found in {CONVERTED_DIR}")
    return freq_ref, np.vstack(log_psds), pd.DataFrame(rows)


def remove_aperiodic_background(freqs: np.ndarray, log_psds: np.ndarray) -> np.ndarray:
    fit_mask = ~line_mask(freqs)
    x = np.vstack([np.ones(np.sum(fit_mask)), np.log10(freqs[fit_mask])]).T
    residuals = []
    for y in log_psds:
        coef = np.linalg.lstsq(x, y[fit_mask], rcond=None)[0]
        trend = coef[0] + coef[1] * np.log10(freqs)
        residuals.append(y - trend)
    return np.vstack(residuals)


def detect_peaks(
    freqs: np.ndarray,
    y: np.ndarray,
    exclude_line: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    smooth = gaussian_filter1d(y, sigma=SMOOTH_SIGMA_BINS)
    detect_mask = np.ones_like(freqs, dtype=bool)
    if exclude_line:
        detect_mask &= ~line_mask(freqs)
    detect_freqs = freqs[detect_mask]
    detect_y = smooth[detect_mask]
    df = float(np.nanmedian(np.diff(freqs))) if len(freqs) > 1 else 1.0
    min_distance_bins = max(1, int(np.ceil(MIN_PEAK_DISTANCE_HZ / df)))
    prominence = max(MIN_PROMINENCE, PROMINENCE_STD_FRACTION * float(np.nanstd(detect_y)))
    peaks_local, props = signal.find_peaks(detect_y, prominence=prominence, distance=min_distance_bins)
    return detect_freqs[peaks_local], props, smooth


def bootstrap_peak_support(freqs: np.ndarray, residuals: np.ndarray, candidate_freqs: np.ndarray) -> np.ndarray:
    if len(candidate_freqs) == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    support = np.zeros(len(candidate_freqs), dtype=float)
    for _ in tqdm(range(BOOTSTRAPS), desc="Bootstrapping PSD peaks"):
        sample_idx = rng.integers(0, residuals.shape[0], size=residuals.shape[0])
        y = np.nanmedian(residuals[sample_idx], axis=0)
        boot_freqs, _props, _smooth = detect_peaks(freqs, y, exclude_line=True)
        for i, candidate in enumerate(candidate_freqs):
            if np.any(np.abs(boot_freqs - candidate) <= PEAK_SUPPORT_TOLERANCE_HZ):
                support[i] += 1
    return support / BOOTSTRAPS


def plot_raw_psd(freqs: np.ndarray, log_psds: np.ndarray, peak_table: pd.DataFrame) -> None:
    median = np.nanmedian(log_psds, axis=0)
    q25, q75 = np.nanpercentile(log_psds, [25, 75], axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(freqs, q25, q75, color="#d8d0cc", alpha=0.55, label="LFP IQR")
    ax.plot(freqs, median, color="#8b0000", lw=2.0, label="Median LFP PSD")
    ax.axvspan(LINE_FREQ - LINE_HALF_WIDTH, LINE_FREQ + LINE_HALF_WIDTH, color="#6f6f6f", alpha=0.18, label="Excluded line band")
    for _, row in peak_table[peak_table["included_for_order"].astype(bool)].iterrows():
        ax.axvline(row["freq_hz"], color="#D85A30", lw=1.2, ls="--")
        ax.text(row["freq_hz"], ax.get_ylim()[1], f"{row['freq_hz']:.0f} Hz", ha="center", va="top", fontsize=9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log PSD")
    ax.set_title("Macaque Movement LFP PSD", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "summary" / "macaque_lfp_embedding_order_raw_psd.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_figure4c_style_psd(freqs: np.ndarray, log_psds: np.ndarray, peak_table: pd.DataFrame) -> None:
    median = 10 ** np.nanmedian(log_psds, axis=0)
    q25, q75 = 10 ** np.nanpercentile(log_psds, [25, 75], axis=0)

    artifact_mask = line_harmonic_mask(freqs)
    median = interpolate_masked_curve(freqs, median, artifact_mask)
    q25 = interpolate_masked_curve(freqs, q25, artifact_mask)
    q75 = interpolate_masked_curve(freqs, q75, artifact_mask)

    included = peak_table[peak_table["included_for_order"].astype(bool)]
    y_min = 10 ** np.floor(np.log10(np.nanmin(q25[q25 > 0])))
    y_max = 10 ** np.ceil(np.log10(np.nanmax(q75)))

    fig, ax = plt.subplots(figsize=(2.15, 1.75))
    for _, row in included.iterrows():
        ax.axvspan(
            max(FREQ_LOW, row["freq_hz"] - PEAK_MARKER_HALF_WIDTH_HZ),
            row["freq_hz"] + PEAK_MARKER_HALF_WIDTH_HZ,
            color="#b8b8b8",
            alpha=0.38,
            lw=0,
            zorder=0,
        )
    ax.fill_between(freqs, q25, q75, color="#d0d0d0", alpha=0.65, lw=0, zorder=1)
    ax.plot(freqs, median, color="#303030", lw=1.35, zorder=2)
    for _, row in included.iterrows():
        ax.plot(row["freq_hz"], median[np.argmin(np.abs(freqs - row["freq_hz"]))], "o", ms=2.5, color="#303030", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(FREQ_LOW, FIGURE4C_FREQ_HIGH)
    ax.set_ylim(y_min, y_max)
    ax.set_title("LFP PSD", fontsize=8, pad=2)
    ax.set_xlabel("Frequency (Hz)", fontsize=7, labelpad=1)
    ax.set_ylabel("PSD (a.u.)", fontsize=7, labelpad=1)
    ax.tick_params(axis="both", labelsize=6, width=0.6, length=2.4, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.tight_layout(pad=0.35)
    out = PLOT_DIR / "summary" / "macaque_lfp_embedding_order_figure4c_style.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_residual_psd(freqs: np.ndarray, residuals: np.ndarray, peak_table: pd.DataFrame) -> None:
    median = np.nanmedian(residuals, axis=0)
    q25, q75 = np.nanpercentile(residuals, [25, 75], axis=0)
    smooth = gaussian_filter1d(median, sigma=SMOOTH_SIGMA_BINS)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(freqs, q25, q75, color="#d8d0cc", alpha=0.55, label="LFP IQR")
    ax.plot(freqs, smooth, color="#8b0000", lw=2.0, label="Median residual PSD")
    ax.axhline(0, color="#201715", lw=0.8, alpha=0.45)
    ax.axvspan(LINE_FREQ - LINE_HALF_WIDTH, LINE_FREQ + LINE_HALF_WIDTH, color="#6f6f6f", alpha=0.18, label="Excluded line band")
    included = peak_table[peak_table["included_for_order"].astype(bool)]
    ax.scatter(included["freq_hz"], included["median_residual_log_psd"], color="#D85A30", s=42, zorder=3)
    for _, row in included.iterrows():
        ax.text(row["freq_hz"], row["median_residual_log_psd"] + 0.025, f"{row['freq_hz']:.0f} Hz", ha="center", fontsize=9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("1/f-corrected log PSD")
    ax.set_title("Aperiodic-Corrected PSD Peaks", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "summary" / "macaque_lfp_embedding_order_residual_psd.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_peak_support(peak_table: pd.DataFrame) -> None:
    included = peak_table[peak_table["included_for_order"].astype(bool)].copy()
    if included.empty:
        return
    labels = [f"{row.freq_hz:.0f} Hz\n{row.band}" for row in included.itertuples()]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    bars = ax.bar(np.arange(len(included)), included["bootstrap_support"], color="#8b0000", edgecolor="#4a0d08", linewidth=0.8)
    ax.axhline(ROBUST_SUPPORT_THRESHOLD, color="#201715", lw=1.0, ls="--", alpha=0.5)
    ax.set_xticks(np.arange(len(included)), labels)
    ax.set_ylabel("Bootstrap support")
    ax.set_ylim(0, 1.05)
    ax.set_title("Dataset-Level PSD Peak Support", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    for bar, value in zip(bars, included["bootstrap_support"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = PLOT_DIR / "summary" / "macaque_lfp_embedding_order_peak_support.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", default="movement")
    parser.add_argument("--max-lfps", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    paths = load_paths(max_lfps=args.max_lfps)
    freqs, log_psds, lfp_table = compute_lfp_log_psds(paths, args.epoch)
    residuals = remove_aperiodic_background(freqs, log_psds)

    median_residual = np.nanmedian(residuals, axis=0)
    candidate_freqs, props, smooth_residual = detect_peaks(freqs, median_residual, exclude_line=True)
    supports = bootstrap_peak_support(freqs, residuals, candidate_freqs)

    rows = []
    for i, freq in enumerate(candidate_freqs):
        included = bool(supports[i] >= ROBUST_SUPPORT_THRESHOLD)
        rows.append(
            {
                "freq_hz": float(freq),
                "band": band_label(float(freq)),
                "median_residual_log_psd": float(smooth_residual[np.argmin(np.abs(freqs - freq))]),
                "prominence": float(props["prominences"][i]),
                "bootstrap_support": float(supports[i]),
                "included_for_order": included,
                "exclusion_reason": "" if included else f"support below {ROBUST_SUPPORT_THRESHOLD:.2f}",
            }
        )

    raw_freqs, raw_props, raw_smooth = detect_peaks(freqs, np.nanmedian(log_psds, axis=0), exclude_line=False)
    for i, freq in enumerate(raw_freqs):
        if line_mask(np.asarray([freq]))[0]:
            rows.append(
                {
                    "freq_hz": float(freq),
                    "band": "line_noise",
                    "median_residual_log_psd": float(raw_smooth[np.argmin(np.abs(freqs - freq))]),
                    "prominence": float(raw_props["prominences"][i]),
                    "bootstrap_support": np.nan,
                    "included_for_order": False,
                    "exclusion_reason": f"within {LINE_FREQ:.0f} Hz line-noise band",
                }
            )

    peak_table = pd.DataFrame(rows).sort_values(["included_for_order", "freq_hz"], ascending=[False, True])
    robust_peaks = peak_table[peak_table["included_for_order"].astype(bool)]
    n_modes = int(len(robust_peaks))
    ar_order = 2 * n_modes
    lag_embedding_dimension = 2 * n_modes + 1

    selection = pd.DataFrame(
        [
            {
                "epoch": args.epoch,
                "n_lfps": int(len(lfp_table)),
                "frequency_range_hz": f"{FREQ_LOW:g}-{FREQ_HIGH:g}",
                "line_exclusion_hz": f"{LINE_FREQ - LINE_HALF_WIDTH:g}-{LINE_FREQ + LINE_HALF_WIDTH:g}",
                "robust_peak_count": n_modes,
                "robust_peak_frequencies_hz": ", ".join(f"{x:.1f}" for x in robust_peaks["freq_hz"]),
                "oscillatory_mode_count_K": n_modes,
                "ar_order_2K": ar_order,
                "recommended_lag_embedding_dimension_2K_plus_1": lag_embedding_dimension,
            }
        ]
    )

    write_csv(lfp_table, TABLE_DIR / "embedding_order_lfp_psd_manifest.csv")
    write_csv(peak_table, TABLE_DIR / "embedding_order_psd_peaks.csv")
    write_csv(selection, TABLE_DIR / "embedding_order_selection.csv")

    conclusion = [
        "# Macaque LFP Lag-Embedding Order Selection",
        "",
        f"Movement-aligned PSDs were aggregated across {len(lfp_table)} unique LFPs.",
        "For each LFP, trial PSDs were median-aggregated, log-transformed, and corrected for a linear 1/f background in log-frequency space.",
        f"Peaks within {LINE_FREQ - LINE_HALF_WIDTH:g}-{LINE_FREQ + LINE_HALF_WIDTH:g} Hz were excluded as line noise before counting biological oscillatory modes.",
        "",
        f"Robust biological PSD peaks were found at: {', '.join(f'{x:.1f} Hz' for x in robust_peaks['freq_hz'])}.",
        f"This gives K={n_modes} oscillatory modes, corresponding to AR order 2K={ar_order} and recommended lag-embedding dimension 2K+1={lag_embedding_dimension}.",
        "",
        f"Conclusion: use one dataset-level lag-embedding dimension of {lag_embedding_dimension} for macaque LFP analyses if the embedding order is chosen from PSD peak count.",
    ]
    (TABLE_DIR / "embedding_order_conclusion.md").write_text("\n".join(conclusion) + "\n")

    plot_raw_psd(freqs, log_psds, peak_table)
    plot_residual_psd(freqs, residuals, peak_table)
    plot_peak_support(peak_table)
    fig_freqs, fig_log_psds, _fig_lfp_table = compute_lfp_log_psds(
        paths,
        args.epoch,
        freq_low=FREQ_LOW,
        freq_high=FIGURE4C_FREQ_HIGH,
    )
    plot_figure4c_style_psd(fig_freqs, fig_log_psds, peak_table)

    print(selection.to_string(index=False))
    print(f"Wrote PSD plots under {PLOT_DIR / 'summary'}")
    print(f"Wrote {TABLE_DIR / 'embedding_order_conclusion.md'}")


if __name__ == "__main__":
    main()
