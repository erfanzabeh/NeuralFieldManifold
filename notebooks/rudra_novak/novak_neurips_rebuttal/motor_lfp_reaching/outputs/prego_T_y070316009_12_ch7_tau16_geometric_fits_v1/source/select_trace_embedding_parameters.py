#!/usr/bin/env python
"""Select per-LFP tau and lag-embedding dimension for macaque LFP torus features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from motor_lfp_utils import (
    CONVERTED_DIR,
    FS,
    PLOT_DIR,
    RANDOM_SEED,
    TABLE_DIR,
    bandpass_filter,
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

TAU_MIN = 1
TAU_MAX = 100
AMI_BINS = 48
AMI_MAX_PAIRS = 80_000
ACF_MIN_LAG = 2

SMOOTH_SIGMA_BINS = 1.2
MIN_PEAK_DISTANCE_HZ = 3.0
PEAK_SUPPORT_TOLERANCE_HZ = 2.0
BOOTSTRAPS = 120
ROBUST_SUPPORT_THRESHOLD = 0.50
MIN_PROMINENCE = 0.02
PROMINENCE_STD_FRACTION = 0.20
MIN_EMBED_POINTS = 300
MAX_OSCILLATORY_MODES = 4

OUTPUT_SUFFIX = "pertrace_tau_dim"


def load_paths(max_lfps: int | None = None, monkeys: list[str] | None = None) -> list[Path]:
    paths = sorted(CONVERTED_DIR.glob("*.npz"))
    if monkeys:
        wanted = {f"monkey{monkey}" for monkey in monkeys}
        paths = [path for path in paths if any(path.stem.startswith(prefix) for prefix in wanted)]
    return paths[:max_lfps] if max_lfps is not None else paths


def suffix_token(suffix: str | None) -> str:
    clean = str(suffix or "").strip().strip("_")
    return f"_{clean}" if clean else ""


def table_path(name: str, suffix: str | None = OUTPUT_SUFFIX) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def plot_path(name: str, suffix: str | None = OUTPUT_SUFFIX) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


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


def preprocess_segments(segments: np.ndarray) -> np.ndarray:
    processed = []
    for segment in segments:
        x = detrend_zscore(segment)
        try:
            x = bandpass_filter(x, fs=FS, low=FREQ_LOW, high=FREQ_HIGH)
        except ValueError:
            pass
        processed.append(x.astype(np.float32, copy=False))
    if not processed:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(processed)


def average_mutual_information(
    segments: np.ndarray,
    lags: np.ndarray,
    n_bins: int,
    max_pairs: int,
    seed: int,
) -> np.ndarray:
    pooled = segments.ravel()
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 100:
        return np.full(len(lags), np.nan, dtype=float)
    low, high = np.nanpercentile(pooled, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(len(lags), np.nan, dtype=float)
    bins = np.linspace(low, high, n_bins + 1)
    rng = np.random.default_rng(seed)
    ami = np.full(len(lags), np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag >= segments.shape[1]:
            continue
        x = segments[:, :-lag].reshape(-1)
        y = segments[:, lag:].reshape(-1)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) > max_pairs:
            idx = rng.choice(len(x), size=max_pairs, replace=False)
            x = x[idx]
            y = y[idx]
        hist, _x_edges, _y_edges = np.histogram2d(x, y, bins=(bins, bins))
        total = float(hist.sum())
        if total <= 0:
            continue
        pxy = hist / total
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        denom = px[:, None] * py[None, :]
        nz = (pxy > 0) & (denom > 0)
        ami[i] = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    return ami


def median_autocorrelation(segments: np.ndarray, lags: np.ndarray) -> np.ndarray:
    acf = np.full(len(lags), np.nan, dtype=float)
    for i, lag in enumerate(lags):
        vals = []
        for segment in segments:
            x = np.asarray(segment, dtype=float)
            if lag >= len(x):
                continue
            a = x[:-lag]
            b = x[lag:]
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if denom > 1e-12:
                vals.append(float(np.dot(a, b) / denom))
        if vals:
            acf[i] = float(np.nanmedian(vals))
    return acf


def first_local_minimum(values: np.ndarray, lags: np.ndarray, min_lag: int) -> int | None:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return None
    smooth = gaussian_filter1d(np.where(finite, values, np.nanmedian(values[finite])), sigma=1.0)
    for i in range(1, len(smooth) - 1):
        if lags[i] < min_lag:
            continue
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            return int(lags[i])
    return None


def select_tau(segments: np.ndarray, max_tau: int, seed: int) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray]:
    lags = np.arange(TAU_MIN, max_tau + 1, dtype=int)
    ami = average_mutual_information(segments, lags, n_bins=AMI_BINS, max_pairs=AMI_MAX_PAIRS, seed=seed)
    tau = first_local_minimum(ami, lags, min_lag=TAU_MIN + 1)
    if tau is not None:
        return tau, "ami_first_local_minimum", lags, ami, median_autocorrelation(segments, lags)

    finite_ami = ami[np.isfinite(ami)]
    if finite_ami.size:
        threshold = float(finite_ami[0] / np.e)
        below = lags[np.isfinite(ami) & (ami <= threshold)]
        if len(below):
            return int(below[0]), "ami_one_over_e_fallback", lags, ami, median_autocorrelation(segments, lags)

    acf = median_autocorrelation(segments, lags)
    finite_acf = np.isfinite(acf)
    zero = lags[finite_acf & (acf <= 0)]
    if len(zero):
        return int(zero[0]), "acf_first_zero_fallback", lags, ami, acf
    tau = first_local_minimum(acf, lags, min_lag=ACF_MIN_LAG)
    if tau is not None:
        return tau, "acf_first_local_minimum_fallback", lags, ami, acf
    if finite_ami.size:
        return int(lags[np.nanargmin(ami)]), "ami_global_minimum_fallback", lags, ami, acf
    return 20, "default_20ms_fallback", lags, ami, acf


def compute_log_psd(segments: np.ndarray, freq_high: float) -> tuple[np.ndarray, np.ndarray]:
    freq_ref: np.ndarray | None = None
    trial_psds = []
    for segment in segments:
        x = detrend_zscore(segment)
        nperseg = min(1024, len(x))
        noverlap = min(nperseg // 2, max(0, nperseg - 1))
        freqs, psd = signal.welch(x, fs=FS, nperseg=nperseg, noverlap=noverlap)
        mask = (freqs >= FREQ_LOW) & (freqs <= freq_high)
        if freq_ref is None:
            freq_ref = freqs[mask]
        trial_psds.append(psd[mask])
    if freq_ref is None or not trial_psds:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    return freq_ref, np.log10(np.vstack(trial_psds) + 1e-12)


def remove_aperiodic_background(freqs: np.ndarray, log_psds: np.ndarray) -> np.ndarray:
    fit_mask = ~line_mask(freqs)
    if np.sum(fit_mask) < 3:
        return log_psds - np.nanmedian(log_psds, axis=1, keepdims=True)
    x = np.vstack([np.ones(np.sum(fit_mask)), np.log10(freqs[fit_mask])]).T
    residuals = []
    for y in log_psds:
        coef = np.linalg.lstsq(x, y[fit_mask], rcond=None)[0]
        trend = coef[0] + coef[1] * np.log10(freqs)
        residuals.append(y - trend)
    return np.vstack(residuals)


def detect_peaks(freqs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    smooth = gaussian_filter1d(y, sigma=SMOOTH_SIGMA_BINS)
    detect_mask = ~line_mask(freqs)
    detect_freqs = freqs[detect_mask]
    detect_y = smooth[detect_mask]
    df = float(np.nanmedian(np.diff(freqs))) if len(freqs) > 1 else 1.0
    min_distance_bins = max(1, int(np.ceil(MIN_PEAK_DISTANCE_HZ / df)))
    prominence = max(MIN_PROMINENCE, PROMINENCE_STD_FRACTION * float(np.nanstd(detect_y)))
    peaks_local, props = signal.find_peaks(detect_y, prominence=prominence, distance=min_distance_bins)
    return detect_freqs[peaks_local], props, smooth


def bootstrap_peak_support(freqs: np.ndarray, residuals: np.ndarray, candidate_freqs: np.ndarray, seed: int, n_bootstraps: int) -> np.ndarray:
    if len(candidate_freqs) == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    support = np.zeros(len(candidate_freqs), dtype=float)
    for _ in range(n_bootstraps):
        sample_idx = rng.integers(0, residuals.shape[0], size=residuals.shape[0])
        y = np.nanmedian(residuals[sample_idx], axis=0)
        boot_freqs, _props, _smooth = detect_peaks(freqs, y)
        for i, candidate in enumerate(candidate_freqs):
            if np.any(np.abs(boot_freqs - candidate) <= PEAK_SUPPORT_TOLERANCE_HZ):
                support[i] += 1
    return support / max(1, n_bootstraps)


def adjust_embedding_for_epoch_length(dim: int, tau: int, segment_len: int, min_points: int) -> tuple[int, int, str]:
    notes = []
    dim = int(dim)
    tau = int(tau)
    while dim > 3 and segment_len - (dim - 1) * tau < min_points:
        dim -= 2
        notes.append("reduced_dim_for_min_points")
    if segment_len - (dim - 1) * tau < min_points:
        max_tau = max(1, (segment_len - min_points) // max(1, dim - 1))
        if tau > max_tau:
            tau = max_tau
            notes.append("reduced_tau_for_min_points")
    return dim, tau, ";".join(notes)


def select_embedding_dim(robust_count: int) -> tuple[int, int, str]:
    if robust_count <= 0:
        return 1, 3, "minimum_3d_no_robust_psd_peak"
    modes = min(int(robust_count), MAX_OSCILLATORY_MODES)
    reason = "psd_peak_count"
    if robust_count > MAX_OSCILLATORY_MODES:
        reason = f"capped_at_{MAX_OSCILLATORY_MODES}_modes"
    return modes, 2 * modes + 1, reason


def process_lfp(path: Path, epoch: str, max_tau: int, n_bootstraps: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        segments, keep = extract_epoch_segments(data, epoch)
        monkey = str(data["monkey"])
        session_id = str(data["session_id"])
        lfp_id = str(data["lfp_id"])

    if len(segments) == 0:
        row = {
            "lfp_uid": path.stem,
            "monkey": monkey,
            "session_id": session_id,
            "lfp_id": lfp_id,
            "epoch_for_selection": epoch,
            "status": "no_valid_segments",
            "n_trials_used": 0,
            "n_samples_per_epoch": 0,
            "torus_tau": 20,
            "torus_tau_ms": 20.0,
            "tau_method": "default_no_segments",
            "robust_peak_count": 0,
            "selected_mode_count_K": 1,
            "torus_embedding_dim": 3,
            "embedding_dim_method": "minimum_3d_no_segments",
            "included_peak_frequencies_hz": "",
            "dominant_peak_hz": np.nan,
            "embedding_adjustment": "",
            "torus_param_source": "pertrace_unsupervised_movement_signal",
            "torus_param_id": "pertrace_tau20_embed3",
        }
        return row, [], {}

    processed = preprocess_segments(segments)
    tau, tau_method, lags, ami, acf = select_tau(processed, max_tau=max_tau, seed=seed)

    freqs, log_psds = compute_log_psd(segments, FREQ_HIGH)
    residuals = remove_aperiodic_background(freqs, log_psds)
    median_residual = np.nanmedian(residuals, axis=0)
    candidate_freqs, props, smooth_residual = detect_peaks(freqs, median_residual)
    supports = bootstrap_peak_support(freqs, residuals, candidate_freqs, seed=seed + 17, n_bootstraps=n_bootstraps)

    peak_rows: list[dict[str, object]] = []
    included_freqs = []
    for i, freq in enumerate(candidate_freqs):
        included = bool(supports[i] >= ROBUST_SUPPORT_THRESHOLD)
        if included:
            included_freqs.append(float(freq))
        peak_rows.append(
            {
                "lfp_uid": path.stem,
                "monkey": monkey,
                "session_id": session_id,
                "lfp_id": lfp_id,
                "freq_hz": float(freq),
                "band": band_label(float(freq)),
                "median_residual_log_psd": float(smooth_residual[np.argmin(np.abs(freqs - freq))]),
                "prominence": float(props["prominences"][i]),
                "bootstrap_support": float(supports[i]),
                "included_for_order": included,
                "exclusion_reason": "" if included else f"support below {ROBUST_SUPPORT_THRESHOLD:.2f}",
            }
        )

    robust_count = len(included_freqs)
    selected_modes, embedding_dim, dim_method = select_embedding_dim(robust_count)
    embedding_dim, tau, adjustment = adjust_embedding_for_epoch_length(
        dim=embedding_dim,
        tau=tau,
        segment_len=int(segments.shape[1]),
        min_points=MIN_EMBED_POINTS,
    )
    if adjustment:
        dim_method = f"{dim_method};{adjustment}"

    dominant_peak = float(included_freqs[0]) if included_freqs else np.nan
    row = {
        "lfp_uid": path.stem,
        "monkey": monkey,
        "session_id": session_id,
        "lfp_id": lfp_id,
        "epoch_for_selection": epoch,
        "status": "ok",
        "n_trials_used": int(len(segments)),
        "n_samples_per_epoch": int(segments.shape[1]),
        "torus_tau": int(tau),
        "torus_tau_ms": float(tau * 1000.0 / FS),
        "tau_method": tau_method,
        "robust_peak_count": int(robust_count),
        "selected_mode_count_K": int(selected_modes),
        "torus_embedding_dim": int(embedding_dim),
        "embedding_dim_method": dim_method,
        "included_peak_frequencies_hz": ", ".join(f"{freq:.1f}" for freq in included_freqs),
        "dominant_peak_hz": dominant_peak,
        "embedding_adjustment": adjustment,
        "torus_param_source": "pertrace_unsupervised_movement_signal",
        "torus_param_id": f"pertrace_tau{int(tau)}_embed{int(embedding_dim)}",
    }

    fig_freqs, fig_log_psds = compute_log_psd(segments, FIGURE4C_FREQ_HIGH)
    diagnostics = {
        "lfp_uid": path.stem,
        "freqs": fig_freqs,
        "median_log_psd": np.nanmedian(fig_log_psds, axis=0) if len(fig_log_psds) else np.array([], dtype=float),
        "included_freqs": np.asarray(included_freqs, dtype=float),
        "lags": lags,
        "ami": ami,
        "acf": acf,
    }
    return row, peak_rows, diagnostics


def plot_parameter_summary(params: pd.DataFrame, suffix: str | None) -> None:
    ok = params[params["status"] == "ok"].copy()
    if ok.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))

    axes[0, 0].hist(ok["torus_tau_ms"].astype(float), bins=24, color="#8b0000", alpha=0.85, edgecolor="white")
    axes[0, 0].set_xlabel("Selected tau (ms)")
    axes[0, 0].set_ylabel("LFPs")
    axes[0, 0].set_title("Per-LFP Tau")

    dim_counts = ok["torus_embedding_dim"].astype(int).value_counts().sort_index()
    axes[0, 1].bar(dim_counts.index.astype(str), dim_counts.values, color="#D85A30", edgecolor="#6f1009")
    axes[0, 1].set_xlabel("Embedding dimension")
    axes[0, 1].set_ylabel("LFPs")
    axes[0, 1].set_title("Per-LFP Embedding Dimension")

    for monkey, sub in ok.groupby("monkey", sort=True):
        axes[1, 0].scatter(
            sub["dominant_peak_hz"].astype(float),
            sub["torus_tau_ms"].astype(float),
            s=18,
            alpha=0.65,
            label=f"Monkey {monkey}",
        )
    axes[1, 0].set_xlabel("First included PSD peak (Hz)")
    axes[1, 0].set_ylabel("Selected tau (ms)")
    axes[1, 0].set_title("Tau vs PSD Peak")
    axes[1, 0].legend(frameon=False, fontsize=8)

    peak_counts = ok["robust_peak_count"].astype(int).value_counts().sort_index()
    axes[1, 1].bar(peak_counts.index.astype(str), peak_counts.values, color="#6f6f6f", edgecolor="#201715")
    axes[1, 1].set_xlabel("Robust PSD peak count")
    axes[1, 1].set_ylabel("LFPs")
    axes[1, 1].set_title("Detected Oscillatory Modes")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)
    fig.suptitle("Per-Trace Lag-Embedding Parameter Selection", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path("lfp_embedding_parameter_summary.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_figure4c_style_psd(diagnostics: list[dict[str, Any]], suffix: str | None) -> None:
    usable = [diag for diag in diagnostics if len(diag.get("freqs", [])) and len(diag.get("median_log_psd", []))]
    if not usable:
        return
    freqs = usable[0]["freqs"]
    log_psds = np.vstack([diag["median_log_psd"] for diag in usable if len(diag["median_log_psd"]) == len(freqs)])
    if len(log_psds) == 0:
        return
    median = 10 ** np.nanmedian(log_psds, axis=0)
    q25, q75 = 10 ** np.nanpercentile(log_psds, [25, 75], axis=0)
    artifact_mask = line_harmonic_mask(freqs)
    median = interpolate_masked_curve(freqs, median, artifact_mask)
    q25 = interpolate_masked_curve(freqs, q25, artifact_mask)
    q75 = interpolate_masked_curve(freqs, q75, artifact_mask)

    all_peaks = np.concatenate([diag["included_freqs"] for diag in usable if len(diag["included_freqs"])])
    y_min = 10 ** np.floor(np.log10(np.nanmin(q25[q25 > 0])))
    y_max = 10 ** np.ceil(np.log10(np.nanmax(q75)))

    fig, ax = plt.subplots(figsize=(2.25, 1.85))
    if len(all_peaks):
        for freq in all_peaks:
            ax.axvline(freq, color="#b7a9a6", alpha=0.045, lw=0.7, zorder=0)
    ax.fill_between(freqs, q25, q75, color="#d0d0d0", alpha=0.65, lw=0, zorder=1)
    ax.plot(freqs, median, color="#303030", lw=1.35, zorder=2)
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
    fig.savefig(plot_path("macaque_lfp_pertrace_embedding_figure4c_style.png", suffix), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_tau_examples(params: pd.DataFrame, diagnostics: list[dict[str, Any]], suffix: str | None, n_examples: int = 6) -> None:
    ok = params[params["status"] == "ok"].copy()
    if ok.empty:
        return
    examples = ok.sort_values(["monkey", "session_id", "lfp_id"]).head(n_examples)
    diag_by_uid = {diag["lfp_uid"]: diag for diag in diagnostics}
    fig, axes = plt.subplots(len(examples), 2, figsize=(7.6, 1.75 * len(examples)), squeeze=False)
    for row_idx, row in enumerate(examples.itertuples()):
        diag = diag_by_uid.get(row.lfp_uid)
        if diag is None:
            continue
        lags = diag["lags"]
        axes[row_idx, 0].plot(lags, diag["ami"], color="#8b0000", lw=1.2)
        axes[row_idx, 0].axvline(row.torus_tau, color="#201715", lw=0.9, ls="--")
        axes[row_idx, 0].set_ylabel(str(row.lfp_uid), fontsize=7)
        axes[row_idx, 1].plot(lags, diag["acf"], color="#303030", lw=1.2)
        axes[row_idx, 1].axvline(row.torus_tau, color="#201715", lw=0.9, ls="--")
    axes[0, 0].set_title("Average mutual information")
    axes[0, 1].set_title("Median autocorrelation")
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    axes[-1, 0].set_xlabel("Lag (ms)")
    axes[-1, 1].set_xlabel("Lag (ms)")
    fig.suptitle("Example Per-LFP Tau Diagnostics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path("lfp_embedding_tau_examples.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_conclusion(params: pd.DataFrame, suffix: str | None) -> None:
    ok = params[params["status"] == "ok"].copy()
    path = table_path("embedding_parameter_conclusion.md", suffix)
    if ok.empty:
        path.write_text("No valid per-trace embedding parameters were selected.\n")
        return
    dim_counts = ok["torus_embedding_dim"].astype(int).value_counts().sort_index()
    tau_med = float(ok["torus_tau_ms"].median())
    tau_iqr = ok["torus_tau_ms"].quantile([0.25, 0.75]).astype(float)
    lines = [
        "# Per-Trace Macaque LFP Embedding Parameters",
        "",
        f"Selected tau and embedding dimension independently for {len(ok)} unique LFP traces using movement-aligned LFP signal structure only.",
        "Tau was chosen from average mutual information with autocorrelation fallbacks; embedding dimension was chosen from robust per-trace PSD peaks using dimension 2K+1.",
        f"Median tau was {tau_med:.1f} ms (IQR {tau_iqr.iloc[0]:.1f}-{tau_iqr.iloc[1]:.1f} ms).",
        "Embedding-dimension counts: "
        + ", ".join(f"{int(dim)}D={int(count)}" for dim, count in dim_counts.items())
        + ".",
        "",
        "These parameters should be used for torus features in the per-trace rerun, while spectral baselines remain unchanged.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", default="movement")
    parser.add_argument("--max-lfps", type=int, default=None)
    parser.add_argument("--monkeys", nargs="+", default=None, choices=["T", "M"])
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-tau", type=int, default=TAU_MAX)
    parser.add_argument("--n-bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--output-suffix", default=OUTPUT_SUFFIX)
    args = parser.parse_args()

    ensure_dirs()
    paths = load_paths(max_lfps=args.max_lfps, monkeys=args.monkeys)
    if not paths:
        raise FileNotFoundError(f"No converted .npz files found in {CONVERTED_DIR}; run convert_motor_lfp.py first.")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_lfp)(
            path,
            args.epoch,
            args.max_tau,
            args.n_bootstraps,
            RANDOM_SEED + i * 101,
        )
        for i, path in enumerate(tqdm(paths, desc="Selecting per-LFP embedding parameters"))
    )
    param_rows = [row for row, _peaks, _diag in results]
    peak_rows = [peak for _row, peaks, _diag in results for peak in peaks]
    diagnostics = [diag for _row, _peaks, diag in results if diag]

    params = pd.DataFrame(param_rows).sort_values(["monkey", "session_id", "lfp_id", "lfp_uid"])
    peaks = pd.DataFrame(peak_rows)
    write_csv(params, table_path("lfp_embedding_params.csv", args.output_suffix))
    write_csv(peaks, table_path("lfp_embedding_peak_table.csv", args.output_suffix))

    plot_parameter_summary(params, args.output_suffix)
    plot_figure4c_style_psd(diagnostics, args.output_suffix)
    plot_tau_examples(params, diagnostics, args.output_suffix)
    write_conclusion(params, args.output_suffix)

    ok = params[params["status"] == "ok"]
    print(f"Selected per-trace parameters for {len(ok)} unique LFP recordings.")
    if not ok.empty:
        print(ok["torus_embedding_dim"].value_counts().sort_index().to_string())
    print(f"Wrote {table_path('lfp_embedding_params.csv', args.output_suffix)}")


if __name__ == "__main__":
    main()
