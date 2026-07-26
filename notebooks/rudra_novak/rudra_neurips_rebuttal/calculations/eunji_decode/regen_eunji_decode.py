#!/usr/bin/env python
"""Regenerate Novak-style lag/torus decoding on Eunji EKEZ LFP data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import colors as mcolors
from scipy import optimize, signal
from scipy.signal import hilbert, welch
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


UNIT_DIR = Path(__file__).resolve().parent
REBUTTAL_DIR = UNIT_DIR.parents[1]
DATA_DIR = REBUTTAL_DIR / "data" / "ekez"
CACHE_DIR = UNIT_DIR / "cache"
PLOT_DIR = UNIT_DIR / "plots"

RAW_FS = 20_000
TARGET_FS = 400
DOWNSAMPLE = RAW_FS // TARGET_FS
N_CHANNELS_TOTAL = 32
DTYPE = np.int16
DTYPE_BYTES = 2

WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)
EMBED_DIM = 3
EMBED_TAU = 1
RANDOM_SEED = 42

PHASE_ORDER = ["mobile", "immobile", "sleep"]
PHASE_LABELS = {"mobile": "Mobile", "immobile": "Immobile", "sleep": "Sleep"}
PHASE_COLORS = {"mobile": "#D85A30", "immobile": "#3266ad", "sleep": "#1D9E75"}
PHASE_PRIORITY = ["sleep", "immobile", "mobile"]
PROBE_ORDER = [18, 19, 12, 13]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

TORUS_KEYS = ["R1", "R2", "minor_radius", "mse", "mean_error", "frac_inside"]
ALL_TORUS_NAMES = [
    "R1",
    "R2",
    "r",
    "mse",
    "mean_error",
    "frac_inside",
    "n_x",
    "n_y",
    "n_z",
    "u_x",
    "u_y",
    "u_z",
    "v_x",
    "v_y",
    "v_z",
]


def parse_mmss(text: str) -> float:
    minutes, seconds = str(text).split(":")
    return int(minutes) * 60 + float(seconds)


def parse_window(text: str) -> tuple[float, float]:
    matches = re.findall(r"([0-9]+:[0-9]+(?:\.[0-9]+)?)", str(text))
    if len(matches) != 2:
        raise ValueError(f"Could not parse window: {text!r}")
    start, end = parse_mmss(matches[0]), parse_mmss(matches[1])
    if end <= start:
        raise ValueError(f"Window end must be after start: {text!r}")
    return start, end


def parse_channels(text: str) -> list[int]:
    return [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def subtract_interval(base: tuple[float, float], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [base]
    for cut_start, cut_end in cuts:
        next_pieces = []
        for start, end in pieces:
            if cut_end <= start or cut_start >= end:
                next_pieces.append((start, end))
                continue
            if cut_start > start:
                next_pieces.append((start, cut_start))
            if cut_end < end:
                next_pieces.append((cut_end, end))
        pieces = next_pieces
    return [(start, end) for start, end in pieces if end - start >= WINDOW_SEC]


def load_label_rows() -> pd.DataFrame:
    rows = []
    labels = pd.read_excel(DATA_DIR / "timestamp.xlsx")
    for _, row in labels.iterrows():
        start, end = parse_window(row["window"])
        rows.append(
            {
                "mouse_ID": str(row["mouse_ID"]),
                "group": str(row["group"]),
                "date": int(row["date"]),
                "phase": str(row["phase"]).strip().lower(),
                "good_channels": parse_channels(row["good_Ch"]),
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "dat_file": f"LFP0_{int(row['date'])}.dat",
            }
        )
    return pd.DataFrame(rows)


def make_exclusive_intervals(label_rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (mouse, date), group in label_rows.groupby(["mouse_ID", "date"], sort=False):
        assigned: list[tuple[float, float]] = []
        for phase in PHASE_PRIORITY:
            phase_group = group[group["phase"] == phase]
            for _, row in phase_group.iterrows():
                pieces = subtract_interval((row["start_s"], row["end_s"]), assigned)
                for start, end in pieces:
                    new_row = row.to_dict()
                    new_row.update(
                        {
                            "exclusive_start_s": start,
                            "exclusive_end_s": end,
                            "exclusive_duration_s": end - start,
                            "trimmed_s": row["duration_s"] - (end - start),
                        }
                    )
                    out_rows.append(new_row)
                assigned.extend(pieces)
        assigned.sort()
    out = pd.DataFrame(out_rows)
    order_map = {phase: i for i, phase in enumerate(PHASE_ORDER)}
    out["phase_order"] = out["phase"].map(order_map)
    return out.sort_values(["date", "phase_order", "exclusive_start_s"]).drop(columns=["phase_order"])


def data_file_shape(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    frame_bytes = N_CHANNELS_TOTAL * DTYPE_BYTES
    if size % frame_bytes != 0:
        raise ValueError(f"{path.name} size is not divisible by {N_CHANNELS_TOTAL} int16 channels")
    return size // frame_bytes, N_CHANNELS_TOTAL


def memmap_dat(path: Path) -> np.memmap:
    n_samples, n_channels = data_file_shape(path)
    raw = np.memmap(path, dtype=DTYPE, mode="r")
    return raw.reshape(n_samples, n_channels)


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def envelope_normalize(
    x: np.ndarray,
    fs: float,
    fband: tuple[float, float] = (1.0, 40.0),
    env_lp_hz: float = 3.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, a = signal.butter(4, np.asarray(fband) / (fs / 2), btype="bandpass")
    x_bp = signal.filtfilt(b, a, x)
    z = hilbert(x_bp)
    envelope = np.abs(z)
    if env_lp_hz is not None:
        b_lp, a_lp = signal.butter(2, env_lp_hz / (fs / 2), btype="low")
        envelope = signal.filtfilt(b_lp, a_lp, envelope)
    x_flat = x_bp / (envelope + eps)
    x_flat *= np.median(envelope)
    return x_flat, x_bp, envelope


def process_lfp_signal(raw_signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    notch_b, notch_a = signal.iirnotch(60, 30, fs)
    filtered = signal.filtfilt(notch_b, notch_a, raw_signal)
    bandpassed = bandpass_filter(filtered, fs, low=0.5, high=40.0)
    flattened, _, _ = envelope_normalize(bandpassed, fs, fband=(1.0, 40.0), env_lp_hz=3.0)
    return flattened, bandpassed


def lag_embed(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    rows = np.arange(n)[:, None]
    cols = (dim - 1 - np.arange(dim)) * tau
    return x[rows + cols]


def compute_band_power(windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    band_power = np.zeros((windows.shape[0], len(BANDS)), dtype=np.float64)
    for i, window in enumerate(windows):
        freqs, psd = welch(window, fs=TARGET_FS, nperseg=min(512, WINDOW_SAMPLES), noverlap=256)
        for j, (_name, (flo, fhi)) in enumerate(BANDS.items()):
            idx = (freqs >= flo) & (freqs <= fhi)
            band_power[i, j] = np.trapezoid(psd[idx], freqs[idx])
    return np.log10(band_power + 1e-12), list(BANDS.keys())


def compute_embed_stats(windows: np.ndarray) -> np.ndarray:
    feature_list = []
    for window in windows:
        points = lag_embed(window, EMBED_DIM, EMBED_TAU)
        mu = points.mean(axis=0)
        sd = points.std(axis=0)
        cov = np.cov(points.T)
        cov_feat = cov[np.triu_indices(EMBED_DIM)]
        feature_list.append(np.concatenate([mu, sd, cov_feat]))
    return np.asarray(feature_list)


def fit_elliptical_torus_3d(points: np.ndarray, lam: float = 1.0, lam_h: float = 0.5, n_modes: int = 3) -> dict[str, object]:
    pts = np.asarray(points, dtype=np.float64)
    n_points = len(pts)

    center0 = pts.mean(axis=0)
    centered = pts - center0
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    pc_extents = 2.0 * singular_values / np.sqrt(n_points)

    r_target = pc_extents[2] / 2.0
    r1_target = max(pc_extents[0] / 2.0 - r_target, r_target)
    r2_target = max(pc_extents[1] / 2.0 - r_target, r_target)

    u0, _v0, n0 = vt[0], vt[1], vt[2]

    def _build_frame(dn: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        r1 = np.linalg.norm(dn) + 1e-12
        n_ax = dn / r1
        u_cand = u0 - np.dot(u0, n_ax) * n_ax
        u_ax = u_cand / (np.linalg.norm(u_cand) + 1e-12)
        v_ax = np.cross(n_ax, u_ax)
        v_ax = v_ax / (np.linalg.norm(v_ax) + 1e-12)
        return r1, n_ax, u_ax, v_ax

    def _ellipse_torus_distance(
        center: np.ndarray,
        n_ax: np.ndarray,
        u_ax: np.ndarray,
        v_ax: np.ndarray,
        r1: float,
        r2: float,
        fit_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dp = fit_points - center
        along_n = (dp @ n_ax)[:, None] * n_ax
        perp = dp - along_n
        cu = perp @ u_ax
        cv = perp @ v_ax
        phi = np.arctan2(cv / (r2 + 1e-12), cu / (r1 + 1e-12))
        ex = r1 * np.cos(phi)
        ey = r2 * np.sin(phi)
        nearest = ex[:, None] * u_ax + ey[:, None] * v_ax + center
        return np.linalg.norm(fit_points - nearest, axis=1), phi

    def residuals(params: np.ndarray) -> np.ndarray:
        center = params[:3]
        dn = params[3:6]
        r1, n_ax, u_ax, v_ax = _build_frame(dn)
        r2 = r1 * np.exp(params[6])
        minor = np.exp(params[7])

        dist, phi = _ellipse_torus_distance(center, n_ax, u_ax, v_ax, r1, r2, pts)
        surface_resid = dist - minor

        w = lam * np.sqrt(n_points)
        reg = np.array(
            [
                w * (2.0 * (r1 + minor) - pc_extents[0]) / (pc_extents[0] + 1e-12),
                w * (2.0 * (r2 + minor) - pc_extents[1]) / (pc_extents[1] + 1e-12),
                w * (2.0 * minor - pc_extents[2]) / (pc_extents[2] + 1e-12),
            ]
        )

        homog = []
        for k in range(1, n_modes + 1):
            homog.append(lam_h * np.sqrt(n_points) * np.mean(surface_resid * np.cos(k * phi)))
            homog.append(lam_h * np.sqrt(n_points) * np.mean(surface_resid * np.sin(k * phi)))
        return np.concatenate([surface_resid, reg, np.asarray(homog)])

    ratio0 = np.clip(r2_target / (r1_target + 1e-12), 0.1, 1.0)
    x0 = np.concatenate([center0, n0 * r1_target, [np.log(ratio0)], [np.log(max(r_target, 1e-3))]])

    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    max_r1 = pc_extents[0] / 2.0
    lb[3:6] = -max_r1
    ub[3:6] = max_r1
    ratio_cap = np.clip(pc_extents[1] / (pc_extents[0] + 1e-12), 0.05, 1.0)
    lb[6] = np.log(0.05)
    ub[6] = np.log(ratio_cap)
    lb[7] = np.log(1e-3)
    ub[7] = np.log(pc_extents[2] / 2.0 + 1e-6)

    x0 = np.clip(x0, lb, ub)
    result = optimize.least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        loss="huber",
        f_scale=1.0,
        max_nfev=4000,
    )

    center = result.x[:3]
    dn = result.x[3:6]
    r1, n_ax, u_ax, v_ax = _build_frame(dn)
    r2 = r1 * np.exp(result.x[6])
    minor = np.exp(result.x[7])

    if abs(np.dot(u_ax, u0)) < abs(np.dot(v_ax, u0)):
        r1, r2 = r2, r1
        u_ax, v_ax = v_ax, u_ax

    dist, _ = _ellipse_torus_distance(center, n_ax, u_ax, v_ax, r1, r2, pts)
    signed = dist - minor
    outside_dist = np.maximum(0.0, signed)
    inside_mask = signed <= 0

    return {
        "center": center,
        "direction": n_ax,
        "u_axis": u_ax,
        "v_axis": v_ax,
        "R1": float(r1),
        "R2": float(r2),
        "minor_radius": float(minor),
        "mse": float(np.mean(outside_dist**2)),
        "mean_error": float(np.mean(outside_dist)),
        "frac_inside": float(inside_mask.sum() / n_points),
        "pc_extents": pc_extents,
    }


def safe_fit(window: np.ndarray) -> dict[str, object] | None:
    try:
        return fit_elliptical_torus_3d(lag_embed(window, EMBED_DIM, EMBED_TAU), lam=1.0, lam_h=0.5)
    except Exception:
        return None


def collect_lfp_windows(intervals: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    metadata_rows = []

    for _, row in intervals.iterrows():
        dat_path = DATA_DIR / row["dat_file"]
        lfp_all = memmap_dat(dat_path)
        duration = lfp_all.shape[0] / RAW_FS
        start_s = max(0.0, float(row["exclusive_start_s"]))
        end_s = min(float(row["exclusive_end_s"]), duration)
        if end_s - start_s < WINDOW_SEC:
            continue
        raw_start = int(round(start_s * RAW_FS))
        raw_end = int(round(end_s * RAW_FS))

        for channel in row["good_channels"]:
            if channel < 0 or channel >= N_CHANNELS_TOTAL:
                raise ValueError(f"Channel {channel} is outside the {N_CHANNELS_TOTAL}-channel .dat file")
            raw_trace = np.asarray(lfp_all[raw_start:raw_end, channel], dtype=np.float64)
            downsampled = signal.resample_poly(raw_trace, up=1, down=DOWNSAMPLE)
            processed, _bandpassed = process_lfp_signal(downsampled, TARGET_FS)

            n_windows = len(processed) // WINDOW_SAMPLES
            for win_idx in range(n_windows):
                local_start = win_idx * WINDOW_SAMPLES
                local_end = local_start + WINDOW_SAMPLES
                window = processed[local_start:local_end]
                windows.append(window.astype(np.float32))
                metadata_rows.append(
                    {
                        "mouse_ID": row["mouse_ID"],
                        "group": row["group"],
                        "date": int(row["date"]),
                        "phase": row["phase"],
                        "label": PHASE_ORDER.index(row["phase"]),
                        "channel": int(channel),
                        "window_start_s": start_s + win_idx * WINDOW_SEC,
                        "window_end_s": start_s + (win_idx + 1) * WINDOW_SEC,
                        "source_interval_start_s": start_s,
                        "source_interval_end_s": end_s,
                        "dat_file": row["dat_file"],
                    }
                )

    return np.asarray(windows, dtype=np.float32), pd.DataFrame(metadata_rows)


def balanced_indices(labels: np.ndarray, max_per_class: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    per_class = [np.where(labels == i)[0] for i in range(len(PHASE_ORDER))]
    min_count = min(len(idx) for idx in per_class)
    if max_per_class is not None:
        min_count = min(min_count, max_per_class)
    chosen = []
    for idx in per_class:
        chosen.append(rng.choice(idx, size=min_count, replace=False))
    all_idx = np.concatenate(chosen)
    rng.shuffle(all_idx)
    return all_idx


def torus_feature_arrays(windows: np.ndarray, n_jobs: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    fits = Parallel(n_jobs=n_jobs, verbose=10)(delayed(safe_fit)(window) for window in windows)

    torus_features = np.full((len(windows), len(TORUS_KEYS)), np.nan, dtype=np.float64)
    orient_features = np.full((len(windows), 9), np.nan, dtype=np.float64)
    failed = []
    for i, fit in enumerate(fits):
        if fit is None:
            failed.append(i)
            continue
        for j, key in enumerate(TORUS_KEYS):
            torus_features[i, j] = fit[key]
        orient_features[i, 0:3] = fit["direction"]
        orient_features[i, 3:6] = fit["u_axis"]
        orient_features[i, 6:9] = fit["v_axis"]

    for arr in (torus_features, orient_features):
        nan_rows = np.isnan(arr).any(axis=1)
        if nan_rows.any():
            col_median = np.nanmedian(arr, axis=0)
            arr[nan_rows] = col_median
    return torus_features, orient_features, failed


def decode_features(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pred = cross_val_predict(lda, features, labels, cv=cv)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average=None)
    cm = confusion_matrix(labels, pred, normalize="true")
    return pred, acc, f1, cm


def representative_indices(features: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    scaled = StandardScaler().fit_transform(features)
    centroids = {state: scaled[labels == state].mean(axis=0) for state in range(len(PHASE_ORDER))}
    reps = {}
    for state in range(len(PHASE_ORDER)):
        state_idx = np.where(labels == state)[0]
        state_feats = scaled[state_idx]
        d_own = np.linalg.norm(state_feats - centroids[state], axis=1)
        other_centroids = [centroid for key, centroid in centroids.items() if key != state]
        d_other = sum(np.linalg.norm(state_feats - centroid, axis=1) for centroid in other_centroids)
        score = d_other - 2.0 * d_own
        reps[state] = int(state_idx[np.argmax(score)])
    return reps


def kde_fill(ax: plt.Axes, values_by_state: dict[str, np.ndarray], xlabel: str) -> None:
    all_values = np.concatenate([v for v in values_by_state.values() if len(v) > 1])
    pad = 0.05 * (all_values.max() - all_values.min() + 1e-12)
    xgrid = np.linspace(all_values.min() - pad, all_values.max() + pad, 300)
    for phase in PHASE_ORDER:
        values = values_by_state[phase]
        if len(values) < 2:
            continue
        kde = gaussian_kde(values, bw_method=0.3)
        ax.fill_between(xgrid, kde(xgrid), alpha=0.3, color=PHASE_COLORS[phase], label=PHASE_LABELS[phase])
        ax.plot(xgrid, kde(xgrid), color=PHASE_COLORS[phase], lw=1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_decode_figure(cache: dict[str, object]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    labels_bal = cache["labels_bal"]
    signal_windows_bal = cache["signal_windows_bal"]
    all_torus_bal = cache["all_torus_bal"]
    band_f1 = cache["band_f1"]
    all_torus_f1 = cache["all_torus_f1"]
    band_acc = cache["band_acc"]
    all_torus_acc = cache["all_torus_acc"]
    cm_all_torus = cache["cm_all_torus"]
    lda_xy = cache["lda_xy"]
    reps = cache["representative_balanced_indices"]

    state_labels = [PHASE_LABELS[p] for p in PHASE_ORDER]
    state_colors = [PHASE_COLORS[p] for p in PHASE_ORDER]

    cmap_cm = mcolors.LinearSegmentedColormap.from_list(
        "white_majority_darkred",
        [(0.0, "#ffffff"), (0.35, "#aca3a36c"), (0.60, "#c10404cf"), (1.0, "#8b0000ff")],
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.28)

    gs_a = gs[0, 0].subgridspec(1, 3, wspace=0.25)
    radii_names = [(0, r"$R_1$"), (1, r"$R_2$"), (2, r"$r$")]
    for k, (col, label) in enumerate(radii_names):
        ax = fig.add_subplot(gs_a[0, k])
        values_by_state = {phase: all_torus_bal[labels_bal == i, col] for i, phase in enumerate(PHASE_ORDER)}
        kde_fill(ax, values_by_state, label)
        if k > 0:
            ax.set_ylabel("")
        if k == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.text(0.035, 0.965, "A", fontsize=22, fontweight="bold")
    fig.text(0.17, 0.965, "Torus radii by state", fontsize=13, fontweight="bold", ha="center")

    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(cm_all_torus, vmin=0, vmax=1, cmap=cmap_cm)
    for i in range(cm_all_torus.shape[0]):
        for j in range(cm_all_torus.shape[1]):
            color = "white" if cm_all_torus[i, j] > 0.55 else "black"
            ax_b.text(j, i, f"{cm_all_torus[i, j]:.2f}", ha="center", va="center", color=color, fontsize=13)
    ax_b.set_xticks(np.arange(len(state_labels)), state_labels)
    ax_b.set_yticks(np.arange(len(state_labels)), state_labels)
    ax_b.set_xlabel("Predicted label")
    ax_b.set_ylabel("True label")
    ax_b.set_title("Lag embedding decoding accuracy", fontsize=12, fontweight="bold")
    ax_b.grid(False)
    fig.text(0.65, 0.965, "B", fontsize=22, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    for state, (name, color) in enumerate(zip(state_labels, state_colors)):
        mask = labels_bal == state
        ax_c.scatter(lda_xy[mask, 0], lda_xy[mask, 1], s=13, alpha=0.45, color=color, label=name)
        if mask.sum() > 3:
            try:
                xy = lda_xy[mask]
                kde = gaussian_kde(xy.T)
                x = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 80)
                y = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 80)
                xx, yy = np.meshgrid(x, y)
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax_c.contour(xx, yy, zz, levels=5, colors=[color], alpha=0.55, linewidths=1.0)
            except np.linalg.LinAlgError:
                pass
    ax_c.set_xlabel("LD1")
    ax_c.set_ylabel("LD2")
    ax_c.set_title("LDA projection", fontsize=12, fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    fig.text(0.035, 0.49, "C", fontsize=22, fontweight="bold")

    ax_d = fig.add_subplot(gs[1, 1])
    x = np.arange(len(PHASE_ORDER))
    width = 0.28
    ax_d.bar(x - width / 2, band_f1, width, color="#B4B2A9", label=f"Band power ({band_acc:.0%})")
    ax_d.bar(x + width / 2, all_torus_f1, width, color="#8b0000", label=f"All torus ({all_torus_acc:.0%})")
    ax_d.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1, label="Chance (33%)")
    ax_d.set_xticks(x, state_labels)
    ax_d.set_ylim(0, 1)
    ax_d.set_ylabel("F1 score")
    ax_d.set_title("Lag embedding performance vs baseline", fontsize=12, fontweight="bold")
    ax_d.legend(frameon=True, loc="upper right", fontsize=8)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    fig.text(0.65, 0.49, "D", fontsize=22, fontweight="bold")

    fig.savefig(PLOT_DIR / "eunji_decode_ref2_style.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    traj_fig = plt.figure(figsize=(12, 4))
    for plot_idx, phase in enumerate(PHASE_ORDER):
        state = PHASE_ORDER.index(phase)
        ax = traj_fig.add_subplot(1, 3, plot_idx + 1, projection="3d")
        win = signal_windows_bal[int(reps[state])]
        points = lag_embed(win[:800], EMBED_DIM, EMBED_TAU)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], lw=0.7, alpha=0.75, color=PHASE_COLORS[phase])
        ax.set_title(PHASE_LABELS[phase], fontweight="bold")
        ax.set_xlabel(r"$x(t)$")
        ax.set_ylabel(r"$x(t-\tau)$")
        ax.set_zlabel(r"$x(t-2\tau)$")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    traj_fig.suptitle("Representative lag-embedding windows", fontsize=13, fontweight="bold")
    traj_fig.tight_layout()
    traj_fig.savefig(PLOT_DIR / "eunji_representative_lag_embeddings.png", dpi=220, bbox_inches="tight")
    plt.close(traj_fig)


def run(force: bool, n_jobs: int, max_per_class: int | None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cache_npz = CACHE_DIR / "eunji_decode_features.npz"
    if cache_npz.exists() and not force:
        loaded = np.load(cache_npz, allow_pickle=True)
        cache = {key: loaded[key] for key in loaded.files}
        plot_decode_figure(cache)
        print(f"Loaded cached features from {cache_npz}")
        print(f"Wrote {PLOT_DIR / 'eunji_decode_ref2_style.png'}")
        print(f"Wrote {PLOT_DIR / 'eunji_representative_lag_embeddings.png'}")
        return

    label_rows = load_label_rows()
    label_rows.to_csv(CACHE_DIR / "eunji_label_rows_raw.csv", index=False)
    intervals = make_exclusive_intervals(label_rows)
    intervals.to_csv(CACHE_DIR / "eunji_label_intervals_exclusive.csv", index=False)

    windows, metadata = collect_lfp_windows(intervals)
    metadata.to_csv(CACHE_DIR / "eunji_all_windows.csv", index=False)
    if windows.size == 0:
        raise RuntimeError("No LFP windows were extracted from the EKEZ spreadsheet.")

    labels = metadata["label"].to_numpy(dtype=int)
    idx_bal = balanced_indices(labels, max_per_class=max_per_class)
    signal_windows_bal = windows[idx_bal]
    metadata_bal = metadata.iloc[idx_bal].reset_index(drop=True)
    labels_bal = metadata_bal["label"].to_numpy(dtype=int)
    metadata_bal.to_csv(CACHE_DIR / "eunji_balanced_windows.csv", index=False)

    band_all, band_names = compute_band_power(windows)
    embed_all = compute_embed_stats(windows)
    band_bal = band_all[idx_bal]
    embed_bal = embed_all[idx_bal]

    torus_features, orient_features, failed = torus_feature_arrays(signal_windows_bal, n_jobs=n_jobs)
    all_torus_bal = np.hstack([torus_features, orient_features])

    pred_band, band_acc, band_f1, cm_band = decode_features(band_bal, labels_bal)
    pred_embed, embed_acc, embed_f1, cm_embed = decode_features(embed_bal, labels_bal)
    pred_torus6, torus6_acc, torus6_f1, cm_torus6 = decode_features(torus_features, labels_bal)
    pred_all_torus, all_torus_acc, all_torus_f1, cm_all_torus = decode_features(all_torus_bal, labels_bal)

    lda_xy = LinearDiscriminantAnalysis().fit_transform(all_torus_bal, labels_bal)
    reps = representative_indices(all_torus_bal, labels_bal)

    scores = pd.DataFrame(
        [
            {"feature_set": "band_power", "accuracy": band_acc, **{f"f1_{p}": band_f1[i] for i, p in enumerate(PHASE_ORDER)}},
            {"feature_set": "lag_embed_stats", "accuracy": embed_acc, **{f"f1_{p}": embed_f1[i] for i, p in enumerate(PHASE_ORDER)}},
            {"feature_set": "torus_6", "accuracy": torus6_acc, **{f"f1_{p}": torus6_f1[i] for i, p in enumerate(PHASE_ORDER)}},
            {"feature_set": "all_torus_15", "accuracy": all_torus_acc, **{f"f1_{p}": all_torus_f1[i] for i, p in enumerate(PHASE_ORDER)}},
        ]
    )
    scores.to_csv(CACHE_DIR / "eunji_decode_scores.csv", index=False)

    summary = {
        "raw_fs_hz": RAW_FS,
        "target_fs_hz": TARGET_FS,
        "window_sec": WINDOW_SEC,
        "window_samples": WINDOW_SAMPLES,
        "n_raw_label_rows": int(len(label_rows)),
        "n_exclusive_intervals": int(len(intervals)),
        "n_all_windows": int(len(windows)),
        "n_balanced_windows": int(len(idx_bal)),
        "class_counts_all": {PHASE_LABELS[p]: int(np.sum(labels == i)) for i, p in enumerate(PHASE_ORDER)},
        "class_counts_balanced": {PHASE_LABELS[p]: int(np.sum(labels_bal == i)) for i, p in enumerate(PHASE_ORDER)},
        "band_names": band_names,
        "torus_fit_failures_imputed": len(failed),
        "scores": scores.to_dict(orient="records"),
        "probe_order_from_tutorial": PROBE_ORDER,
        "channel_interpretation": "Channels are used as direct NumPy columns, matching dat_file_check.ipynb.",
        "exclusive_label_rule": "When spreadsheet intervals overlap, sleep and immobile windows are kept before mobile windows so no decoded window has two labels.",
    }
    (CACHE_DIR / "eunji_decode_summary.json").write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        cache_npz,
        labels_bal=labels_bal,
        signal_windows_bal=signal_windows_bal,
        band_bal=band_bal,
        embed_bal=embed_bal,
        torus_features=torus_features,
        orient_features=orient_features,
        all_torus_bal=all_torus_bal,
        pred_band=pred_band,
        pred_embed=pred_embed,
        pred_torus6=pred_torus6,
        pred_all_torus=pred_all_torus,
        band_acc=np.asarray(band_acc),
        embed_acc=np.asarray(embed_acc),
        torus6_acc=np.asarray(torus6_acc),
        all_torus_acc=np.asarray(all_torus_acc),
        band_f1=band_f1,
        embed_f1=embed_f1,
        torus6_f1=torus6_f1,
        all_torus_f1=all_torus_f1,
        cm_band=cm_band,
        cm_embed=cm_embed,
        cm_torus6=cm_torus6,
        cm_all_torus=cm_all_torus,
        lda_xy=lda_xy,
        representative_balanced_indices=np.asarray([reps[i] for i in range(len(PHASE_ORDER))], dtype=int),
    )

    plot_decode_figure(
        {
            "labels_bal": labels_bal,
            "signal_windows_bal": signal_windows_bal,
            "all_torus_bal": all_torus_bal,
            "band_f1": band_f1,
            "all_torus_f1": all_torus_f1,
            "band_acc": np.asarray(band_acc),
            "all_torus_acc": np.asarray(all_torus_acc),
            "cm_all_torus": cm_all_torus,
            "lda_xy": lda_xy,
            "representative_balanced_indices": np.asarray([reps[i] for i in range(len(PHASE_ORDER))], dtype=int),
        }
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote {cache_npz}")
    print(f"Wrote {PLOT_DIR / 'eunji_decode_ref2_style.png'}")
    print(f"Wrote {PLOT_DIR / 'eunji_representative_lag_embeddings.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute cached windows/features.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Joblib workers for torus fits.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap after class balancing.")
    args = parser.parse_args()
    run(force=args.force, n_jobs=args.n_jobs, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
