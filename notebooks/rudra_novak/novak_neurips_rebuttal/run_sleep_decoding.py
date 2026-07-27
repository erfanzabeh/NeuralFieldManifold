#!/usr/bin/env python
"""Run per-session spectral-band and torus decoding for mouse EEG sleep states."""

from __future__ import annotations

import argparse
import json
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


UNIT_DIR = Path(__file__).resolve().parent
DATA_DIR = UNIT_DIR / "eeg_npy_data"
CACHE_DIR = UNIT_DIR / "cache"
PLOT_DIR = UNIT_DIR / "plots"
TABLE_DIR = UNIT_DIR / "tables"

FS = 400
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
EMBED_DIM = 3
EMBED_TAU = 1
RANDOM_SEED = 42

STATE_ORDER = [0, 1, 2]
STATE_NAMES = ["Wake", "NREM", "REM"]
STATE_COLORS = ["#D85A30", "#3266ad", "#1D9E75"]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
    "low_gamma": (30.0, 50.0),
}

TORUS_KEYS = ["R1", "R2", "minor_radius", "mse", "mean_error", "frac_inside"]
FEATURE_SETS = [*BANDS.keys(), "all_torus_15"]
FEATURE_LABELS = {
    "delta": "Delta\n(0.5-4 Hz)",
    "theta": "Theta\n(4-8 Hz)",
    "alpha": "Alpha\n(8-12 Hz)",
    "sigma": "Sigma\n(12-16 Hz)",
    "beta": "Beta\n(16-30 Hz)",
    "low_gamma": "Low gamma\n(30-50 Hz)",
    "all_torus_15": "Torus\nfeatures",
}
PAPER_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_red_scale",
    ["#f1d2ca", "#D85A30", "#7f1209"],
)


def session_hour_label(session_id: str) -> str:
    try:
        start_minute = int(session_id.split("_")[1].removeprefix("m"))
    except (IndexError, ValueError):
        return session_id
    return f"Hour {start_minute // 60 + 1}"


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def envelope_normalize(
    x: np.ndarray,
    fs: float,
    fband: tuple[float, float] = (1.0, 50.0),
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


def process_eeg_signal(raw_signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    notch_b, notch_a = signal.iirnotch(60, 30, fs)
    filtered = signal.filtfilt(notch_b, notch_a, np.asarray(raw_signal).ravel())
    bandpassed = bandpass_filter(filtered, fs, low=0.5, high=50.0)
    flattened, _, _ = envelope_normalize(bandpassed, fs, fband=(1.0, 50.0), env_lp_hz=3.0)
    return flattened, bandpassed


def lag_embed(x: np.ndarray, dim: int = EMBED_DIM, tau: int = EMBED_TAU) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    rows = np.arange(n)[:, None]
    cols = (dim - 1 - np.arange(dim)) * tau
    return x[rows + cols]


def compute_band_power(windows: np.ndarray) -> np.ndarray:
    band_power = np.zeros((windows.shape[0], len(BANDS)), dtype=np.float64)
    for i, window in enumerate(windows):
        freqs, psd = welch(window, fs=FS, nperseg=min(512, WINDOW_SAMPLES), noverlap=256)
        for j, (_name, (flo, fhi)) in enumerate(BANDS.items()):
            idx = (freqs >= flo) & (freqs <= fhi)
            band_power[i, j] = np.trapezoid(psd[idx], freqs[idx])
    return np.log10(band_power + 1e-12)


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
    result = optimize.least_squares(residuals, x0, bounds=(lb, ub), loss="huber", f_scale=1.0, max_nfev=4000)

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
        return fit_elliptical_torus_3d(lag_embed(window), lam=1.0, lam_h=0.5)
    except Exception:
        return None


def balanced_indices(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    per_class = [np.where(labels == state)[0] for state in STATE_ORDER]
    min_count = min(len(idx) for idx in per_class)
    chosen = [rng.choice(idx, size=min_count, replace=False) for idx in per_class]
    all_idx = np.concatenate(chosen)
    rng.shuffle(all_idx)
    return all_idx


def torus_feature_arrays(windows: np.ndarray, n_jobs: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    fits = Parallel(n_jobs=n_jobs, verbose=10)(delayed(safe_fit)(window) for window in windows)
    torus_features = np.full((len(windows), len(TORUS_KEYS)), np.nan, dtype=np.float64)
    orient_features = np.full((len(windows), 9), np.nan, dtype=np.float64)
    failed: list[int] = []
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


def decode_features(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pred = cross_val_predict(lda, features, labels, cv=cv)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average=None, labels=STATE_ORDER)
    macro_f1 = f1_score(labels, pred, average="macro", labels=STATE_ORDER)
    cm = confusion_matrix(labels, pred, labels=STATE_ORDER, normalize="true")
    return pred, acc, macro_f1, f1, cm


def load_session_arrays(session_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal_path = DATA_DIR / f"{session_id}_signal.npy"
    time_path = DATA_DIR / f"{session_id}_time.npy"
    state_path = DATA_DIR / f"{session_id}_state.npy"
    if not signal_path.exists() or not time_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Missing converted .npy files for {session_id}; run convert_mat_to_npy.py first.")
    return np.load(signal_path), np.load(time_path), np.load(state_path)


def session_plot_dir(session_id: str) -> Path:
    return PLOT_DIR / session_id.replace("session_", "session_")


def kde_fill(ax: plt.Axes, values_by_state: dict[int, np.ndarray], xlabel: str) -> None:
    usable = [v for v in values_by_state.values() if len(v) > 1 and np.ptp(v) > 0]
    if not usable:
        return
    all_values = np.concatenate(usable)
    pad = 0.05 * (all_values.max() - all_values.min() + 1e-12)
    xgrid = np.linspace(all_values.min() - pad, all_values.max() + pad, 300)
    for state, color, name in zip(STATE_ORDER, STATE_COLORS, STATE_NAMES):
        values = values_by_state[state]
        if len(values) < 2 or np.ptp(values) == 0:
            continue
        kde = gaussian_kde(values, bw_method=0.3)
        ax.fill_between(xgrid, kde(xgrid), alpha=0.3, color=color, label=name)
        ax.plot(xgrid, kde(xgrid), color=color, lw=1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ref2_style(
    session_id: str,
    labels_bal: np.ndarray,
    all_torus: np.ndarray,
    cm_torus: np.ndarray,
    lda_xy: np.ndarray,
    score_rows: list[dict[str, object]],
) -> None:
    out_dir = session_plot_dir(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap_cm = mcolors.LinearSegmentedColormap.from_list(
        "white_majority_darkred",
        [(0.0, "#ffffff"), (0.35, "#aca3a36c"), (0.60, "#c10404cf"), (1.0, "#8b0000ff")],
    )

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.28)

    gs_a = gs[0, 0].subgridspec(1, 3, wspace=0.28)
    for plot_idx, (col, label) in enumerate([(0, r"$R_1$"), (1, r"$R_2$"), (2, r"$r$")]):
        ax = fig.add_subplot(gs_a[0, plot_idx])
        values_by_state = {state: all_torus[labels_bal == state, col] for state in STATE_ORDER}
        kde_fill(ax, values_by_state, label)
        if plot_idx > 0:
            ax.set_ylabel("")
        if plot_idx == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.text(0.035, 0.965, "A", fontsize=22, fontweight="bold")
    fig.text(0.18, 0.965, f"{session_id}: torus radii by state", fontsize=13, fontweight="bold", ha="center")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(cm_torus, vmin=0, vmax=1, cmap=cmap_cm)
    for i in range(cm_torus.shape[0]):
        for j in range(cm_torus.shape[1]):
            color = "white" if cm_torus[i, j] > 0.55 else "black"
            ax_b.text(j, i, f"{cm_torus[i, j]:.2f}", ha="center", va="center", color=color, fontsize=13)
    ax_b.set_xticks(np.arange(len(STATE_NAMES)), STATE_NAMES)
    ax_b.set_yticks(np.arange(len(STATE_NAMES)), STATE_NAMES)
    ax_b.set_xlabel("Predicted label")
    ax_b.set_ylabel("True label")
    ax_b.set_title("Torus decoding accuracy", fontsize=12, fontweight="bold")
    ax_b.grid(False)
    fig.text(0.65, 0.965, "B", fontsize=22, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    for state, name, color in zip(STATE_ORDER, STATE_NAMES, STATE_COLORS):
        mask = labels_bal == state
        ax_c.scatter(lda_xy[mask, 0], lda_xy[mask, 1], s=14, alpha=0.48, color=color, label=name)
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
    ax_c.set_title("Torus LDA projection", fontsize=12, fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    fig.text(0.035, 0.49, "C", fontsize=22, fontweight="bold")

    ax_d = fig.add_subplot(gs[1, 1])
    valid_rows = [row for row in score_rows if row["feature_set"] in FEATURE_SETS]
    x = np.arange(len(STATE_ORDER))
    total_w = 0.84
    width = total_w / len(valid_rows)
    band_colors = {
        "delta": "#8e79b9",
        "theta": "#3266ad",
        "alpha": "#1D9E75",
        "sigma": "#67A9A5",
        "beta": "#D85A30",
        "low_gamma": "#C8A23A",
        "all_torus_15": "#8b0000",
    }
    for idx, row in enumerate(valid_rows):
        offset = (idx - len(valid_rows) / 2 + 0.5) * width
        f1_vals = np.array([row["f1_wake"], row["f1_nrem"], row["f1_rem"]], dtype=float)
        label = f"{row['feature_set']} ({row['macro_f1']:.0%})"
        ax_d.bar(x + offset, f1_vals, width, color=band_colors[row["feature_set"]], label=label, edgecolor="white", linewidth=0.4)
    ax_d.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1, label="Chance (33%)")
    ax_d.set_xticks(x, STATE_NAMES)
    ax_d.set_ylim(0, 1)
    ax_d.set_ylabel("F1 score")
    ax_d.set_title("Per-band baseline vs torus", fontsize=12, fontweight="bold")
    ax_d.legend(frameon=True, loc="upper right", fontsize=7)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    fig.text(0.65, 0.49, "D", fontsize=22, fontweight="bold")

    fig.savefig(out_dir / "ref2_style.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def process_session(session_id: str, force: bool, n_jobs: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npz = CACHE_DIR / f"{session_id}.npz"
    scores_cache = CACHE_DIR / f"{session_id}_scores.csv"
    status_cache = CACHE_DIR / f"{session_id}_status.json"

    if not force and cache_npz.exists() and scores_cache.exists() and status_cache.exists():
        return pd.read_csv(scores_cache).to_dict(orient="records"), json.loads(status_cache.read_text())

    eeg, _time, labels = load_session_arrays(session_id)
    class_counts = {STATE_NAMES[state].lower(): int(np.sum(labels == state)) for state in STATE_ORDER}
    status = {
        "session_id": session_id,
        "status": "ok",
        "n_windows": int(len(labels)),
        "wake_count": class_counts["wake"],
        "nrem_count": class_counts["nrem"],
        "rem_count": class_counts["rem"],
        "torus_fit_failures_imputed": 0,
    }

    if min(class_counts.values()) < 5:
        status["status"] = "insufficient_classes"
        status["reason"] = "All three classes need at least 5 windows for 5-fold CV."
        rows = []
        for feature_set in FEATURE_SETS:
            rows.append(
                {
                    "session_id": session_id,
                    "feature_set": feature_set,
                    "status": status["status"],
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "f1_wake": np.nan,
                    "f1_nrem": np.nan,
                    "f1_rem": np.nan,
                    **{k: v for k, v in status.items() if k.endswith("_count")},
                }
            )
        pd.DataFrame(rows).to_csv(scores_cache, index=False)
        status_cache.write_text(json.dumps(status, indent=2))
        return rows, status

    processed, _raw_bandpassed = process_eeg_signal(eeg, FS)
    windows = processed[: len(labels) * WINDOW_SAMPLES].reshape(len(labels), WINDOW_SAMPLES)
    idx_bal = balanced_indices(labels)
    labels_bal = labels[idx_bal].astype(int)
    windows_bal = windows[idx_bal].astype(np.float32)

    band_all = compute_band_power(windows)
    band_bal = band_all[idx_bal]
    torus_features, orient_features, failed = torus_feature_arrays(windows_bal, n_jobs=n_jobs)
    all_torus = np.hstack([torus_features, orient_features])
    status["torus_fit_failures_imputed"] = int(len(failed))

    predictions = []
    confusion_mats = []
    rows = []
    for band_idx, band_name in enumerate(BANDS):
        pred, acc, macro_f1, f1_vals, cm = decode_features(band_bal[:, [band_idx]], labels_bal)
        predictions.append(pred)
        confusion_mats.append(cm)
        rows.append(
            {
                "session_id": session_id,
                "feature_set": band_name,
                "status": "ok",
                "accuracy": acc,
                "macro_f1": macro_f1,
                "f1_wake": f1_vals[0],
                "f1_nrem": f1_vals[1],
                "f1_rem": f1_vals[2],
                **{k: v for k, v in status.items() if k.endswith("_count")},
            }
        )

    pred_torus, acc_torus, macro_f1_torus, f1_torus, cm_torus = decode_features(all_torus, labels_bal)
    predictions.append(pred_torus)
    confusion_mats.append(cm_torus)
    rows.append(
        {
            "session_id": session_id,
            "feature_set": "all_torus_15",
            "status": "ok",
            "accuracy": acc_torus,
            "macro_f1": macro_f1_torus,
            "f1_wake": f1_torus[0],
            "f1_nrem": f1_torus[1],
            "f1_rem": f1_torus[2],
            **{k: v for k, v in status.items() if k.endswith("_count")},
        }
    )

    lda_xy = LinearDiscriminantAnalysis().fit_transform(all_torus, labels_bal)
    plot_ref2_style(session_id, labels_bal, all_torus, cm_torus, lda_xy, rows)

    np.savez_compressed(
        cache_npz,
        labels_bal=labels_bal,
        idx_bal=idx_bal,
        windows_bal=windows_bal,
        band_bal=band_bal,
        torus_features=torus_features,
        orient_features=orient_features,
        all_torus_15=all_torus,
        predictions=np.asarray(predictions),
        confusion_mats=np.asarray(confusion_mats),
        feature_sets=np.asarray(FEATURE_SETS),
        lda_xy=lda_xy,
    )
    pd.DataFrame(rows).to_csv(scores_cache, index=False)
    status_cache.write_text(json.dumps(status, indent=2))
    return rows, status


def markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "recording_hour",
        "wake_count",
        "nrem_count",
        "rem_count",
        "status",
        "delta",
        "theta",
        "alpha",
        "sigma",
        "beta",
        "low_gamma",
        "all_torus_15",
    ]
    display = df.copy()
    display["recording_hour"] = display["session_id"].map(session_hour_label)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append("--" if np.isnan(val) else f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_summary_bar(scores: pd.DataFrame, summary_dir: Path) -> None:
    valid = scores[(scores["status"] == "ok") & scores["macro_f1"].notna()].copy()
    if valid.empty:
        return

    stats = (
        valid.groupby("feature_set")["macro_f1"]
        .agg(mean_macro_f1="mean", std_macro_f1="std", n_sessions="count")
        .reindex(FEATURE_SETS)
        .dropna(subset=["mean_macro_f1"])
    )
    display_stats = stats.rename(columns={"mean_macro_f1": "mean_f1", "std_macro_f1": "std_f1"})
    display_stats.to_csv(TABLE_DIR / "session_feature_f1_summary.csv")

    means = stats["mean_macro_f1"].to_numpy(dtype=float)
    stds = stats["std_macro_f1"].fillna(0.0).to_numpy(dtype=float)
    norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = PAPER_RED_CMAP(norm(means))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(stats))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#6f1009",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "ecolor": "#3a0a06", "capthick": 1.1},
    )

    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, [FEATURE_LABELS[idx] for idx in stats.index])
    ax.set_ylim(0, min(1.0, max(0.78, float(np.nanmax(means + stds)) + 0.08)))
    ax.set_ylabel("F1")
    ax.set_title("Average Sleep-Stage Decoding F1 Across Sessions", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.018,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#3a0a06",
        )

    fig.tight_layout()
    fig.savefig(summary_dir / "session_f1_barplot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_summary_bar(scores: pd.DataFrame, summary_dir: Path) -> None:
    valid = scores[(scores["status"] == "ok") & scores["accuracy"].notna()].copy()
    if valid.empty:
        return

    stats = (
        valid.groupby("feature_set")["accuracy"]
        .agg(mean_accuracy="mean", std_accuracy="std", n_sessions="count")
        .reindex(FEATURE_SETS)
        .dropna(subset=["mean_accuracy"])
    )
    stats.to_csv(TABLE_DIR / "session_feature_accuracy_summary.csv")

    means = stats["mean_accuracy"].to_numpy(dtype=float)
    stds = stats["std_accuracy"].fillna(0.0).to_numpy(dtype=float)
    norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = PAPER_RED_CMAP(norm(means))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(stats))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#6f1009",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "ecolor": "#3a0a06", "capthick": 1.1},
    )

    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, [FEATURE_LABELS[idx] for idx in stats.index])
    ax.set_ylim(0, min(1.0, max(0.78, float(np.nanmax(means + stds)) + 0.08)))
    ax.set_ylabel("Accuracy")
    ax.set_title("Average Sleep-Stage Decoding Accuracy Across Sessions", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.018,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#3a0a06",
        )

    fig.tight_layout()
    fig.savefig(summary_dir / "session_accuracy_barplot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(pivot: pd.DataFrame, summary_dir: Path, title: str, colorbar_label: str, out_name: str) -> None:
    valid_pivot = pivot.dropna(how="all")
    if valid_pivot.empty:
        return

    ordered_cols = [col for col in FEATURE_SETS if col in valid_pivot.columns]
    fig_h = max(4, 0.32 * len(valid_pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(valid_pivot[ordered_cols].to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(ordered_cols)), [FEATURE_LABELS[col] for col in ordered_cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(valid_pivot)), [session_hour_label(idx) for idx in valid_pivot.index])
    ax.set_ylabel("Recording Hour")
    ax.set_title(title, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    for i in range(len(valid_pivot)):
        for j, col in enumerate(ordered_cols):
            val = valid_pivot.iloc[i][col]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(summary_dir / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_outputs(score_rows: list[dict[str, object]], statuses: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(TABLE_DIR / "session_decode_scores.csv", index=False)
    pd.DataFrame(statuses).to_csv(TABLE_DIR / "session_class_counts.csv", index=False)

    pivot = scores.pivot_table(index="session_id", columns="feature_set", values="macro_f1", aggfunc="first")
    status_df = pd.DataFrame(statuses).set_index("session_id")
    summary = status_df[["wake_count", "nrem_count", "rem_count", "status"]].join(pivot)
    summary = summary.reset_index()
    for feature_set in FEATURE_SETS:
        if feature_set not in summary.columns:
            summary[feature_set] = np.nan
    markdown_table(summary, TABLE_DIR / "session_summary.md")

    summary_dir = PLOT_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_heatmap(pivot, summary_dir, "Sleep-Stage Decoding F1 Across Recording Hours", "F1", "session_f1_heatmap.png")
    plot_summary_bar(scores, summary_dir)

    accuracy_pivot = scores.pivot_table(index="session_id", columns="feature_set", values="accuracy", aggfunc="first")
    plot_metric_heatmap(
        accuracy_pivot,
        summary_dir,
        "Sleep-Stage Decoding Accuracy Across Recording Hours",
        "Accuracy",
        "session_accuracy_heatmap.png",
    )
    plot_accuracy_summary_bar(scores, summary_dir)


def available_sessions() -> list[str]:
    manifest_path = DATA_DIR / "session_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        return manifest["session_id"].tolist()
    return sorted(path.name.replace("_signal.npy", "") for path in DATA_DIR.glob("session_*_signal.npy"))


def run(force: bool, n_jobs: int, only_sessions: list[str] | None) -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} does not exist; run convert_mat_to_npy.py first.")
    sessions = available_sessions()
    if only_sessions:
        requested = set(only_sessions)
        sessions = [session for session in sessions if session in requested]
        missing = sorted(requested - set(sessions))
        if missing:
            raise ValueError(f"Requested sessions not found: {missing}")
    if not sessions:
        raise RuntimeError("No converted sessions found.")

    all_rows: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for session_id in sessions:
        print(f"=== {session_id} ===")
        rows, status = process_session(session_id, force=force, n_jobs=n_jobs)
        all_rows.extend(rows)
        statuses.append(status)
        print(json.dumps(status, indent=2))

    write_summary_outputs(all_rows, statuses)
    print(f"Wrote {TABLE_DIR / 'session_decode_scores.csv'}")
    print(f"Wrote {TABLE_DIR / 'session_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute cached features and plots.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Joblib workers for torus fitting.")
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        help="Restrict to one session_id; may be passed multiple times.",
    )
    args = parser.parse_args()
    run(force=args.force, n_jobs=args.n_jobs, only_sessions=args.sessions)


if __name__ == "__main__":
    main()
