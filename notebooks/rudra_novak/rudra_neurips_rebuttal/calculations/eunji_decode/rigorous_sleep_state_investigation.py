#!/usr/bin/env python
"""Rigorous EKEZ state-decoding investigation.

This script is deliberately not a Novak-style one-panel rerun. It separates
native-rate LFP spectral/QC analyses from high-rate torus analyses, evaluates
decoding at scientifically meaningful scales, and writes organized plot groups.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regen_eunji_decode import (
    ALL_TORUS_NAMES,
    DATA_DIR,
    DTYPE,
    N_CHANNELS_TOTAL,
    PHASE_COLORS,
    PHASE_LABELS,
    PHASE_ORDER,
    RAW_FS,
    TORUS_KEYS,
    UNIT_DIR,
    fit_elliptical_torus_3d,
    lag_embed,
    memmap_dat,
)


CACHE_DIR = UNIT_DIR / "cache"
PLOT_DIR = UNIT_DIR / "plots"
BASE_CACHE_DIR = UNIT_DIR / "cache"

WINDOW_SEC = 2.0
EDGE_BUFFER_SEC = 2.0
MAX_PER_SESSION_CHANNEL_STATE = 24
PRIMARY_TORUS_FS = 2_000
TORUS_LOW_HZ = 0.5
TORUS_HIGH_HZ = 200.0
TORUS_TAU_MS = 2.0
RANDOM_SEED = 42

STATE_NAMES = [PHASE_LABELS[p] for p in PHASE_ORDER]
STATE_COLORS = [PHASE_COLORS[p] for p in PHASE_ORDER]
STATE_TO_INT = {phase: i for i, phase in enumerate(PHASE_ORDER)}

SPECTRAL_BANDS = {
    "slow": (0.5, 1.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (10.0, 15.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 55.0),
    "high_gamma": (65.0, 120.0),
    "hfo": (120.0, 250.0),
}

FEATURE_SETS = {
    "native_spectrum": lambda cols: [
        c for c in cols if c.startswith("spec_") or c.startswith("ratio_log10_") or c.startswith("aperiodic_")
    ],
    "highrate_dynamics": lambda cols: [c for c in cols if c.startswith("dyn_")],
    "torus_shape": lambda cols: [
        c for c in cols if c.startswith("torus_") and c not in {"torus_fit_failed", "torus_tau_ms", "torus_analysis_fs"}
    ],
    "spectrum_plus_torus": lambda cols: [
        c
        for c in cols
        if c.startswith("spec_")
        or c.startswith("ratio_log10_")
        or c.startswith("aperiodic_")
        or (c.startswith("torus_") and c not in {"torus_fit_failed", "torus_tau_ms", "torus_analysis_fs"})
    ],
    "all_features": lambda cols: [
        c
        for c in cols
        if c.startswith("spec_")
        or c.startswith("ratio_log10_")
        or c.startswith("aperiodic_")
        or c.startswith("dyn_")
        or (c.startswith("torus_") and c not in {"torus_fit_failed", "torus_tau_ms", "torus_analysis_fs"})
    ],
}


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-12 else np.std(x)
    if scale < 1e-12:
        return np.zeros_like(x)
    return (x - med) / scale


def anti_alias_resample(x: np.ndarray, target_fs: int) -> np.ndarray:
    gcd = math.gcd(RAW_FS, target_fs)
    up = target_fs // gcd
    down = RAW_FS // gcd
    return signal.resample_poly(x, up=up, down=down)


def notch_filter(x: np.ndarray, fs: float) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    for freq in [60.0, 120.0, 180.0]:
        if freq < fs / 2 - 1:
            b, a = signal.iirnotch(freq, 30, fs)
            out = signal.filtfilt(b, a, out)
    return out


def bandpass_sos(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    hi = min(high, fs / 2 - 1)
    sos = signal.butter(4, [low / (fs / 2), hi / (fs / 2)], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def choose_evenly_spaced(group: pd.DataFrame, n: int) -> pd.DataFrame:
    group = group.sort_values("window_start_s")
    if len(group) <= n:
        return group
    idx = np.linspace(0, len(group) - 1, n).round().astype(int)
    return group.iloc[np.unique(idx)]


def make_analysis_table(max_per_cell: int, edge_buffer_sec: float) -> pd.DataFrame:
    all_windows = pd.read_csv(BASE_CACHE_DIR / "eunji_all_windows.csv")
    kept = all_windows[
        (all_windows["window_start_s"] >= all_windows["source_interval_start_s"] + edge_buffer_sec)
        & (all_windows["window_end_s"] <= all_windows["source_interval_end_s"] - edge_buffer_sec)
    ].copy()
    sampled = []
    for _, group in kept.groupby(["mouse_ID", "date", "channel", "phase"], sort=False):
        sampled.append(choose_evenly_spaced(group, max_per_cell))
    out = pd.concat(sampled, ignore_index=True)
    out["label"] = out["phase"].map(STATE_TO_INT).astype(int)
    out["session"] = out["mouse_ID"].astype(str) + "_" + out["date"].astype(str)
    out["session_channel"] = out["session"] + "_ch" + out["channel"].astype(str)
    out = out.sort_values(["mouse_ID", "date", "channel", "window_start_s"]).reset_index(drop=True)
    return out


def window_raw(row: pd.Series) -> np.ndarray:
    dat = memmap_dat(DATA_DIR / row["dat_file"])
    start = int(round(row["window_start_s"] * RAW_FS))
    end = int(round(row["window_end_s"] * RAW_FS))
    return np.asarray(dat[start:end, int(row["channel"])], dtype=np.float64)


def spectral_features(raw: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    x = notch_filter(raw, RAW_FS)
    nperseg = min(8192, len(x))
    freqs, psd = signal.welch(x, fs=RAW_FS, nperseg=nperseg, noverlap=nperseg // 2)
    use = (freqs >= 0.5) & (freqs <= 250.0)
    total = np.trapezoid(psd[use], freqs[use]) + 1e-18
    features: dict[str, float] = {}
    for name, (lo, hi) in SPECTRAL_BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        power = np.trapezoid(psd[idx], freqs[idx]) + 1e-18
        features[f"spec_log_abs_{name}"] = np.log10(power)
        features[f"spec_rel_{name}"] = power / total
    features["ratio_theta_delta"] = features["spec_rel_theta"] / (features["spec_rel_delta"] + 1e-12)
    features["ratio_gamma_delta"] = (features["spec_rel_low_gamma"] + features["spec_rel_high_gamma"]) / (
        features["spec_rel_delta"] + 1e-12
    )
    features["ratio_beta_delta"] = features["spec_rel_beta"] / (features["spec_rel_delta"] + 1e-12)
    features["ratio_log10_theta_delta"] = np.log10(features["ratio_theta_delta"] + 1e-12)
    features["ratio_log10_gamma_delta"] = np.log10(features["ratio_gamma_delta"] + 1e-12)
    features["ratio_log10_beta_delta"] = np.log10(features["ratio_beta_delta"] + 1e-12)

    slope_idx = (freqs >= 2.0) & (freqs <= 120.0)
    slope_idx &= ~((freqs >= 55.0) & (freqs <= 65.0))
    slope_idx &= ~((freqs >= 115.0) & (freqs <= 125.0))
    xx = np.log10(freqs[slope_idx] + 1e-12)
    yy = np.log10(psd[slope_idx] + 1e-18)
    if len(xx) > 3:
        slope, intercept = np.polyfit(xx, yy, 1)
    else:
        slope, intercept = np.nan, np.nan
    features["aperiodic_slope_2_120"] = float(slope)
    features["aperiodic_offset_2_120"] = float(intercept)

    return features, freqs.astype(np.float32), psd.astype(np.float32)


def autocorr_lag_seconds(x: np.ndarray, fs: float) -> tuple[float, float]:
    y = robust_zscore(x)
    ac = signal.correlate(y, y, mode="full", method="fft")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-12)
    below = np.where(ac < 1 / np.e)[0]
    zero = np.where(ac < 0)[0]
    return (
        float(below[0] / fs) if len(below) else float(len(ac) / fs),
        float(zero[0] / fs) if len(zero) else float(len(ac) / fs),
    )


def highrate_signal(raw: np.ndarray, fs: int) -> np.ndarray:
    x = anti_alias_resample(raw, fs)
    x = notch_filter(x, fs)
    x = bandpass_sos(x, fs, TORUS_LOW_HZ, TORUS_HIGH_HZ)
    return robust_zscore(x)


def dynamic_features(x: np.ndarray, fs: int) -> dict[str, float]:
    ac_e, ac_zero = autocorr_lag_seconds(x, fs)
    dx = np.diff(x)
    return {
        "dyn_line_length": float(np.mean(np.abs(dx))),
        "dyn_rms": float(np.sqrt(np.mean(x**2))),
        "dyn_ptp": float(np.ptp(x)),
        "dyn_autocorr_e_s": ac_e,
        "dyn_autocorr_zero_s": ac_zero,
        "dyn_hjorth_mobility": float(np.sqrt(np.var(dx) / (np.var(x) + 1e-12))),
    }


def torus_features_from_signal(x: np.ndarray, fs: int) -> dict[str, float]:
    tau_samples = max(1, int(round(TORUS_TAU_MS * fs / 1000)))
    points = lag_embed(x, 3, tau_samples)
    fit = fit_elliptical_torus_3d(points, lam=1.0, lam_h=0.5)
    features: dict[str, float] = {}
    for key in TORUS_KEYS:
        out_key = "torus_r" if key == "minor_radius" else f"torus_{key}"
        features[out_key] = float(fit[key])
    orient = np.concatenate([fit["direction"], fit["u_axis"], fit["v_axis"]])
    for name, value in zip(ALL_TORUS_NAMES[6:], orient):
        features[f"torus_{name}"] = float(value)
    features["torus_tau_ms"] = TORUS_TAU_MS
    features["torus_analysis_fs"] = fs
    return features


def one_feature_row(idx: int, row: pd.Series, torus_fs: int) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    raw = window_raw(row)
    spec, freqs, psd = spectral_features(raw)
    x_hi = highrate_signal(raw, torus_fs)
    dyn = dynamic_features(x_hi, torus_fs)
    try:
        torus = torus_features_from_signal(x_hi, torus_fs)
        torus_failed = 0
    except Exception:
        torus = {f"torus_{name}": np.nan for name in ALL_TORUS_NAMES}
        torus["torus_r"] = np.nan
        torus["torus_tau_ms"] = TORUS_TAU_MS
        torus["torus_analysis_fs"] = torus_fs
        torus_failed = 1
    qc = {
        "qc_raw_mean": float(np.mean(raw)),
        "qc_raw_std": float(np.std(raw)),
        "qc_raw_ptp": float(np.ptp(raw)),
        "qc_clip_fraction": float(np.mean((raw <= np.iinfo(DTYPE).min + 1) | (raw >= np.iinfo(DTYPE).max - 1))),
        "torus_fit_failed": torus_failed,
    }
    out = {"analysis_row": idx}
    out.update(qc)
    out.update(spec)
    out.update(dyn)
    out.update(torus)
    return out, freqs, psd


def impute_nan_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if out[col].isna().any():
            med = out[col].median()
            out[col] = out[col].fillna(med)
    return out


def balance_training_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idxs = []
    min_count = min(np.sum(y == state) for state in range(len(PHASE_ORDER)))
    for state in range(len(PHASE_ORDER)):
        state_idx = np.where(y == state)[0]
        idxs.append(rng.choice(state_idx, size=min_count, replace=False))
    out = np.concatenate(idxs)
    rng.shuffle(out)
    return out


def split_specs(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    specs: list[tuple[str, np.ndarray]] = []
    specs.append(("leave_session_out", df["session"].to_numpy()))
    specs.append(("leave_channel_out", df["channel"].astype(str).to_numpy()))
    specs.append(("leave_mouse_out", df["mouse_ID"].to_numpy()))
    specs.append(("blocked_stratified_cv", np.array(["all"] * len(df))))
    for mouse, sub in df.groupby("mouse_ID"):
        if sub["date"].nunique() > 1:
            groups = np.array([f"not_{mouse}"] * len(df), dtype=object)
            groups[sub.index.to_numpy()] = sub["date"].astype(str).to_numpy()
            specs.append((f"leave_day_out_{mouse}", groups))
    return specs


def evaluate_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    groups: np.ndarray,
    label_override: np.ndarray | None = None,
    seed: int = RANDOM_SEED,
) -> list[dict[str, object]]:
    y_true = df["label"].to_numpy()
    y_fit = label_override if label_override is not None else y_true
    x = df[feature_cols].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    results = []

    if split_name == "blocked_stratified_cv":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        iterator = cv.split(x, y_fit)
    else:
        logo = LeaveOneGroupOut()
        iterator = logo.split(x, y_fit, groups)

    for fold, (train, test) in enumerate(iterator):
        if len(np.unique(y_fit[train])) < len(PHASE_ORDER) or len(np.unique(y_true[test])) < 2:
            continue
        train_bal_local = balance_training_indices(y_fit[train], rng)
        train_bal = train[train_bal_local]
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        model.fit(x[train_bal], y_fit[train_bal])
        pred = model.predict(x[test])
        f1_each = f1_score(y_true[test], pred, labels=list(range(len(PHASE_ORDER))), average=None, zero_division=0)
        cm = confusion_matrix(y_true[test], pred, labels=list(range(len(PHASE_ORDER))), normalize="true")
        results.append(
            {
                "split": split_name,
                "fold": fold,
                "held_out": str(np.unique(groups[test])[0]) if len(np.unique(groups[test])) == 1 else "blocked",
                "n_train": int(len(train_bal)),
                "n_test": int(len(test)),
                "accuracy": float(accuracy_score(y_true[test], pred)),
                "macro_f1": float(f1_score(y_true[test], pred, average="macro", zero_division=0)),
                "f1_mobile": float(f1_each[0]),
                "f1_immobile": float(f1_each[1]),
                "f1_sleep": float(f1_each[2]),
                "confusion": cm.tolist(),
            }
        )
    return results


def evaluate_all_decoders(df: pd.DataFrame, n_null: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = list(df.columns)
    feature_cols_by_set = {name: selector(all_cols) for name, selector in FEATURE_SETS.items()}
    usable_cols = sorted({col for cols in feature_cols_by_set.values() for col in cols})
    df = impute_nan_features(df, usable_cols)

    rows = []
    for split_name, groups in split_specs(df):
        for feature_set, cols in feature_cols_by_set.items():
            if not cols:
                continue
            for result in evaluate_split(df, cols, split_name, groups):
                result["feature_set"] = feature_set
                rows.append(result)
    scores = pd.DataFrame(rows)

    null_rows = []
    primary_groups = df["session"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED + 1000)
    null_feature_sets = ["native_spectrum", "torus_shape", "spectrum_plus_torus"]
    y_true = df["label"].to_numpy()
    for feature_set in null_feature_sets:
        cols = feature_cols_by_set[feature_set]
        for null_kind in ["label_permutation", "session_channel_circular_shift"]:
            for rep in range(n_null):
                if null_kind == "label_permutation":
                    y_null = rng.permutation(y_true)
                else:
                    y_null = y_true.copy()
                    for _, idx in df.groupby("session_channel").groups.items():
                        idx = np.asarray(list(idx))
                        if len(idx) > 3:
                            shift = int(rng.integers(1, len(idx)))
                            order = np.argsort(df.iloc[idx]["window_start_s"].to_numpy())
                            sorted_idx = idx[order]
                            y_null[sorted_idx] = np.roll(y_null[sorted_idx], shift)
                res = evaluate_split(df, cols, "leave_session_out", primary_groups, label_override=y_null, seed=RANDOM_SEED + rep)
                if res:
                    null_rows.append(
                        {
                            "feature_set": feature_set,
                            "null_kind": null_kind,
                            "rep": rep,
                            "macro_f1": float(np.mean([r["macro_f1"] for r in res])),
                        }
                    )
    null_scores = pd.DataFrame(null_rows)
    return scores, null_scores


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, feature_set), group in scores.groupby(["split", "feature_set"]):
        rows.append(
            {
                "split": split,
                "feature_set": feature_set,
                "n_folds": int(len(group)),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_sem": group["macro_f1"].std(ddof=1) / np.sqrt(len(group)) if len(group) > 1 else 0.0,
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_sem": group["accuracy"].std(ddof=1) / np.sqrt(len(group)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def extract_features(force: bool, n_jobs: int, max_per_cell: int, torus_fs: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = CACHE_DIR / f"features_fs{torus_fs}_cap{max_per_cell}.csv"
    meta_path = CACHE_DIR / f"analysis_windows_cap{max_per_cell}.csv"
    psd_path = CACHE_DIR / f"native_psd_fs{torus_fs}_cap{max_per_cell}.npz"
    if feature_path.exists() and meta_path.exists() and psd_path.exists() and not force:
        df = pd.read_csv(feature_path)
        df = ensure_derived_features(df)
        psd_cache = np.load(psd_path)
        return df, psd_cache["freqs"], psd_cache["psd"]

    meta = make_analysis_table(max_per_cell=max_per_cell, edge_buffer_sec=EDGE_BUFFER_SEC)
    meta.to_csv(meta_path, index=False)
    outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(one_feature_row)(idx, row, torus_fs) for idx, row in meta.iterrows()
    )
    feature_rows = [out[0] for out in outputs]
    freqs = outputs[0][1]
    psd = np.vstack([out[2] for out in outputs])
    features = pd.DataFrame(feature_rows)
    df = pd.concat([meta.reset_index(drop=True), features.drop(columns=["analysis_row"])], axis=1)
    df = ensure_derived_features(df)
    torus_cols = [c for c in df.columns if c.startswith("torus_") and c not in {"torus_fit_failed"}]
    df = impute_nan_features(df, torus_cols)
    df.to_csv(feature_path, index=False)
    np.savez_compressed(psd_path, freqs=freqs, psd=psd)
    return df, freqs, psd


def ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in ["theta_delta", "gamma_delta", "beta_delta"]:
        raw_col = f"ratio_{name}"
        log_col = f"ratio_log10_{name}"
        if raw_col in out.columns and log_col not in out.columns:
            out[log_col] = np.log10(out[raw_col].astype(float) + 1e-12)
    return out


def plot_data_qc(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    intervals = pd.read_csv(BASE_CACHE_DIR / "eunji_label_intervals_exclusive.csv")
    y_labels = []
    for y, ((mouse, date), group) in enumerate(intervals.groupby(["mouse_ID", "date"], sort=False)):
        y_labels.append(f"{mouse}\\n{date}")
        for _, row in group.iterrows():
            ax.plot(
                [row["exclusive_start_s"] / 60, row["exclusive_end_s"] / 60],
                [y, y],
                lw=8,
                solid_capstyle="butt",
                color=PHASE_COLORS[row["phase"]],
            )
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_xlabel("Time in recording (min)")
    ax.set_title("Exclusive labeled intervals")

    ax = axes[0, 1]
    counts = df.groupby(["phase", "channel"]).size().unstack(fill_value=0).loc[PHASE_ORDER]
    counts.plot(kind="bar", ax=ax, color=["#777777", "#b44444", "#336699"])
    ax.set_xticklabels(STATE_NAMES, rotation=0)
    ax.set_ylabel("Analysis windows")
    ax.set_title("Balanced-by-cell analysis table")

    ax = axes[1, 0]
    for phase, color in zip(PHASE_ORDER, STATE_COLORS):
        vals = df.loc[df["phase"] == phase, "qc_raw_std"]
        ax.hist(vals, bins=30, alpha=0.35, color=color, density=True, label=PHASE_LABELS[phase])
    ax.set_xlabel("Raw window SD")
    ax.set_ylabel("Density")
    ax.set_title("Raw amplitude QC")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    fail = df.groupby("phase")["torus_fit_failed"].mean().reindex(PHASE_ORDER)
    ax.bar(STATE_NAMES, fail, color=STATE_COLORS)
    ax.set_ylim(0, max(0.02, fail.max() * 1.3 + 1e-6))
    ax.set_ylabel("Failed fit fraction")
    ax.set_title("Torus fit failure check")
    for one in axes.ravel():
        one.spines["top"].set_visible(False)
        one.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_data_qc_label_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_psd_atlas(df: pd.DataFrame, freqs: np.ndarray, psd: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    mask_freq = (freqs >= 0.5) & (freqs <= 250)
    for ax, channel in zip(axes, sorted(df["channel"].unique())):
        ch_mask = df["channel"].to_numpy() == channel
        for phase, color in zip(PHASE_ORDER, STATE_COLORS):
            rows = ch_mask & (df["phase"].to_numpy() == phase)
            if rows.sum() == 0:
                continue
            curves = psd[rows][:, mask_freq]
            med = np.median(curves, axis=0)
            lo = np.percentile(curves, 25, axis=0)
            hi = np.percentile(curves, 75, axis=0)
            ax.loglog(freqs[mask_freq], med, color=color, lw=1.8, label=PHASE_LABELS[phase])
            ax.fill_between(freqs[mask_freq], lo, hi, color=color, alpha=0.15)
        ax.axvspan(55, 65, color="0.85", alpha=0.35)
        ax.axvspan(115, 125, color="0.85", alpha=0.25)
        ax.set_title(f"Channel {channel}")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, ls="--", alpha=0.2)
    axes[0].set_ylabel("Native-rate PSD")
    axes[0].legend(frameon=False)
    fig.suptitle("Native 20 kHz PSD by state and channel", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_native_psd_state_atlas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_atlas(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    torus_plot_cols = [("torus_R1", r"$R_1$"), ("torus_R2", r"$R_2$"), ("torus_r", r"$r$")]
    for ax, (col, title) in zip(axes[0], torus_plot_cols):
        data = [df.loc[df["phase"] == phase, col].to_numpy() for phase in PHASE_ORDER]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], STATE_COLORS):
            body.set_facecolor(color)
            body.set_alpha(0.35)
        parts["cmeans"].set_color("black")
        ax.set_xticks([1, 2, 3], STATE_NAMES)
        ax.set_title(title)
        ax.set_ylabel("Torus value")

    heat_cols = [f"spec_rel_{name}" for name in ["delta", "theta", "beta", "low_gamma", "high_gamma", "hfo"]]
    heat = df.groupby("phase")[heat_cols].mean().reindex(PHASE_ORDER)
    ax = axes[1, 0]
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="magma")
    ax.set_yticks(range(len(PHASE_ORDER)), STATE_NAMES)
    ax.set_xticks(range(len(heat_cols)), [c.replace("spec_rel_", "") for c in heat_cols], rotation=35, ha="right")
    ax.set_title("Mean relative bandpower")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    ratio_cols = ["ratio_log10_theta_delta", "ratio_log10_gamma_delta", "aperiodic_slope_2_120"]
    for col in ratio_cols:
        means = df.groupby("phase")[col].mean().reindex(PHASE_ORDER)
        sem = df.groupby("phase")[col].sem().reindex(PHASE_ORDER)
        ax.errorbar(STATE_NAMES, means, yerr=sem, marker="o", capsize=3, label=col.replace("ratio_log10_", "log ").replace("_", " "))
    ax.set_title("Spectral summaries")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    dyn_cols = ["dyn_line_length", "dyn_autocorr_e_s", "dyn_hjorth_mobility"]
    for col in dyn_cols:
        z = (df[col] - df[col].mean()) / (df[col].std() + 1e-12)
        means = z.groupby(df["phase"]).mean().reindex(PHASE_ORDER)
        sem = z.groupby(df["phase"]).sem().reindex(PHASE_ORDER)
        ax.errorbar(STATE_NAMES, means, yerr=sem, marker="o", capsize=3, label=col.replace("dyn_", ""))
    ax.set_title("High-rate dynamics (z-scored)")
    ax.legend(frameon=False, fontsize=8)

    for one in axes.ravel():
        one.spines["top"].set_visible(False)
        one.spines["right"].set_visible(False)
    fig.suptitle("State feature atlas", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_feature_state_atlas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_decode_matrix(summary: pd.DataFrame) -> None:
    split_order = [
        "blocked_stratified_cv",
        "leave_session_out",
        "leave_channel_out",
        "leave_mouse_out",
        "leave_day_out_ekez003",
        "leave_day_out_ekez004",
    ]
    feature_order = list(FEATURE_SETS.keys())
    pivot = summary.pivot(index="feature_set", columns="split", values="macro_f1_mean").reindex(feature_order)[split_order]
    sem = summary.pivot(index="feature_set", columns="split", values="macro_f1_sem").reindex(feature_order)[split_order]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            err = sem.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}\n+/-{err:.2f}", ha="center", va="center", color="white" if val < 0.65 else "black", fontsize=8)
    ax.set_yticks(range(len(feature_order)), [x.replace("_", " ") for x in feature_order])
    ax.set_xticks(range(len(split_order)), [x.replace("_", " ") for x in split_order], rotation=30, ha="right")
    ax.set_title("Macro-F1 by feature set and generalization scale", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_decode_generalization_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nulls(scores: pd.DataFrame, null_scores: pd.DataFrame) -> None:
    real = scores[scores["split"] == "leave_session_out"].groupby("feature_set")["macro_f1"].mean()
    feature_sets = ["native_spectrum", "torus_shape", "spectrum_plus_torus"]
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(14, 4), sharey=True)
    for ax, feature_set in zip(axes, feature_sets):
        for null_kind, color in [("label_permutation", "#999999"), ("session_channel_circular_shift", "#4c78a8")]:
            vals = null_scores[
                (null_scores["feature_set"] == feature_set) & (null_scores["null_kind"] == null_kind)
            ]["macro_f1"].to_numpy()
            if len(vals):
                ax.hist(vals, bins=25, alpha=0.45, density=True, color=color, label=null_kind.replace("_", " "))
        ax.axvline(real.get(feature_set, np.nan), color="#8b0000", lw=2, label="real")
        ax.set_title(feature_set.replace("_", " "))
        ax.set_xlabel("Leave-session macro-F1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Null density")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Null distributions for primary generalization test", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_null_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(scores: pd.DataFrame) -> None:
    within = scores[scores["split"] == "leave_session_out"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, feature_set in zip(axes, ["native_spectrum", "torus_shape"]):
        sub = within[within["feature_set"] == feature_set].copy()
        sub["held_out_short"] = sub["held_out"].str.replace("ekez", "e", regex=False)
        ax.bar(sub["held_out_short"], sub["macro_f1"], color="#8b0000" if feature_set == "torus_shape" else "#B4B2A9")
        ax.axhline(1 / 3, color="black", ls="--", alpha=0.4)
        ax.set_ylim(0, 1)
        ax.set_title(feature_set.replace("_", " "))
        ax.tick_params(axis="x", rotation=35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Macro-F1 per held-out session")
    fig.suptitle("Session-level robustness", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_session_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_incremental_gain(scores: pd.DataFrame) -> None:
    rows = []
    for split, group in scores.groupby("split"):
        base = group[group["feature_set"] == "native_spectrum"].set_index("fold")["macro_f1"]
        combo = group[group["feature_set"] == "spectrum_plus_torus"].set_index("fold")["macro_f1"]
        shared = base.index.intersection(combo.index)
        for fold in shared:
            rows.append({"split": split, "fold": fold, "delta_macro_f1": combo.loc[fold] - base.loc[fold]})
    delta = pd.DataFrame(rows)
    delta.to_csv(CACHE_DIR / "incremental_torus_gain.csv", index=False)

    order = [
        "blocked_stratified_cv",
        "leave_session_out",
        "leave_channel_out",
        "leave_mouse_out",
        "leave_day_out_ekez003",
        "leave_day_out_ekez004",
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    means = delta.groupby("split")["delta_macro_f1"].mean().reindex(order)
    sems = delta.groupby("split")["delta_macro_f1"].sem().reindex(order)
    ax.bar(np.arange(len(order)), means, yerr=sems, color="#8b0000", alpha=0.85, capsize=4)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(len(order)), [x.replace("_", " ") for x in order], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 gain over native spectrum")
    ax.set_title("Incremental value of torus features", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_incremental_torus_gain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_sensitivity(base_df: pd.DataFrame, n_jobs: int, force: bool) -> pd.DataFrame:
    path = CACHE_DIR / "sampling_rate_sensitivity.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    rows = []
    small_meta = make_analysis_table(max_per_cell=8, edge_buffer_sec=EDGE_BUFFER_SEC)
    y = small_meta["label"].to_numpy()
    groups = small_meta["session"].to_numpy()
    for fs in [400, 1_000, 2_000]:
        cache_file = CACHE_DIR / f"sensitivity_torus_fs{fs}.csv"
        if cache_file.exists() and not force:
            feat_df = pd.read_csv(cache_file)
        else:
            outputs = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(sensitivity_one_row)(idx, row, fs) for idx, row in small_meta.iterrows()
            )
            feat_df = pd.concat([small_meta.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
            torus_cols = [c for c in feat_df.columns if c.startswith("torus_")]
            feat_df = impute_nan_features(feat_df, torus_cols)
            feat_df.to_csv(cache_file, index=False)
        torus_cols = [c for c in feat_df.columns if c.startswith("torus_") and c not in {"torus_fit_failed"}]
        res = evaluate_split(feat_df, torus_cols, "leave_session_out", groups)
        rows.append(
            {
                "analysis_fs": fs,
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in res])),
                "macro_f1_sem": float(np.std([r["macro_f1"] for r in res], ddof=1) / np.sqrt(len(res))) if len(res) > 1 else 0.0,
                "n_windows": int(len(feat_df)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def sensitivity_one_row(idx: int, row: pd.Series, fs: int) -> dict[str, float]:
    raw = window_raw(row)
    x = highrate_signal(raw, fs)
    try:
        features = torus_features_from_signal(x, fs)
        features["torus_fit_failed"] = 0
    except Exception:
        features = {f"torus_{name}": np.nan for name in ALL_TORUS_NAMES}
        features["torus_r"] = np.nan
        features["torus_fit_failed"] = 1
    return features


def plot_sensitivity(sens: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(sens["analysis_fs"], sens["macro_f1_mean"], yerr=sens["macro_f1_sem"], marker="o", capsize=4, color="#8b0000")
    ax.axhline(1 / 3, color="black", ls="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xticks(sens["analysis_fs"], [str(int(v)) for v in sens["analysis_fs"]])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Torus analysis rate (Hz)")
    ax.set_ylabel("Leave-session macro-F1")
    ax.set_title("Sampling-rate sensitivity", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_sampling_rate_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--max-per-cell", type=int, default=MAX_PER_SESSION_CHANNEL_STATE)
    parser.add_argument("--torus-fs", type=int, default=PRIMARY_TORUS_FS)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df, freqs, psd = extract_features(
        force=args.force,
        n_jobs=args.n_jobs,
        max_per_cell=args.max_per_cell,
        torus_fs=args.torus_fs,
    )
    scores_path = CACHE_DIR / f"decode_scores_fs{args.torus_fs}_cap{args.max_per_cell}.csv"
    null_path = CACHE_DIR / f"decode_nulls_fs{args.torus_fs}_cap{args.max_per_cell}.csv"
    summary_path = CACHE_DIR / f"decode_summary_fs{args.torus_fs}_cap{args.max_per_cell}.csv"
    if scores_path.exists() and null_path.exists() and not args.force:
        scores = pd.read_csv(scores_path)
        null_scores = pd.read_csv(null_path)
    else:
        scores, null_scores = evaluate_all_decoders(df, n_null=args.n_null)
        scores.to_csv(scores_path, index=False)
        null_scores.to_csv(null_path, index=False)
    summary = summarize_scores(scores)
    summary.to_csv(summary_path, index=False)

    plot_data_qc(df)
    plot_psd_atlas(df, freqs, psd)
    plot_feature_atlas(df)
    plot_decode_matrix(summary)
    plot_nulls(scores, null_scores)
    plot_robustness(scores)
    plot_incremental_gain(scores)
    sens = run_sensitivity(df, n_jobs=args.n_jobs, force=args.force)
    plot_sensitivity(sens)

    run_summary = {
        "analysis_windows": int(len(df)),
        "edge_buffer_sec": EDGE_BUFFER_SEC,
        "max_per_session_channel_state": args.max_per_cell,
        "native_psd_fs_hz": RAW_FS,
        "primary_torus_fs_hz": args.torus_fs,
        "torus_band_hz": [TORUS_LOW_HZ, TORUS_HIGH_HZ],
        "torus_tau_ms": TORUS_TAU_MS,
        "n_null_per_kind": args.n_null,
        "class_counts": {PHASE_LABELS[p]: int(np.sum(df["phase"].to_numpy() == p)) for p in PHASE_ORDER},
        "torus_fit_failure_fraction": float(df["torus_fit_failed"].mean()),
        "best_leave_session": summary[summary["split"] == "leave_session_out"]
        .sort_values("macro_f1_mean", ascending=False)
        .head(3)
        .to_dict(orient="records"),
        "leave_session_torus_gain_over_native_spectrum": float(
            summary[
                (summary["split"] == "leave_session_out") & (summary["feature_set"] == "spectrum_plus_torus")
            ]["macro_f1_mean"].iloc[0]
            - summary[
                (summary["split"] == "leave_session_out") & (summary["feature_set"] == "native_spectrum")
            ]["macro_f1_mean"].iloc[0]
        ),
    }
    (CACHE_DIR / "rigorous_run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2))
    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
