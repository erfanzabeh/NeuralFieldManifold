#!/usr/bin/env python
"""Additional torus-advantage analyses for the EKEZ unit.

These plots deliberately move away from F1. The goal is to test whether torus
features contain dynamical structure beyond native spectral power, using
cross-domain R2, residual state variance, PSD-matched nulls, RMS matching,
conditional permutation drops, nuisance stability, and time trajectories.
"""

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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rigorous_sleep_state_investigation import (
    CACHE_DIR,
    PLOT_DIR,
    PHASE_ORDER,
    PRIMARY_TORUS_FS,
    RANDOM_SEED,
    STATE_COLORS,
    STATE_NAMES,
    ensure_derived_features,
    highrate_signal,
    impute_nan_features,
    torus_features_from_signal,
    window_raw,
)


FEATURE_PATH = CACHE_DIR / "features_fs2000_cap24.csv"


def spectrum_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("spec_") or c.startswith("ratio_log10_") or c.startswith("aperiodic_")
    ]


def torus_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("torus_") and c not in {"torus_fit_failed", "torus_tau_ms", "torus_analysis_fs"}
    ]


def dyn_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("dyn_")]


def load_features() -> pd.DataFrame:
    df = ensure_derived_features(pd.read_csv(FEATURE_PATH))
    cols = spectrum_cols(df) + torus_cols(df) + dyn_cols(df)
    return impute_nan_features(df, cols)


def one_hot(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((len(labels), len(PHASE_ORDER)), dtype=float)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def grouped_r2(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    alpha: float = 10.0,
) -> pd.DataFrame:
    x = df[feature_cols].to_numpy(dtype=float)
    y = one_hot(df["label"].to_numpy())
    groups = df[group_col].to_numpy()
    logo = LeaveOneGroupOut()
    rows = []
    for fold, (train, test) in enumerate(logo.split(x, y, groups)):
        if len(np.unique(df.iloc[train]["label"])) < len(PHASE_ORDER):
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        baseline = y[train].mean(axis=0, keepdims=True)
        ss_res = np.sum((y[test] - pred) ** 2)
        ss_tot = np.sum((y[test] - baseline) ** 2) + 1e-12
        brier = np.mean((y[test] - pred) ** 2)
        rows.append(
            {
                "group_col": group_col,
                "fold": fold,
                "held_out": str(np.unique(groups[test])[0]),
                "n_test": int(len(test)),
                "r2_onehot": float(1.0 - ss_res / ss_tot),
                "brier": float(brier),
            }
        )
    return pd.DataFrame(rows)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2) + 1e-12
    ss_between = 0.0
    for label in np.unique(labels):
        vals = values[labels == label]
        ss_between += len(vals) * (vals.mean() - grand) ** 2
    return float(ss_between / ss_total)


def residualize_torus_from_spectrum(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = spectrum_cols(df)
    torus = torus_cols(df)
    x = df[spec].to_numpy(dtype=float)
    groups = df["session"].to_numpy()
    logo = LeaveOneGroupOut()
    residual = np.zeros((len(df), len(torus)), dtype=float)
    pred = np.zeros_like(residual)
    rows = []
    y_all = df[torus].to_numpy(dtype=float)
    for fold, (train, test) in enumerate(logo.split(x, groups=groups)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y_all[train])
        pred[test] = model.predict(x[test])
    residual = y_all - pred
    out = df.copy()
    labels = df["label"].to_numpy()
    for j, col in enumerate(torus):
        ss_res = np.sum((y_all[:, j] - pred[:, j]) ** 2)
        ss_tot = np.sum((y_all[:, j] - y_all[:, j].mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[f"resid_{col}"] = residual[:, j]
        rows.append(
            {
                "torus_feature": col,
                "spectrum_to_torus_loso_r2": float(r2),
                "raw_state_eta2": eta_squared(y_all[:, j], labels),
                "residual_state_eta2": eta_squared(residual[:, j], labels),
            }
        )
    return out, pd.DataFrame(rows)


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    phase = rng.uniform(0, 2 * np.pi, size=len(fft))
    phase[0] = np.angle(fft[0])
    if len(phase) > 1:
        phase[-1] = np.angle(fft[-1])
    y = np.fft.irfft(mag * np.exp(1j * phase), n=len(x))
    y = y - np.median(y)
    scale = np.median(np.abs(y)) * 1.4826
    return y / (scale + 1e-12)


def one_psd_null(row_idx: int, row: pd.Series, rep: int, fs: int) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED + 100_000 * rep + row_idx)
    raw = window_raw(row)
    x = highrate_signal(raw, fs)
    x_null = phase_randomize(x, rng)
    try:
        feats = torus_features_from_signal(x_null, fs)
        failed = 0
    except Exception:
        feats = {col: np.nan for col in torus_cols(pd.DataFrame(columns=[]))}
        failed = 1
    out = {"row_index": row_idx, "rep": rep, "null_failed": failed}
    for key, value in feats.items():
        if key.startswith("torus_") and key not in {"torus_tau_ms", "torus_analysis_fs"}:
            out[key] = float(value)
    return out


def build_psd_matched_nulls(df: pd.DataFrame, n_reps: int, fs: int, n_jobs: int, force: bool) -> pd.DataFrame:
    path = CACHE_DIR / f"psd_matched_phase_null_torus_fs{fs}_reps{n_reps}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    jobs = [(idx, row, rep) for rep in range(n_reps) for idx, row in df.iterrows()]
    rows = Parallel(n_jobs=n_jobs, verbose=10)(delayed(one_psd_null)(idx, row, rep, fs) for idx, row, rep in jobs)
    null_df = pd.DataFrame(rows)
    cols = [c for c in null_df.columns if c.startswith("torus_")]
    null_df = impute_nan_features(null_df, cols)
    null_df.to_csv(path, index=False)
    return null_df


def rms_matched_subset(df: pd.DataFrame, bins: int = 12) -> pd.DataFrame:
    out_rows = []
    tmp = df.copy()
    tmp["rms_bin"] = pd.qcut(tmp["qc_raw_std"], q=bins, duplicates="drop")
    rng = np.random.default_rng(RANDOM_SEED + 303)
    for _, group in tmp.groupby("rms_bin", observed=True):
        per_phase = {phase: group[group["phase"] == phase] for phase in PHASE_ORDER}
        min_count = min(len(v) for v in per_phase.values())
        if min_count < 2:
            continue
        for phase, sub in per_phase.items():
            take = sub.sample(n=min_count, random_state=int(rng.integers(0, 1_000_000)))
            out_rows.append(take)
    matched = pd.concat(out_rows, ignore_index=True)
    return matched.drop(columns=["rms_bin"])


def conditional_r2_drops(df: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    spec = spectrum_cols(df)
    torus = torus_cols(df)
    combined = spec + torus
    x = df[combined].to_numpy(dtype=float)
    y = one_hot(df["label"].to_numpy())
    groups = df["session"].to_numpy()
    spec_idx = np.array([combined.index(c) for c in spec])
    torus_idx = np.array([combined.index(c) for c in torus])
    rows = []
    rng = np.random.default_rng(RANDOM_SEED + 404)
    logo = LeaveOneGroupOut()
    for fold, (train, test) in enumerate(logo.split(x, y, groups)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train])
        baseline = y[train].mean(axis=0, keepdims=True)
        pred = model.predict(x[test])
        base_r2 = 1.0 - np.sum((y[test] - pred) ** 2) / (np.sum((y[test] - baseline) ** 2) + 1e-12)
        for group_name, cols in [("spectrum", spec_idx), ("torus", torus_idx)]:
            for rep in range(n_perm):
                xp = x[test].copy()
                perm = rng.permutation(len(test))
                xp[:, cols] = xp[perm][:, cols]
                pred_p = model.predict(xp)
                r2_p = 1.0 - np.sum((y[test] - pred_p) ** 2) / (np.sum((y[test] - baseline) ** 2) + 1e-12)
                rows.append(
                    {
                        "fold": fold,
                        "held_out": str(np.unique(groups[test])[0]),
                        "permuted_group": group_name,
                        "rep": rep,
                        "base_r2": float(base_r2),
                        "permuted_r2": float(r2_p),
                        "r2_drop": float(base_r2 - r2_p),
                    }
                )
    return pd.DataFrame(rows)


def centroid_distance_ratio(df: pd.DataFrame, cols: list[str], domain_col: str) -> float:
    x = StandardScaler().fit_transform(df[cols].to_numpy(dtype=float))
    labels = df["phase"].to_numpy()
    domains = df[domain_col].astype(str).to_numpy()

    def mean_pairwise_centroid_distance(keys: np.ndarray) -> float:
        centroids = []
        for key in np.unique(keys):
            centroids.append(x[keys == key].mean(axis=0))
        if len(centroids) < 2:
            return np.nan
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        return float(np.mean(dists))

    state_dist = mean_pairwise_centroid_distance(labels)
    domain_dist = mean_pairwise_centroid_distance(domains)
    return float(state_dist / (domain_dist + 1e-12))


def plot_state_r2(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "native spectrum": spectrum_cols(df),
        "torus shape": torus_cols(df),
        "spectrum + torus": spectrum_cols(df) + torus_cols(df),
        "all non-F1 features": spectrum_cols(df) + torus_cols(df) + dyn_cols(df),
    }
    splits = {"leave session": "session", "leave channel": "channel", "leave mouse": "mouse_ID"}
    rows = []
    for split_label, group_col in splits.items():
        for feat_label, cols in feature_sets.items():
            res = grouped_r2(df, cols, group_col=group_col)
            res["feature_set"] = feat_label
            res["split"] = split_label
            rows.append(res)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(CACHE_DIR / "state_onehot_r2_by_scale.csv", index=False)

    summary = out.groupby(["split", "feature_set"])["r2_onehot"].agg(["mean", "sem"]).reset_index()
    pivot = summary.pivot(index="feature_set", columns="split", values="mean")
    sem = summary.pivot(index="feature_set", columns="split", values="sem")
    order_rows = list(feature_sets.keys())
    order_cols = list(splits.keys())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    mat = pivot.reindex(order_rows)[order_cols].to_numpy()
    im = ax.imshow(mat, cmap="coolwarm", vmin=-0.2, vmax=0.8)
    for i, feat in enumerate(order_rows):
        for j, split in enumerate(order_cols):
            val = pivot.loc[feat, split]
            err = sem.loc[feat, split]
            ax.text(j, i, f"{val:.2f}\n+/-{err:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(order_cols)), order_cols)
    ax.set_yticks(range(len(order_rows)), order_rows)
    ax.set_title("Held-out state variance explained (one-hot R2)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Cross-domain R2")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_state_r2_not_f1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residual_torus(df: pd.DataFrame) -> pd.DataFrame:
    resid_df, stats = residualize_torus_from_spectrum(df)
    stats.to_csv(CACHE_DIR / "residual_torus_after_spectrum.csv", index=False)
    resid_cols = [c for c in resid_df.columns if c.startswith("resid_torus_")]
    pcs = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(StandardScaler().fit_transform(resid_df[resid_cols]))
    resid_df["resid_torus_pc1"] = pcs[:, 0]
    resid_df["resid_torus_pc2"] = pcs[:, 1]
    resid_df[["mouse_ID", "date", "channel", "phase", "window_start_s", "resid_torus_pc1", "resid_torus_pc2"]].to_csv(
        CACHE_DIR / "residual_torus_pca.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for phase, color, name in zip(PHASE_ORDER, STATE_COLORS, STATE_NAMES):
        mask = resid_df["phase"] == phase
        ax.scatter(resid_df.loc[mask, "resid_torus_pc1"], resid_df.loc[mask, "resid_torus_pc2"], s=16, alpha=0.45, color=color, label=name)
    ax.set_xlabel("Residual torus PC1")
    ax.set_ylabel("Residual torus PC2")
    ax.set_title("Torus residuals after spectrum")
    ax.legend(frameon=False)

    ax = axes[1]
    top = stats.sort_values("residual_state_eta2", ascending=False).head(10)
    ax.scatter(top["spectrum_to_torus_loso_r2"], top["residual_state_eta2"], s=70, color="#8b0000")
    for _, row in top.iterrows():
        label = row["torus_feature"].replace("torus_", "")
        ax.text(row["spectrum_to_torus_loso_r2"], row["residual_state_eta2"], label, fontsize=8, ha="left", va="bottom")
    ax.axvline(0, color="black", lw=1, alpha=0.4)
    ax.set_xlabel("R2: spectrum predicts torus")
    ax.set_ylabel("State eta2 after removing spectrum")
    ax.set_title("Non-spectral state structure")
    for one in axes:
        one.spines["top"].set_visible(False)
        one.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_residual_torus_after_spectrum.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return resid_df


def plot_psd_nulls(df: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    selected = ["torus_R1", "torus_R2", "torus_r", "torus_mean_error", "torus_frac_inside"]
    rows = []
    for col in selected:
        null_mean = null_df.groupby("row_index")[col].mean().reindex(df.index)
        real = df[col].to_numpy(dtype=float)
        diff = real - null_mean.to_numpy(dtype=float)
        rows.append(
            {
                "feature": col,
                "real_mean": float(np.mean(real)),
                "null_mean": float(np.mean(null_mean)),
                "paired_delta": float(np.mean(diff)),
                "paired_dz": float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)),
                "fraction_real_gt_null_mean": float(np.mean(diff > 0)),
            }
        )
    effects = pd.DataFrame(rows)
    effects.to_csv(CACHE_DIR / "psd_matched_phase_null_effects.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    x = np.arange(len(selected))
    ax.bar(x, effects["paired_dz"], color="#8b0000", alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [c.replace("torus_", "") for c in selected], rotation=30, ha="right")
    ax.set_ylabel("Paired effect size vs PSD-matched null")
    ax.set_title("Real torus metrics beyond PSD")

    ax = axes[1]
    for col, color in zip(["torus_R2", "torus_r", "torus_frac_inside"], ["#8b0000", "#3266ad", "#1D9E75"]):
        null_mean = null_df.groupby("row_index")[col].mean().reindex(df.index)
        diff = df[col].to_numpy(dtype=float) - null_mean.to_numpy(dtype=float)
        ax.hist(diff, bins=35, density=True, alpha=0.35, color=color, label=col.replace("torus_", ""))
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Real - PSD-matched null")
    ax.set_ylabel("Density")
    ax.set_title("Paired null differences")
    ax.legend(frameon=False)
    for one in axes:
        one.spines["top"].set_visible(False)
        one.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_psd_matched_torus_nulls.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return effects


def plot_rms_matched(df: pd.DataFrame) -> pd.DataFrame:
    matched = rms_matched_subset(df)
    matched.to_csv(CACHE_DIR / "rms_matched_analysis_windows.csv", index=False)
    feature_sets = {
        "native spectrum": spectrum_cols(df),
        "torus shape": torus_cols(df),
        "spectrum + torus": spectrum_cols(df) + torus_cols(df),
    }
    rows = []
    for table_name, table in [("full", df), ("RMS matched", matched)]:
        for feature_set, cols in feature_sets.items():
            res = grouped_r2(table, cols, group_col="session")
            rows.append(
                {
                    "table": table_name,
                    "feature_set": feature_set,
                    "r2_mean": float(res["r2_onehot"].mean()),
                    "r2_sem": float(res["r2_onehot"].sem()),
                    "n_windows": int(len(table)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CACHE_DIR / "rms_matched_state_r2.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for phase, color, name in zip(PHASE_ORDER, STATE_COLORS, STATE_NAMES):
        ax.hist(df.loc[df["phase"] == phase, "qc_raw_std"], bins=35, alpha=0.25, density=True, color=color)
        ax.hist(matched.loc[matched["phase"] == phase, "qc_raw_std"], bins=20, histtype="step", density=True, color=color, lw=1.8, label=name)
    ax.set_xlabel("Raw window SD")
    ax.set_ylabel("Density")
    ax.set_title("RMS matching")
    ax.legend(frameon=False)

    ax = axes[1]
    width = 0.35
    labels = list(feature_sets.keys())
    full = out[out["table"] == "full"].set_index("feature_set").loc[labels]
    rms = out[out["table"] == "RMS matched"].set_index("feature_set").loc[labels]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, full["r2_mean"], width, yerr=full["r2_sem"], color="#B4B2A9", capsize=3, label="full")
    ax.bar(x + width / 2, rms["r2_mean"], width, yerr=rms["r2_sem"], color="#8b0000", capsize=3, label="RMS matched")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Leave-session state R2")
    ax.set_title("State structure after amplitude matching")
    ax.legend(frameon=False)
    for one in axes:
        one.spines["top"].set_visible(False)
        one.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_rms_matched_state_structure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_conditional_importance(drops: pd.DataFrame) -> pd.DataFrame:
    drops.to_csv(CACHE_DIR / "conditional_group_r2_drops.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [drops.loc[drops["permuted_group"] == group, "r2_drop"].to_numpy() for group in ["spectrum", "torus"]]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], ["#B4B2A9", "#8b0000"]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    parts["cmeans"].set_color("black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks([1, 2], ["spectrum", "torus"])
    ax.set_ylabel("Drop in held-out state R2")
    ax.set_title("Conditional information in combined model", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "13_conditional_r2_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return drops.groupby("permuted_group")["r2_drop"].agg(["mean", "sem"]).reset_index()


def plot_domain_stability(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "native spectrum": spectrum_cols(df),
        "torus shape": torus_cols(df),
        "spectrum + torus": spectrum_cols(df) + torus_cols(df),
    }
    domains = {"session": "session", "channel": "channel", "mouse": "mouse_ID"}
    rows = []
    for fs_name, cols in feature_sets.items():
        for domain_name, col in domains.items():
            rows.append({"feature_set": fs_name, "domain": domain_name, "state_to_domain_distance_ratio": centroid_distance_ratio(df, cols, col)})
    out = pd.DataFrame(rows)
    out.to_csv(CACHE_DIR / "domain_stability_ratios.csv", index=False)
    pivot = out.pivot(index="feature_set", columns="domain", values="state_to_domain_distance_ratio")
    fig, ax = plt.subplots(figsize=(7, 4))
    mat = pivot.loc[list(feature_sets.keys()), list(domains.keys())].to_numpy()
    im = ax.imshow(mat, cmap="viridis")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] < np.nanmean(mat) else "black")
    ax.set_xticks(range(len(domains)), list(domains.keys()))
    ax.set_yticks(range(len(feature_sets)), list(feature_sets.keys()))
    ax.set_title("State separation relative to nuisance shifts", fontweight="bold")
    fig.colorbar(im, ax=ax, label="state centroid distance / domain centroid distance")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "14_domain_stability_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_temporal_trajectories(resid_df: pd.DataFrame) -> None:
    traces = resid_df.groupby(["session", "channel"]).size().sort_values(ascending=False).head(3).index.tolist()
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 3.1 * len(traces)), sharex=False)
    if len(traces) == 1:
        axes = [axes]
    for ax, (session, channel) in zip(axes, traces):
        sub = resid_df[(resid_df["session"] == session) & (resid_df["channel"] == channel)].sort_values("window_start_s").copy()
        t = sub["window_start_s"].to_numpy() / 60
        for col, label, color in [
            ("resid_torus_pc1", "residual torus PC1", "#8b0000"),
            ("spec_rel_theta", "theta rel power", "#3266ad"),
            ("qc_raw_std", "raw SD", "#777777"),
        ]:
            vals = sub[col].to_numpy(dtype=float)
            vals = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-12)
            breaks = np.where(np.diff(t) > 0.12)[0] + 1
            for seg in np.split(np.arange(len(t)), breaks):
                if len(seg) > 1:
                    ax.plot(t[seg], vals[seg], lw=1.2, color=color, alpha=0.8, label=label if seg[0] == 0 else None)
                elif len(seg) == 1:
                    ax.scatter(t[seg], vals[seg], s=14, color=color, alpha=0.8, label=label if seg[0] == 0 else None)
        y0 = ax.get_ylim()[0]
        for phase, color in zip(PHASE_ORDER, STATE_COLORS):
            mask = sub["phase"] == phase
            ax.scatter(t[mask], np.full(mask.sum(), y0), color=color, s=18, marker="s", alpha=0.7)
        ax.set_title(f"{session}, ch{channel}")
        ax.set_ylabel("z-score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time in recording (min)")
    axes[0].legend(frameon=False, loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Temporal trajectories: residual torus vs spectrum/amplitude", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "15_temporal_torus_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-null-reps", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--torus-fs", type=int, default=PRIMARY_TORUS_FS)
    args = parser.parse_args()

    df = load_features()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    state_r2 = plot_state_r2(df)
    resid_df = plot_residual_torus(df)
    null_df = build_psd_matched_nulls(df, n_reps=args.n_null_reps, fs=args.torus_fs, n_jobs=args.n_jobs, force=args.force)
    null_effects = plot_psd_nulls(df, null_df)
    rms = plot_rms_matched(df)
    drops = conditional_r2_drops(df, n_perm=args.n_perm)
    drop_summary = plot_conditional_importance(drops)
    stability = plot_domain_stability(df)
    plot_temporal_trajectories(resid_df)

    summary = {
        "n_windows": int(len(df)),
        "n_psd_matched_null_reps": int(args.n_null_reps),
        "state_r2_leave_session": state_r2[state_r2["split"] == "leave session"]
        .groupby("feature_set")["r2_onehot"]
        .mean()
        .to_dict(),
        "top_residual_torus_eta2": pd.read_csv(CACHE_DIR / "residual_torus_after_spectrum.csv")
        .sort_values("residual_state_eta2", ascending=False)
        .head(5)
        .to_dict(orient="records"),
        "psd_null_effects": null_effects.to_dict(orient="records"),
        "conditional_r2_drop": drop_summary.to_dict(orient="records"),
        "domain_stability": stability.to_dict(orient="records"),
    }
    (CACHE_DIR / "torus_advantage_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote additional torus advantage plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
