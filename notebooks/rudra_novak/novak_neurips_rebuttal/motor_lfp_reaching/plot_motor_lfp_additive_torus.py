#!/usr/bin/env python
"""Decode macaque reaching with spectral features, torus features, and fused feature sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from scipy import stats

from motor_lfp_utils import (
    BANDS,
    CACHE_DIR,
    CONVERTED_DIR,
    DIRECTION_LABELS,
    PLOT_DIR,
    TABLE_DIR,
    decode_features,
    target_values,
    valid_target_mask,
    write_csv,
)


EPOCH = "movement"
TARGET = "direction"
RELEVANT_BAND = "beta"
TORUS_FEATURE = "torus_features"
AVERAGE_PSD_FEATURE = "average_psd"
ALL_BAND_FEATURE = "all_band_power"
TORUS_RELEVANT_FEATURE = "torus_plus_relevant_band"
TORUS_AVERAGE_FEATURE = "torus_plus_average_psd"
TORUS_ALL_BAND_FEATURE = "torus_plus_all_band_power"
FEATURE_ORDER = [
    "delta",
    "theta",
    "alpha",
    "relevant_band",
    "low_gamma",
    AVERAGE_PSD_FEATURE,
    ALL_BAND_FEATURE,
    TORUS_FEATURE,
    TORUS_RELEVANT_FEATURE,
    TORUS_AVERAGE_FEATURE,
    TORUS_ALL_BAND_FEATURE,
]
FEATURE_LABELS = {
    "delta": "Delta\n(2-4 Hz)",
    "theta": "Theta\n(4-8 Hz)",
    "alpha": "Alpha\n(8-13 Hz)",
    "relevant_band": "Relevant band\n(13-30 Hz)",
    "low_gamma": "Low gamma\n(30-55 Hz)",
    AVERAGE_PSD_FEATURE: "Average\nPSD",
    ALL_BAND_FEATURE: "All band\npower",
    TORUS_FEATURE: "Torus\nfeatures",
    TORUS_RELEVANT_FEATURE: "Torus +\nRelevant band",
    TORUS_AVERAGE_FEATURE: "Torus +\nAverage PSD",
    TORUS_ALL_BAND_FEATURE: "Torus +\nAll band power",
}
COMPARISON_ORDER = [
    "relevant_band",
    TORUS_RELEVANT_FEATURE,
    AVERAGE_PSD_FEATURE,
    TORUS_AVERAGE_FEATURE,
    ALL_BAND_FEATURE,
    TORUS_ALL_BAND_FEATURE,
    TORUS_FEATURE,
]
PAPER_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_red_scale",
    ["#f1d2ca", "#D85A30", "#7f1209"],
)
REVIEWER_F1_MIN_YLIM = 0.30
N_TORUS_POINTS = 300
MAX_NFEV = 1200


def suffix_token(suffix: str | None) -> str:
    clean = str(suffix or "").strip().strip("_")
    return f"_{clean}" if clean else ""


def table_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def additive_plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / "additive_torus" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def nonlinear_cache_path(lfp_uid: str, tau: int, embedding_dim: int) -> Path:
    embed = "" if int(embedding_dim) == 3 else f"_embed{int(embedding_dim)}"
    exact = CACHE_DIR / "nonlinear_refit" / f"{lfp_uid}_{EPOCH}_tau{int(tau)}{embed}_pts{N_TORUS_POINTS}_nfev{MAX_NFEV}.npz"
    if exact.exists():
        return exact
    candidates = sorted(
        (CACHE_DIR / "nonlinear_refit").glob(
            f"{lfp_uid}_{EPOCH}_tau{int(tau)}*pts{N_TORUS_POINTS}_nfev{MAX_NFEV}.npz"
        )
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Expected one nonlinear torus cache for {lfp_uid} tau={tau}, dim={embedding_dim}; found {len(candidates)}"
    )


def feature_matrix(
    feature_set: str,
    band_power: np.ndarray,
    average_psd: np.ndarray,
    torus: np.ndarray,
) -> np.ndarray:
    band_names = list(BANDS.keys())
    if feature_set in {"delta", "theta", "alpha", "low_gamma"}:
        return band_power[:, [band_names.index(feature_set)]]
    if feature_set == "relevant_band":
        return band_power[:, [band_names.index(RELEVANT_BAND)]]
    if feature_set == AVERAGE_PSD_FEATURE:
        return average_psd
    if feature_set == ALL_BAND_FEATURE:
        return band_power
    if feature_set == TORUS_FEATURE:
        return torus
    if feature_set == TORUS_RELEVANT_FEATURE:
        return np.hstack([torus, band_power[:, [band_names.index(RELEVANT_BAND)]]])
    if feature_set == TORUS_AVERAGE_FEATURE:
        return np.hstack([torus, average_psd])
    if feature_set == TORUS_ALL_BAND_FEATURE:
        return np.hstack([torus, band_power])
    raise ValueError(f"Unknown feature set: {feature_set}")


def load_full6_lfp_metadata(input_suffix: str) -> pd.DataFrame:
    scores = pd.read_csv(table_path("nonlinear_refit_direction_movement_scores.csv", input_suffix))
    manifest = pd.read_csv(TABLE_DIR / "lfp_manifest.csv")
    full6 = set(manifest.loc[manifest["has_all_6_directions"].astype(bool), "lfp_uid"])
    torus_rows = scores[
        (scores["feature_set"] == "torus_nonlinear_15")
        & (scores["epoch"] == EPOCH)
        & (scores["target"] == TARGET)
        & scores["lfp_uid"].isin(full6)
    ].copy()
    columns = [
        "lfp_uid",
        "monkey",
        "session_id",
        "lfp_id",
        "torus_tau",
        "torus_tau_ms",
        "torus_embedding_dim",
        "torus_param_source",
        "torus_param_id",
        "torus_fit_success_fraction",
    ]
    meta = torus_rows[columns].drop_duplicates("lfp_uid").sort_values(["monkey", "session_id", "lfp_id"])
    return meta.reset_index(drop=True)


def decode_lfp(row: pd.Series) -> list[dict[str, object]]:
    lfp_uid = str(row["lfp_uid"])
    converted_path = CONVERTED_DIR / f"{lfp_uid}.npz"
    cache_path = nonlinear_cache_path(lfp_uid, int(row["torus_tau"]), int(row["torus_embedding_dim"]))
    with np.load(cache_path, allow_pickle=True) as cached:
        keep = cached["keep"]
        band_power = cached["band_power"].astype(float)
        average_psd = cached["average_psd"].astype(float)
        torus = cached["torus"].astype(float)
    with np.load(converted_path, allow_pickle=True) as data:
        labels, classes = target_values(data, TARGET, keep)
        mask = valid_target_mask(data, TARGET, keep)
    band_power = band_power[mask]
    average_psd = average_psd[mask]
    torus = torus[mask]

    rows: list[dict[str, object]] = []
    for feature_set in FEATURE_ORDER:
        x = feature_matrix(feature_set, band_power, average_psd, torus)
        result = decode_features(x, labels, classes=DIRECTION_LABELS, standardize=True)
        rows.append(
            {
                "analysis_epoch": EPOCH,
                "target": TARGET,
                "lfp_uid": lfp_uid,
                "monkey": str(row["monkey"]),
                "session_id": str(row["session_id"]),
                "lfp_id": row["lfp_id"],
                "feature_set": feature_set,
                "feature_label": FEATURE_LABELS[feature_set].replace("\n", " "),
                "source_relevant_band": RELEVANT_BAND,
                "status": result["status"],
                "accuracy": result["accuracy"],
                "f1": result["f1"],
                "n_classes": result["n_classes"],
                "n_trials_balanced": result["n_trials_balanced"],
                "per_class_n": result["per_class_n"],
                "torus_tau": int(row["torus_tau"]),
                "torus_tau_ms": float(row["torus_tau_ms"]),
                "torus_embedding_dim": int(row["torus_embedding_dim"]),
                "torus_param_source": str(row["torus_param_source"]),
                "torus_param_id": str(row["torus_param_id"]),
                "torus_fit_success_fraction": float(row["torus_fit_success_fraction"]),
                "cache_path": str(cache_path.relative_to(cache_path.parents[1])),
            }
        )
    return rows


def build_decode_scores(input_suffix: str) -> pd.DataFrame:
    meta = load_full6_lfp_metadata(input_suffix)
    rows: list[dict[str, object]] = []
    for _, row in meta.iterrows():
        rows.extend(decode_lfp(row))
    return pd.DataFrame(rows)


def add_analysis_groups(scores: pd.DataFrame) -> pd.DataFrame:
    valid = scores[(scores["status"] == "ok") & scores["f1"].notna()].copy()
    pooled = valid.copy()
    pooled["analysis_level"] = "pooled"
    pooled["analysis_label"] = "Pooled macaques"
    by_monkey = valid.copy()
    by_monkey["analysis_level"] = "monkey"
    by_monkey["analysis_label"] = by_monkey["monkey"].map(lambda value: f"Monkey {value}")
    return pd.concat([pooled, by_monkey], ignore_index=True)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = add_analysis_groups(scores)
    summary = (
        grouped.groupby(["analysis_level", "analysis_label", "feature_set", "feature_label"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            n_lfps=("lfp_uid", "nunique"),
            mean_balanced_trials=("n_trials_balanced", "mean"),
            mean_torus_fit_success=("torus_fit_success_fraction", "mean"),
            median_torus_tau_ms=("torus_tau_ms", "median"),
            median_torus_embedding_dim=("torus_embedding_dim", "median"),
        )
    )
    summary["feature_order"] = summary["feature_set"].map({name: i for i, name in enumerate(FEATURE_ORDER)})
    return summary.sort_values(["analysis_level", "analysis_label", "feature_order"]).drop(columns="feature_order")


def star_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def holm_correct(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()


def compute_significance(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = add_analysis_groups(scores)
    pairs = [
        ("torus_plus_relevant_band", "relevant_band", "primary"),
        ("torus_plus_average_psd", "average_psd", "primary"),
        ("torus_plus_all_band_power", "all_band_power", "primary"),
        ("torus_plus_relevant_band", "torus_features", "fused_vs_torus"),
        ("torus_plus_average_psd", "torus_features", "fused_vs_torus"),
        ("torus_plus_all_band_power", "torus_features", "fused_vs_torus"),
        ("torus_features", "relevant_band", "torus_vs_spectral"),
        ("torus_features", "average_psd", "torus_vs_spectral"),
        ("torus_features", "all_band_power", "torus_vs_spectral"),
    ]
    rows: list[dict[str, object]] = []
    for (analysis_level, analysis_label), subset in grouped.groupby(["analysis_level", "analysis_label"], sort=False):
        wide = subset.pivot_table(index="lfp_uid", columns="feature_set", values="f1", aggfunc="mean")
        group_rows: list[dict[str, object]] = []
        for left, right, comparison_family in pairs:
            paired = wide[[left, right]].dropna()
            statistic = np.nan
            p_value = np.nan
            if len(paired) >= 5:
                try:
                    statistic, p_value = stats.wilcoxon(
                        paired[left],
                        paired[right],
                        zero_method="wilcox",
                        alternative="two-sided",
                    )
                except ValueError:
                    statistic, p_value = np.nan, 1.0
            group_rows.append(
                {
                    "analysis_level": analysis_level,
                    "analysis_label": analysis_label,
                    "comparison_family": comparison_family,
                    "comparison": f"{left} vs {right}",
                    "feature_left": left,
                    "feature_right": right,
                    "feature_left_label": FEATURE_LABELS[left].replace("\n", " "),
                    "feature_right_label": FEATURE_LABELS[right].replace("\n", " "),
                    "n_pairs": int(len(paired)),
                    "mean_left": float(paired[left].mean()) if len(paired) else np.nan,
                    "mean_right": float(paired[right].mean()) if len(paired) else np.nan,
                    "mean_difference_left_minus_right": float((paired[left] - paired[right]).mean()) if len(paired) else np.nan,
                    "wilcoxon_statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_uncorrected": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
        corrected = holm_correct([row["p_uncorrected"] for row in group_rows])
        for row, p_holm in zip(group_rows, corrected):
            row["p_holm"] = p_holm
            row["significance"] = star_label(p_holm)
            rows.append(row)
    return pd.DataFrame(rows)


def axis_ylim(mean_values: np.ndarray, std_values: np.ndarray, extra: float = 0.06) -> float:
    top = float(np.nanmax(mean_values + np.nan_to_num(std_values, nan=0.0))) + extra
    return max(REVIEWER_F1_MIN_YLIM, min(0.55, top))


def plot_feature_bar(summary: pd.DataFrame, analysis_label: str, filename: str, suffix: str) -> None:
    sub = summary[summary["analysis_label"] == analysis_label].set_index("feature_set").reindex(FEATURE_ORDER).dropna(subset=["mean_f1"])
    means = sub["mean_f1"].to_numpy(float)
    stds = sub["std_f1"].fillna(0.0).to_numpy(float)
    norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = PAPER_RED_CMAP(norm(means))

    fig, ax = plt.subplots(figsize=(13.6, 5.0))
    x = np.arange(len(sub))
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
    ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, [FEATURE_LABELS[idx] for idx in sub.index])
    ylim = axis_ylim(means, stds, extra=0.075)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("F1")
    ax.set_title(f"Reach Direction Decoding From Movement LFP ({analysis_label})", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for bar, mean, std in zip(bars, means, stds):
        label_y = min(ylim - 0.012, mean + std + 0.010)
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(additive_plot_path(filename, suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(summary: pd.DataFrame, suffix: str) -> None:
    labels = ["Pooled macaques", "Monkey M", "Monkey T"]
    pivot = summary.pivot(index="analysis_label", columns="feature_set", values="mean_f1").reindex(index=labels, columns=FEATURE_ORDER)
    vmax = max(REVIEWER_F1_MIN_YLIM, float(np.nanmax(pivot.to_numpy(float))) + 0.04)

    fig, ax = plt.subplots(figsize=(13.8, 4.0))
    im = ax.imshow(pivot.to_numpy(float), vmin=0.0, vmax=vmax, cmap=PAPER_RED_CMAP, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.22 else "#1f1715", fontsize=8)
    ax.set_xticks(np.arange(len(FEATURE_ORDER)), [FEATURE_LABELS[idx].replace("\n", " ") for idx in FEATURE_ORDER], rotation=32, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Analysis group")
    ax.set_title("Reach Direction Decoding With Additive Torus Features", fontweight="bold")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(additive_plot_path("additive_torus_feature_f1_heatmap.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    analysis_label: str,
    filename: str,
    suffix: str,
) -> None:
    sub = summary[summary["analysis_label"] == analysis_label].set_index("feature_set").reindex(COMPARISON_ORDER)
    means = sub["mean_f1"].to_numpy(float)
    stds = sub["std_f1"].fillna(0.0).to_numpy(float)
    colors = ["#de7a59", "#8b0000", "#8b8b88", "#8b0000", "#b33b25", "#8b0000", "#5f0000"]
    labels = [FEATURE_LABELS[key] for key in COMPARISON_ORDER]

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(COMPARISON_ORDER))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        alpha=0.84,
        edgecolor="#201715",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "ecolor": "#201715", "capthick": 1.1},
    )
    ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, labels)
    ylim = axis_ylim(means, stds, extra=0.11)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("F1")
    ax.set_title(f"Additive Torus Feature Decoding ({analysis_label})", fontweight="bold")
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean, std in zip(bars, means, stds):
        label_y = min(ylim - 0.055, mean + std + 0.009)
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    sig = significance[
        (significance["analysis_label"] == analysis_label)
        & (significance["comparison_family"] == "primary")
    ].set_index("comparison")
    bracket_pairs = [
        ("torus_plus_relevant_band vs relevant_band", 0, 1, 0.74),
        ("torus_plus_average_psd vs average_psd", 2, 3, 0.82),
        ("torus_plus_all_band_power vs all_band_power", 4, 5, 0.90),
    ]
    h = ylim * 0.010
    for comparison_name, x1, x2, frac in bracket_pairs:
        text = str(sig.loc[comparison_name, "significance"]) if comparison_name in sig.index else "n/a"
        y = ylim * frac
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + ylim * 0.006, text, ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(additive_plot_path(filename, suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_by_macaque_comparison(summary: pd.DataFrame, significance: pd.DataFrame, suffix: str) -> None:
    labels = ["Monkey M", "Monkey T"]
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.0), sharey=True)
    for ax, analysis_label in zip(axes, labels):
        sub = summary[summary["analysis_label"] == analysis_label].set_index("feature_set").reindex(COMPARISON_ORDER)
        means = sub["mean_f1"].to_numpy(float)
        stds = sub["std_f1"].fillna(0.0).to_numpy(float)
        colors = ["#de7a59", "#8b0000", "#8b8b88", "#8b0000", "#b33b25", "#8b0000", "#5f0000"]
        x = np.arange(len(COMPARISON_ORDER))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.84,
            edgecolor="#201715",
            linewidth=0.8,
            error_kw={"elinewidth": 1.1, "ecolor": "#201715", "capthick": 1.1},
        )
        ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
        ax.set_xticks(x, [FEATURE_LABELS[key] for key in COMPARISON_ORDER], fontsize=8)
        ax.set_title(analysis_label, fontweight="bold")
        ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.008, f"{mean:.2f}", ha="center", va="bottom", fontsize=7)

        sig = significance[
            (significance["analysis_label"] == analysis_label)
            & (significance["comparison_family"] == "primary")
        ].set_index("comparison")
        for comparison_name, x1, x2, frac in [
            ("torus_plus_relevant_band vs relevant_band", 0, 1, 0.74),
            ("torus_plus_average_psd vs average_psd", 2, 3, 0.82),
            ("torus_plus_all_band_power vs all_band_power", 4, 5, 0.90),
        ]:
            text = str(sig.loc[comparison_name, "significance"]) if comparison_name in sig.index else "n/a"
            y = 0.38 * frac
            h = 0.004
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
            ax.text((x1 + x2) / 2, y + h + 0.002, text, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 0.38)
    axes[0].set_ylabel("F1")
    fig.suptitle("Additive Torus Feature Decoding by Macaque", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(additive_plot_path("by_macaque_additive_torus_comparison_f1_barplot.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-suffix", default="pertrace_tau_dim")
    parser.add_argument("--output-suffix", default="pertrace_tau_dim")
    args = parser.parse_args()

    (PLOT_DIR / "summary" / "additive_torus").mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    scores = build_decode_scores(args.input_suffix)
    summary = summarize_scores(scores)
    significance = compute_significance(scores)

    write_csv(scores, table_path("additive_torus_feature_decode_scores.csv", args.output_suffix))
    write_csv(summary, table_path("additive_torus_feature_summary.csv", args.output_suffix))
    write_csv(significance, table_path("additive_torus_significance.csv", args.output_suffix))

    plot_feature_bar(summary, "Pooled macaques", "pooled_additive_torus_feature_f1_barplot.png", args.output_suffix)
    plot_feature_bar(summary, "Monkey M", "monkey_M_additive_torus_feature_f1_barplot.png", args.output_suffix)
    plot_feature_bar(summary, "Monkey T", "monkey_T_additive_torus_feature_f1_barplot.png", args.output_suffix)
    plot_feature_heatmap(summary, args.output_suffix)
    plot_comparison_bar(summary, significance, "Pooled macaques", "pooled_additive_torus_comparison_f1_barplot.png", args.output_suffix)
    plot_by_macaque_comparison(summary, significance, args.output_suffix)

    counts = summary.groupby("analysis_label")["n_lfps"].max().to_dict()
    print(f"Decoded additive torus features for {counts}")
    print(f"Wrote {table_path('additive_torus_feature_summary.csv', args.output_suffix)}")
    print(f"Wrote additive torus plots under {PLOT_DIR / 'summary' / 'additive_torus'}")


if __name__ == "__main__":
    main()
