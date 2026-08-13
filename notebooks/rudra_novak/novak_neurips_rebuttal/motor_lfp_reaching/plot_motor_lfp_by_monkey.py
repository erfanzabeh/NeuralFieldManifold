#!/usr/bin/env python
"""Create by-monkey motor-LFP decoding summaries from saved per-trace results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from scipy import stats

from motor_lfp_utils import BANDS, PLOT_DIR, TABLE_DIR, write_csv


FEATURE_ORDER = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "low_gamma",
    "average_psd",
    "all_band_power",
    "torus_nonlinear_15",
]
FEATURE_LABELS = {
    "delta": "Delta\n(2-4 Hz)",
    "theta": "Theta\n(4-8 Hz)",
    "alpha": "Alpha\n(8-13 Hz)",
    "beta": "Beta\n(13-30 Hz)",
    "low_gamma": "Low gamma\n(30-55 Hz)",
    "average_psd": "Average\nPSD",
    "all_band_power": "All band\npower",
    "torus_nonlinear_15": "Torus\nfeatures",
}
PAPER_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_red_scale",
    ["#f1d2ca", "#D85A30", "#7f1209"],
)
RELEVANT_BAND = "beta"
TORUS_FEATURE = "torus_nonlinear_15"
AVERAGE_PSD_FEATURE = "average_psd"
REVIEWER_F1_YLIM = 0.30


def suffix_token(suffix: str | None) -> str:
    clean = str(suffix or "").strip().strip("_")
    return f"_{clean}" if clean else ""


def table_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def by_monkey_plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / "by_monkey" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def summary_plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


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


def load_full6_scores(input_suffix: str) -> pd.DataFrame:
    scores_path = table_path("nonlinear_refit_direction_movement_scores.csv", input_suffix)
    manifest_path = TABLE_DIR / "lfp_manifest.csv"
    scores = pd.read_csv(scores_path)
    manifest = pd.read_csv(manifest_path)
    full6 = manifest.loc[
        manifest["has_all_6_directions"].astype(bool),
        ["lfp_uid", "has_all_6_directions"],
    ]
    merged = scores.merge(full6, on="lfp_uid", how="inner")
    return merged[
        (merged["epoch"] == "movement")
        & (merged["target"] == "direction")
        & (merged["status"] == "ok")
        & merged["f1"].notna()
        & merged["feature_set"].isin(FEATURE_ORDER)
    ].copy()


def summarize_by_monkey(scores: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scores.groupby(["monkey", "feature_set"], as_index=False)
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
        .sort_values(["monkey", "feature_set"])
    )
    summary.insert(0, "analysis_set", "full_6_direction_lfps_by_monkey")
    summary.insert(1, "analysis_label", summary["monkey"].map(lambda x: f"Monkey {x}"))
    return summary


def build_relevant_band_comparison(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = [TORUS_FEATURE, AVERAGE_PSD_FEATURE, RELEVANT_BAND]
    rows: list[dict[str, object]] = []
    for monkey, subset in scores.groupby("monkey", sort=True):
        wide = (
            subset[subset["feature_set"].isin(needed)]
            .pivot_table(
                index=["lfp_uid", "monkey", "session_id", "lfp_id"],
                columns="feature_set",
                values="f1",
                aggfunc="mean",
            )
            .dropna(subset=needed)
            .reset_index()
        )
        role_map = [
            ("torus_features", "Torus features", TORUS_FEATURE),
            ("average_psd", "Average PSD", AVERAGE_PSD_FEATURE),
            ("relevant_band", "Relevant band", RELEVANT_BAND),
        ]
        for _, rec in wide.iterrows():
            for role, label, feature_set in role_map:
                rows.append(
                    {
                        "analysis_set": "full_6_direction_lfps_by_monkey",
                        "analysis_label": f"Monkey {monkey}",
                        "monkey": rec["monkey"],
                        "session_id": rec["session_id"],
                        "lfp_uid": rec["lfp_uid"],
                        "lfp_id": rec["lfp_id"],
                        "feature_role": role,
                        "feature_label": label,
                        "feature_set": feature_set,
                        "relevant_band": RELEVANT_BAND,
                        "relevant_band_label": FEATURE_LABELS[RELEVANT_BAND].replace("\n", " "),
                        "f1": float(rec[feature_set]),
                    }
                )
    comparison = pd.DataFrame(rows)
    significance = compute_significance_table(comparison)
    return comparison, significance


def build_overall_relevant_band_comparison(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = [TORUS_FEATURE, AVERAGE_PSD_FEATURE, RELEVANT_BAND]
    wide = (
        scores[scores["feature_set"].isin(needed)]
        .pivot_table(
            index=["lfp_uid", "monkey", "session_id", "lfp_id"],
            columns="feature_set",
            values="f1",
            aggfunc="mean",
        )
        .dropna(subset=needed)
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    role_map = [
        ("torus_features", "Torus features", TORUS_FEATURE),
        ("average_psd", "Average PSD", AVERAGE_PSD_FEATURE),
        ("relevant_band", "Relevant band", RELEVANT_BAND),
    ]
    for _, rec in wide.iterrows():
        for role, label, feature_set in role_map:
            rows.append(
                {
                    "analysis_set": "full_6_direction_lfps",
                    "analysis_label": "Full 6-direction LFPs",
                    "monkey": "M+T",
                    "source_monkey": rec["monkey"],
                    "session_id": rec["session_id"],
                    "lfp_uid": rec["lfp_uid"],
                    "lfp_id": rec["lfp_id"],
                    "feature_role": role,
                    "feature_label": label,
                    "feature_set": feature_set,
                    "relevant_band": RELEVANT_BAND,
                    "relevant_band_label": FEATURE_LABELS[RELEVANT_BAND].replace("\n", " "),
                    "f1": float(rec[feature_set]),
                }
            )
    comparison = pd.DataFrame(rows)
    significance = compute_significance_table(comparison)
    return comparison, significance


def compute_significance_table(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    pairs = [
        ("torus_features", "average_psd"),
        ("torus_features", "relevant_band"),
        ("relevant_band", "average_psd"),
    ]
    rows: list[dict[str, object]] = []
    for (analysis_set, analysis_label, monkey), subset in comparison.groupby(
        ["analysis_set", "analysis_label", "monkey"],
        sort=True,
    ):
        wide = subset.pivot_table(index="lfp_uid", columns="feature_role", values="f1", aggfunc="mean")
        group_rows = []
        for left, right in pairs:
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
                    "analysis_set": analysis_set,
                    "analysis_label": analysis_label,
                    "monkey": monkey,
                    "comparison": f"{left} vs {right}",
                    "feature_left": left,
                    "feature_right": right,
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


def summarize_relevant_band_bars(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    comparison = comparison.copy()
    if "analysis_set" not in comparison.columns:
        comparison["analysis_set"] = "full_6_direction_lfps_by_monkey"
    if "feature_label" not in comparison.columns:
        comparison["feature_label"] = comparison["feature_role"].map(
            {
                "torus_features": "Torus features",
                "average_psd": "Average PSD",
                "relevant_band": "Relevant band",
            }
        )
    role_order = ["torus_features", "average_psd", "relevant_band"]
    summary = (
        comparison.groupby(["analysis_set", "analysis_label", "monkey", "feature_role", "feature_label"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            n_lfps=("lfp_uid", "nunique"),
        )
    )
    summary["role_order"] = summary["feature_role"].map({role: i for i, role in enumerate(role_order)})
    return summary.sort_values(["analysis_label", "role_order"]).drop(columns=["role_order"]).reset_index(drop=True)


def plot_monkey_barplots(summary: pd.DataFrame, output_suffix: str) -> None:
    out_dir = PLOT_DIR / "summary" / "by_monkey"
    out_dir.mkdir(parents=True, exist_ok=True)
    for monkey, subset in summary.groupby("monkey", sort=True):
        sub = subset.set_index("feature_set").reindex(FEATURE_ORDER).dropna(subset=["mean_f1"])
        means = sub["mean_f1"].to_numpy(float)
        stds = sub["std_f1"].fillna(0.0).to_numpy(float)
        norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
        bar_colors = PAPER_RED_CMAP(norm(means))

        fig, ax = plt.subplots(figsize=(10.4, 4.8))
        x = np.arange(len(sub))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=bar_colors,
            edgecolor="#6f1009",
            linewidth=0.8,
            error_kw={"elinewidth": 1.1, "ecolor": "#3a0a06", "capthick": 1.1},
        )
        ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
        ax.set_xticks(x, [FEATURE_LABELS[idx] for idx in sub.index])
        ax.set_ylim(0, REVIEWER_F1_YLIM)
        ax.set_ylabel("F1")
        ax.set_title(f"Reach Direction Decoding From Movement LFP (Monkey {monkey})", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        for bar, mean, std in zip(bars, means, stds):
            label_y = min(REVIEWER_F1_YLIM - 0.012, mean + std + 0.012)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(
            by_monkey_plot_path(f"direction_movement_feature_f1_barplot_monkey_{monkey}.png", output_suffix),
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_by_monkey_heatmap(summary: pd.DataFrame, output_suffix: str) -> None:
    pivot = summary.pivot(index="analysis_label", columns="feature_set", values="mean_f1").reindex(
        index=["Monkey M", "Monkey T"],
        columns=FEATURE_ORDER,
    )
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    im = ax.imshow(pivot.to_numpy(float), vmin=0.0, vmax=REVIEWER_F1_YLIM, cmap=PAPER_RED_CMAP, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.21 else "#1f1715", fontsize=9)
    ax.set_xticks(np.arange(len(FEATURE_ORDER)), [FEATURE_LABELS[idx].replace("\n", " ") for idx in FEATURE_ORDER], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Macaque")
    ax.set_title("Reach Direction Decoding by Macaque", fontweight="bold")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(
        by_monkey_plot_path("direction_movement_feature_f1_heatmap_by_monkey.png", output_suffix),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_relevant_band_comparison_barplot(
    bar_summary: pd.DataFrame,
    significance: pd.DataFrame,
    output_suffix: str,
) -> None:
    if bar_summary.empty:
        return
    monkey_labels = ["Monkey M", "Monkey T"]
    roles = ["torus_features", "average_psd", "relevant_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", "Relevant band\nBeta (13-30 Hz)"]
    colors = ["#8b0000", "#6f6f6f", "#D85A30"]
    pair_positions = {
        "torus_features vs average_psd": (0, 1),
        "torus_features vs relevant_band": (0, 2),
        "relevant_band vs average_psd": (2, 1),
    }
    fixed_y = {
        "torus_features vs average_psd": REVIEWER_F1_YLIM * 0.855,
        "torus_features vs relevant_band": REVIEWER_F1_YLIM * 0.910,
        "relevant_band vs average_psd": REVIEWER_F1_YLIM * 0.960,
    }
    height = REVIEWER_F1_YLIM * 0.010

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), sharey=True)
    for ax, label in zip(axes, monkey_labels):
        sub = bar_summary[bar_summary["analysis_label"] == label].set_index("feature_role").reindex(roles)
        means = sub["mean_f1"].to_numpy(float)
        stds = sub["std_f1"].fillna(0.0).to_numpy(float)
        x = np.arange(len(roles))
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
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, REVIEWER_F1_YLIM)
        ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, mean, std in zip(bars, means, stds):
            label_y = min(REVIEWER_F1_YLIM - 0.056, mean + std + 0.008)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

        sig = significance[significance["analysis_label"] == label].set_index("comparison")
        for comparison_name, (x1, x2) in pair_positions.items():
            text = str(sig.loc[comparison_name, "significance"]) if comparison_name in sig.index else "n/a"
            y = fixed_y[comparison_name]
            ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#1f1715", lw=1.0, clip_on=False)
            ax.text((x1 + x2) / 2, y + height + REVIEWER_F1_YLIM * 0.002, text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].set_ylabel("F1")
    fig.suptitle("Reach Direction Decoding", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(
        by_monkey_plot_path("direction_movement_torus_avgpsd_relevantband_f1_barplot_by_monkey.png", output_suffix),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_overall_relevant_band_comparison_barplot(
    bar_summary: pd.DataFrame,
    significance: pd.DataFrame,
    output_suffix: str,
) -> None:
    if bar_summary.empty:
        return
    roles = ["torus_features", "average_psd", "relevant_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", "Relevant band\nBeta (13-30 Hz)"]
    colors = ["#8b0000", "#6f6f6f", "#D85A30"]
    sub = bar_summary.set_index("feature_role").reindex(roles)
    means = sub["mean_f1"].to_numpy(float)
    stds = sub["std_f1"].fillna(0.0).to_numpy(float)
    x = np.arange(len(roles))

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
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
    ax.set_ylabel("F1")
    ax.set_title("Reach Direction Decoding", fontweight="bold")
    ax.set_ylim(0, REVIEWER_F1_YLIM)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean, std in zip(bars, means, stds):
        label_y = min(REVIEWER_F1_YLIM - 0.056, mean + std + 0.008)
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    sig = significance.set_index("comparison")
    pair_positions = {
        "torus_features vs average_psd": (0, 1),
        "torus_features vs relevant_band": (0, 2),
        "relevant_band vs average_psd": (2, 1),
    }
    fixed_y = {
        "torus_features vs average_psd": REVIEWER_F1_YLIM * 0.855,
        "torus_features vs relevant_band": REVIEWER_F1_YLIM * 0.910,
        "relevant_band vs average_psd": REVIEWER_F1_YLIM * 0.960,
    }
    height = REVIEWER_F1_YLIM * 0.010
    for comparison_name, (x1, x2) in pair_positions.items():
        text = str(sig.loc[comparison_name, "significance"]) if comparison_name in sig.index else "n/a"
        y = fixed_y[comparison_name]
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + height + REVIEWER_F1_YLIM * 0.002, text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        summary_plot_path("direction_movement_torus_avgpsd_relevantband_f1_barplot.png", output_suffix),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_overall_relevant_band_comparison_boxplot(
    comparison: pd.DataFrame,
    significance: pd.DataFrame,
    output_suffix: str,
) -> None:
    if comparison.empty:
        return
    roles = ["torus_features", "average_psd", "relevant_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", "Relevant band\nBeta (13-30 Hz)"]
    colors = ["#8b0000", "#6f6f6f", "#D85A30"]
    values = [comparison.loc[comparison["feature_role"] == role, "f1"].to_numpy(float) for role in roles]

    fig, ax = plt.subplots(figsize=(4.2, 5.1))
    box = ax.boxplot(values, positions=np.arange(1, 4), widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#201715")
        patch.set_linewidth(0.8)
    for item in box["whiskers"] + box["caps"] + box["medians"]:
        item.set_color("#201715")
        item.set_linewidth(1.0)
    ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(np.arange(1, 4), labels)
    ax.set_ylabel("F1")
    ax.set_title("Reach Direction Decoding", fontweight="bold")
    ax.set_ylim(0, REVIEWER_F1_YLIM)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sig = significance.set_index("comparison")
    pair_positions = {
        "torus_features vs average_psd": (1, 2),
        "torus_features vs relevant_band": (1, 3),
        "relevant_band vs average_psd": (3, 2),
    }
    fixed_y = {
        "torus_features vs average_psd": REVIEWER_F1_YLIM * 0.855,
        "torus_features vs relevant_band": REVIEWER_F1_YLIM * 0.910,
        "relevant_band vs average_psd": REVIEWER_F1_YLIM * 0.960,
    }
    height = REVIEWER_F1_YLIM * 0.010
    for comparison_name, (x1, x2) in pair_positions.items():
        text = str(sig.loc[comparison_name, "significance"]) if comparison_name in sig.index else "n/a"
        y = fixed_y[comparison_name]
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + height + REVIEWER_F1_YLIM * 0.002, text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        summary_plot_path("direction_movement_torus_avgpsd_relevantband_f1_boxplot.png", output_suffix),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_relevant_band_comparison(comparison: pd.DataFrame, significance: pd.DataFrame, output_suffix: str) -> None:
    if comparison.empty:
        return
    monkey_labels = ["Monkey M", "Monkey T"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.1), sharey=True)
    roles = ["torus_features", "average_psd", "relevant_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", "Relevant band\nBeta (13-30 Hz)"]
    colors = ["#8b0000", "#6f6f6f", "#D85A30"]
    pair_positions = {
        "torus_features vs average_psd": (1, 2),
        "torus_features vs relevant_band": (1, 3),
        "relevant_band vs average_psd": (3, 2),
    }
    fixed_y = {
        "torus_features vs average_psd": REVIEWER_F1_YLIM * 0.855,
        "torus_features vs relevant_band": REVIEWER_F1_YLIM * 0.910,
        "relevant_band vs average_psd": REVIEWER_F1_YLIM * 0.960,
    }
    height = REVIEWER_F1_YLIM * 0.010

    for ax, label in zip(axes, monkey_labels):
        subset = comparison[comparison["analysis_label"] == label]
        values = [subset.loc[subset["feature_role"] == role, "f1"].to_numpy(float) for role in roles]
        box = ax.boxplot(values, positions=np.arange(1, 4), widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("#201715")
            patch.set_linewidth(0.8)
        for item in box["whiskers"] + box["caps"] + box["medians"]:
            item.set_color("#201715")
            item.set_linewidth(1.0)
        ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
        ax.set_xticks(np.arange(1, 4), labels)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, REVIEWER_F1_YLIM)
        ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        sig = significance[significance["analysis_label"] == label].set_index("comparison")
        for comparison_name, (x1, x2) in pair_positions.items():
            text = str(sig.loc[comparison_name, "significance"]) if comparison_name in sig.index else "n/a"
            y = fixed_y[comparison_name]
            ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#1f1715", lw=1.0, clip_on=False)
            ax.text((x1 + x2) / 2, y + height + REVIEWER_F1_YLIM * 0.002, text, ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].set_ylabel("F1")
    fig.suptitle("Reach Direction Decoding", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(
        by_monkey_plot_path("direction_movement_torus_avgpsd_relevantband_f1_boxplot_by_monkey.png", output_suffix),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-suffix", default="pertrace_tau_dim")
    parser.add_argument("--output-suffix", default="pertrace_tau_dim")
    args = parser.parse_args()

    scores = load_full6_scores(args.input_suffix)
    summary = summarize_by_monkey(scores)
    comparison, significance = build_relevant_band_comparison(scores)
    bar_summary = summarize_relevant_band_bars(comparison)
    overall_comparison, overall_significance = build_overall_relevant_band_comparison(scores)
    overall_bar_summary = summarize_relevant_band_bars(overall_comparison)

    write_csv(summary, table_path("nonlinear_refit_direction_movement_summary_by_monkey.csv", args.output_suffix))
    write_csv(comparison, table_path("torus_avgpsd_relevantband_f1_scores_by_monkey.csv", args.output_suffix))
    write_csv(significance, table_path("torus_avgpsd_relevantband_significance_by_monkey.csv", args.output_suffix))
    write_csv(bar_summary, table_path("torus_avgpsd_relevantband_f1_bar_summary_by_monkey.csv", args.output_suffix))
    write_csv(overall_comparison, table_path("torus_avgpsd_relevantband_f1_scores.csv", args.output_suffix))
    write_csv(overall_significance, table_path("torus_avgpsd_relevantband_significance.csv", args.output_suffix))
    write_csv(overall_bar_summary, table_path("torus_avgpsd_relevantband_f1_bar_summary.csv", args.output_suffix))

    plot_monkey_barplots(summary, args.output_suffix)
    plot_by_monkey_heatmap(summary, args.output_suffix)
    plot_overall_relevant_band_comparison_barplot(overall_bar_summary, overall_significance, args.output_suffix)
    plot_overall_relevant_band_comparison_boxplot(overall_comparison, overall_significance, args.output_suffix)
    plot_relevant_band_comparison_barplot(bar_summary, significance, args.output_suffix)
    plot_relevant_band_comparison(comparison, significance, args.output_suffix)

    counts = summary.groupby("monkey")["n_lfps"].max().to_dict()
    print(f"Full six-direction LFP counts by monkey: {counts}")
    print(f"Wrote {table_path('nonlinear_refit_direction_movement_summary_by_monkey.csv', args.output_suffix)}")
    print(f"Wrote by-monkey plots under {PLOT_DIR / 'summary' / 'by_monkey'}")


if __name__ == "__main__":
    main()
