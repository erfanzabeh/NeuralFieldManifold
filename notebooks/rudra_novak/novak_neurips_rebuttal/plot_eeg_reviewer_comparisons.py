#!/usr/bin/env python
"""Make reviewer-style EEG sleep decoding comparison plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from scipy import signal, stats

from run_sleep_decoding import (
    BANDS,
    CACHE_DIR,
    FEATURE_LABELS,
    FS,
    PLOT_DIR,
    RANDOM_SEED,
    TABLE_DIR,
    decode_features,
)


TORUS_FEATURE = "all_torus_15"
AVERAGE_PSD_FEATURE = "average_psd"
ALL_BAND_FEATURE = "all_band_power"
RELEVANT_BAND = "delta"
SINGLE_BANDS = list(BANDS.keys())
FEATURE_ORDER = [*SINGLE_BANDS, AVERAGE_PSD_FEATURE, ALL_BAND_FEATURE, TORUS_FEATURE]
FEATURE_LABELS_REVIEW = {
    **FEATURE_LABELS,
    AVERAGE_PSD_FEATURE: "Average\nPSD",
    ALL_BAND_FEATURE: "All band\npower",
}
PAPER_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_red_scale",
    ["#f1d2ca", "#D85A30", "#7f1209"],
)


def compute_average_psd(windows: np.ndarray, low: float = 0.5, high: float = 50.0) -> np.ndarray:
    out = np.zeros((len(windows), 1), dtype=float)
    for i, window in enumerate(windows):
        freqs, psd = signal.welch(window, fs=FS, nperseg=min(512, len(window)), noverlap=256)
        mask = (freqs >= low) & (freqs <= high)
        out[i, 0] = float(np.mean(psd[mask])) if mask.any() else np.nan
    return np.log10(out + 1e-12)


def suffix_token(suffix: str | None) -> str:
    clean = str(suffix or "").strip().strip("_")
    return f"_{clean}" if clean else ""


def table_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def feature_matrix(feature_set: str, band_power: np.ndarray, average_psd: np.ndarray, torus: np.ndarray) -> np.ndarray:
    if feature_set in BANDS:
        return band_power[:, [SINGLE_BANDS.index(feature_set)]]
    if feature_set == AVERAGE_PSD_FEATURE:
        return average_psd
    if feature_set == ALL_BAND_FEATURE:
        return band_power
    if feature_set == TORUS_FEATURE:
        return torus
    raise ValueError(f"Unknown feature set: {feature_set}")


def session_hour_label(session_id: str) -> str:
    try:
        start_minute = int(session_id.split("_")[1].removeprefix("m"))
    except (IndexError, ValueError):
        return session_id
    return f"Hour {start_minute // 60 + 1}"


def build_scores() -> pd.DataFrame:
    class_counts = pd.read_csv(TABLE_DIR / "session_class_counts.csv")
    valid_sessions = class_counts.loc[class_counts["status"] == "ok", "session_id"].tolist()
    rows: list[dict[str, object]] = []
    for session_id in valid_sessions:
        cache_path = CACHE_DIR / f"{session_id}.npz"
        if not cache_path.exists():
            continue
        with np.load(cache_path, allow_pickle=True) as data:
            labels = data["labels_bal"].astype(int)
            windows = data["windows_bal"].astype(float)
            band_power = data["band_bal"].astype(float)
            torus = data[TORUS_FEATURE].astype(float)
        average_psd = compute_average_psd(windows)
        for feature_set in FEATURE_ORDER:
            x = feature_matrix(feature_set, band_power, average_psd, torus)
            _pred, acc, macro_f1, f1_vals, _cm = decode_features(x, labels)
            rows.append(
                {
                    "session_id": session_id,
                    "recording_hour": session_hour_label(session_id),
                    "feature_set": feature_set,
                    "accuracy": acc,
                    "f1": macro_f1,
                    "f1_wake": f1_vals[0],
                    "f1_nrem": f1_vals[1],
                    "f1_rem": f1_vals[2],
                }
            )
    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("feature_set", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            n_sessions=("session_id", "nunique"),
        )
        .set_index("feature_set")
        .reindex(FEATURE_ORDER)
        .reset_index()
    )


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


def best_band_comparison(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    band_means = scores[scores["feature_set"].isin(SINGLE_BANDS)].groupby("feature_set")["f1"].mean().reindex(SINGLE_BANDS)
    best_band = str(band_means.idxmax())
    selection = pd.DataFrame(
        [
            {
                "analysis_label": "Mouse EEG valid sessions",
                "n_sessions": int(scores["session_id"].nunique()),
                "selected_band": best_band,
                "selected_band_label": FEATURE_LABELS_REVIEW[best_band].replace("\n", " "),
                "selected_band_mean_f1": float(band_means.loc[best_band]),
            }
        ]
    )
    wide = (
        scores[scores["feature_set"].isin([TORUS_FEATURE, AVERAGE_PSD_FEATURE, best_band])]
        .pivot_table(index=["session_id", "recording_hour"], columns="feature_set", values="f1", aggfunc="mean")
        .dropna(subset=[TORUS_FEATURE, AVERAGE_PSD_FEATURE, best_band])
        .reset_index()
    )
    role_map = [
        ("torus_features", "Torus features", TORUS_FEATURE),
        ("average_psd", "Average PSD", AVERAGE_PSD_FEATURE),
        ("best_single_band", "Best single band", best_band),
    ]
    rows = []
    for _, row in wide.iterrows():
        for role, label, feature_set in role_map:
            rows.append(
                {
                    "session_id": row["session_id"],
                    "recording_hour": row["recording_hour"],
                    "feature_role": role,
                    "feature_label": label,
                    "feature_set": feature_set,
                    "selected_band": best_band,
                    "selected_band_label": FEATURE_LABELS_REVIEW[best_band].replace("\n", " "),
                    "f1": float(row[feature_set]),
                }
            )
    comparison = pd.DataFrame(rows)
    significance = compute_significance(comparison)
    return comparison, selection, significance


def relevant_band_comparison(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relevant_band = RELEVANT_BAND
    selection = pd.DataFrame(
        [
            {
                "analysis_label": "Mouse EEG valid sessions",
                "n_sessions": int(scores["session_id"].nunique()),
                "relevant_band": relevant_band,
                "relevant_band_label": FEATURE_LABELS_REVIEW[relevant_band].replace("\n", " "),
            }
        ]
    )
    wide = (
        scores[scores["feature_set"].isin([TORUS_FEATURE, AVERAGE_PSD_FEATURE, relevant_band])]
        .pivot_table(index=["session_id", "recording_hour"], columns="feature_set", values="f1", aggfunc="mean")
        .dropna(subset=[TORUS_FEATURE, AVERAGE_PSD_FEATURE, relevant_band])
        .reset_index()
    )
    role_map = [
        ("torus_features", "Torus features", TORUS_FEATURE),
        ("average_psd", "Average PSD", AVERAGE_PSD_FEATURE),
        ("relevant_band", "Relevant band", relevant_band),
    ]
    rows = []
    for _, row in wide.iterrows():
        for role, label, feature_set in role_map:
            rows.append(
                {
                    "session_id": row["session_id"],
                    "recording_hour": row["recording_hour"],
                    "feature_role": role,
                    "feature_label": label,
                    "feature_set": feature_set,
                    "relevant_band": relevant_band,
                    "relevant_band_label": FEATURE_LABELS_REVIEW[relevant_band].replace("\n", " "),
                    "f1": float(row[feature_set]),
                }
            )
    comparison = pd.DataFrame(rows)
    significance = compute_significance(
        comparison,
        [
            ("torus_features", "average_psd"),
            ("torus_features", "relevant_band"),
            ("relevant_band", "average_psd"),
        ],
    )
    return comparison, selection, significance


def summarize_relevant_band_bars(comparison: pd.DataFrame) -> pd.DataFrame:
    role_order = ["torus_features", "average_psd", "relevant_band"]
    summary = (
        comparison.groupby(["feature_role", "feature_label"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            n_sessions=("session_id", "nunique"),
        )
    )
    summary["role_order"] = summary["feature_role"].map({role: i for i, role in enumerate(role_order)})
    return summary.sort_values("role_order").drop(columns=["role_order"]).reset_index(drop=True)


def compute_significance(
    comparison: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    wide = comparison.pivot_table(index="session_id", columns="feature_role", values="f1", aggfunc="mean")
    if pairs is None:
        pairs = [
            ("torus_features", "average_psd"),
            ("torus_features", "best_single_band"),
            ("best_single_band", "average_psd"),
        ]
    rows = []
    for left, right in pairs:
        paired = wide[[left, right]].dropna()
        statistic = np.nan
        p_value = np.nan
        if len(paired) >= 5:
            try:
                statistic, p_value = stats.wilcoxon(paired[left], paired[right], zero_method="wilcox", alternative="two-sided")
            except ValueError:
                statistic, p_value = np.nan, 1.0
        rows.append(
            {
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
    corrected = holm_correct([row["p_uncorrected"] for row in rows])
    for row, p_holm in zip(rows, corrected):
        row["p_holm"] = p_holm
        row["significance"] = star_label(p_holm)
    return pd.DataFrame(rows)


def plot_bar(summary: pd.DataFrame, suffix: str | None) -> None:
    sub = summary.set_index("feature_set").reindex(FEATURE_ORDER).dropna(subset=["mean_f1"])
    means = sub["mean_f1"].to_numpy(float)
    stds = sub["std_f1"].fillna(0).to_numpy(float)
    norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = PAPER_RED_CMAP(norm(means))
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
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
    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, [FEATURE_LABELS_REVIEW[idx] for idx in sub.index])
    ax.set_ylim(0, min(1.0, max(0.82, float(np.nanmax(means + stds)) + 0.08)))
    ax.set_ylabel("F1")
    ax.set_title("Sleep-Stage Decoding From Mouse EEG", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.018, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path("eeg_sleep_feature_f1_barplot.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_box(comparison: pd.DataFrame, selection: pd.DataFrame, significance: pd.DataFrame, suffix: str | None) -> None:
    roles = ["torus_features", "average_psd", "best_single_band"]
    selected_label = str(selection.iloc[0]["selected_band_label"])
    labels = ["Torus\nfeatures", "Average\nPSD", f"Best band\n{selected_label.replace(' ', chr(10), 1)}"]
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

    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(np.arange(1, 4), labels)
    ax.set_ylabel("F1")
    ax.set_title("Sleep-Stage Decoding", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pair_positions = {
        "torus_features vs average_psd": (1, 2, 0.83),
        "torus_features vs best_single_band": (1, 3, 0.91),
        "best_single_band vs average_psd": (3, 2, 0.75),
    }
    sig_lookup = significance.set_index("comparison")
    for comparison_name, (x1, x2, y) in pair_positions.items():
        text = str(sig_lookup.loc[comparison_name, "significance"]) if comparison_name in sig_lookup.index else "n/a"
        h = 0.025
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path("eeg_sleep_torus_avgpsd_bestband_f1_boxplot.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_relevant_band_bar(
    bar_summary: pd.DataFrame,
    significance: pd.DataFrame,
    suffix: str | None,
) -> None:
    roles = ["torus_features", "average_psd", "relevant_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", "Relevant band\nDelta (0.5-4 Hz)"]
    colors = ["#8b0000", "#6f6f6f", "#D85A30"]
    sub = bar_summary.set_index("feature_role").reindex(roles)
    means = sub["mean_f1"].to_numpy(float)
    stds = sub["std_f1"].fillna(0).to_numpy(float)
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
    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("F1")
    ax.set_title("Sleep-Stage Decoding", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.018, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    pair_positions = {
        "torus_features vs average_psd": (0, 1, 0.83),
        "torus_features vs relevant_band": (0, 2, 0.91),
        "relevant_band vs average_psd": (2, 1, 0.75),
    }
    sig_lookup = significance.set_index("comparison")
    for comparison_name, (x1, x2, y) in pair_positions.items():
        text = str(sig_lookup.loc[comparison_name, "significance"]) if comparison_name in sig_lookup.index else "n/a"
        h = 0.025
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path("eeg_sleep_torus_avgpsd_relevantband_f1_barplot.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_relevant_band_box(comparison: pd.DataFrame, significance: pd.DataFrame, suffix: str | None) -> None:
    roles = ["torus_features", "average_psd", "relevant_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", "Relevant band\nDelta (0.5-4 Hz)"]
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

    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(np.arange(1, 4), labels)
    ax.set_ylabel("F1")
    ax.set_title("Sleep-Stage Decoding", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pair_positions = {
        "torus_features vs average_psd": (1, 2, 0.83),
        "torus_features vs relevant_band": (1, 3, 0.91),
        "relevant_band vs average_psd": (3, 2, 0.75),
    }
    sig_lookup = significance.set_index("comparison")
    for comparison_name, (x1, x2, y) in pair_positions.items():
        text = str(sig_lookup.loc[comparison_name, "significance"]) if comparison_name in sig_lookup.index else "n/a"
        h = 0.025
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path("eeg_sleep_torus_avgpsd_relevantband_f1_boxplot.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    (PLOT_DIR / "summary").mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scores = build_scores()
    summary = summarize_scores(scores)
    comparison, selection, significance = best_band_comparison(scores)
    relevant_comparison, relevant_selection, relevant_significance = relevant_band_comparison(scores)
    relevant_bar_summary = summarize_relevant_band_bars(relevant_comparison)

    scores.to_csv(table_path("eeg_reviewer_feature_scores.csv", args.output_suffix), index=False)
    summary.to_csv(table_path("eeg_reviewer_feature_summary.csv", args.output_suffix), index=False)
    comparison.to_csv(table_path("eeg_reviewer_torus_avgpsd_bestband_f1_scores.csv", args.output_suffix), index=False)
    selection.to_csv(table_path("eeg_reviewer_bestband_selection.csv", args.output_suffix), index=False)
    significance.to_csv(table_path("eeg_reviewer_significance.csv", args.output_suffix), index=False)
    relevant_comparison.to_csv(table_path("eeg_reviewer_torus_avgpsd_relevantband_f1_scores.csv", args.output_suffix), index=False)
    relevant_selection.to_csv(table_path("eeg_reviewer_relevantband_selection.csv", args.output_suffix), index=False)
    relevant_significance.to_csv(table_path("eeg_reviewer_relevantband_significance.csv", args.output_suffix), index=False)
    relevant_bar_summary.to_csv(table_path("eeg_reviewer_torus_avgpsd_relevantband_f1_bar_summary.csv", args.output_suffix), index=False)

    plot_bar(summary, args.output_suffix)
    plot_box(comparison, selection, significance, args.output_suffix)
    plot_relevant_band_bar(relevant_bar_summary, relevant_significance, args.output_suffix)
    plot_relevant_band_box(relevant_comparison, relevant_significance, args.output_suffix)

    print(summary.to_string(index=False))
    print()
    print(significance.to_string(index=False))
    print()
    print(relevant_significance.to_string(index=False))


if __name__ == "__main__":
    main()
