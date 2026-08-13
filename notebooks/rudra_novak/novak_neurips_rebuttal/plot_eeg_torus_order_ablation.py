#!/usr/bin/env python
"""Compare full torus EEG decoding against a no-r geometry ablation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from run_sleep_decoding import CACHE_DIR, PLOT_DIR, RANDOM_SEED, TABLE_DIR, decode_features


FULL_TORUS_FEATURE = "torus_features_15d"
NO_R_FEATURE = "no_r_geometry_11d"
TORUS_CACHE_KEY = "all_torus_15"

FULL_TORUS_NAMES = [
    "R1",
    "R2",
    "minor_radius",
    "mse",
    "mean_error",
    "frac_inside",
    "direction_x",
    "direction_y",
    "direction_z",
    "u_axis_x",
    "u_axis_y",
    "u_axis_z",
    "v_axis_x",
    "v_axis_y",
    "v_axis_z",
]
NO_R_COLUMNS = [0, 1, *range(6, 15)]


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    feature_names: list[str]


FEATURE_SPECS = {
    FULL_TORUS_FEATURE: FeatureSpec(
        label="Torus features (15D)",
        feature_names=FULL_TORUS_NAMES,
    ),
    NO_R_FEATURE: FeatureSpec(
        label="No-r geometry (11D)",
        feature_names=[FULL_TORUS_NAMES[i] for i in NO_R_COLUMNS],
    ),
}
FEATURE_ORDER = [FULL_TORUS_FEATURE, NO_R_FEATURE]


def suffix_token(suffix: str | None) -> str:
    clean = str(suffix or "").strip().strip("_")
    return f"_{clean}" if clean else ""


def table_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def session_hour_label(session_id: str) -> str:
    try:
        start_minute = int(session_id.split("_")[1].removeprefix("m"))
    except (IndexError, ValueError):
        return session_id
    return f"Hour {start_minute // 60 + 1}"


def select_feature_matrix(feature_set: str, all_torus: np.ndarray) -> np.ndarray:
    if all_torus.ndim != 2 or all_torus.shape[1] != len(FULL_TORUS_NAMES):
        raise ValueError(f"Expected all_torus shape (n_samples, 15); got {all_torus.shape}")
    if feature_set == FULL_TORUS_FEATURE:
        return all_torus
    if feature_set == NO_R_FEATURE:
        return all_torus[:, NO_R_COLUMNS]
    raise ValueError(f"Unknown feature set: {feature_set}")


def valid_session_ids() -> list[str]:
    class_counts_path = TABLE_DIR / "session_class_counts.csv"
    if class_counts_path.exists():
        class_counts = pd.read_csv(class_counts_path)
        sessions = class_counts.loc[class_counts["status"] == "ok", "session_id"].astype(str).tolist()
    else:
        sessions = [path.stem for path in sorted(CACHE_DIR.glob("session_*.npz"))]
    return sorted(sessions, key=lambda s: int(s.split("_")[1].removeprefix("m")) if "_m" in s else s)


def build_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_id in valid_session_ids():
        cache_path = CACHE_DIR / f"{session_id}.npz"
        if not cache_path.exists():
            continue
        with np.load(cache_path, allow_pickle=True) as data:
            labels = data["labels_bal"].astype(int)
            all_torus = data[TORUS_CACHE_KEY].astype(float)

        for feature_set in FEATURE_ORDER:
            features = select_feature_matrix(feature_set, all_torus)
            _pred, accuracy, f1, f1_values, _cm = decode_features(features, labels)
            rows.append(
                {
                    "session_id": session_id,
                    "recording_hour": session_hour_label(session_id),
                    "feature_set": feature_set,
                    "feature_label": FEATURE_SPECS[feature_set].label,
                    "n_features": features.shape[1],
                    "accuracy": accuracy,
                    "f1": f1,
                    "f1_wake": f1_values[0],
                    "f1_nrem": f1_values[1],
                    "f1_rem": f1_values[2],
                    "random_seed": RANDOM_SEED,
                }
            )
    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scores.groupby(["feature_set", "feature_label", "n_features"], as_index=False)
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
    return summary


def make_reporting_table(summary: pd.DataFrame, significance: pd.DataFrame) -> pd.DataFrame:
    sig_row = significance.iloc[0] if not significance.empty else pd.Series(dtype=object)
    p_value = sig_row.get("p_value", np.nan)
    sig = sig_row.get("significance", "n/a")
    paired_test = f"paired Wilcoxon p={p_value:.3g} ({sig})" if np.isfinite(p_value) else "paired Wilcoxon n/a"
    scenario_labels = {
        FULL_TORUS_FEATURE: "Correct order encoding",
        NO_R_FEATURE: "Incorrect order decoding",
    }
    report = summary.set_index("feature_set").reindex(FEATURE_ORDER).reset_index()
    report["scenario"] = report["feature_set"].map(scenario_labels)
    report["f1_mean_sd"] = report.apply(lambda row: f"{row['mean_f1']:.3f} +/- {row['std_f1']:.3f} SD", axis=1)
    report["paired_test"] = ["", ""][: len(report)]
    if len(report):
        report.loc[0, "paired_test"] = paired_test
    return report[
        [
            "scenario",
            "f1_mean_sd",
            "paired_test",
        ]
    ]


def write_markdown_table(report: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Mouse EEG Torus Order-Ablation F1 Table",
        "",
        "<table>",
        "  <thead>",
        "    <tr><th>Scenario</th><th>F1</th><th>Paired test</th></tr>",
        "  </thead>",
        "  <tbody>",
    ]
    shared_test = str(report["paired_test"].replace("", np.nan).dropna().iloc[0]) if report["paired_test"].ne("").any() else ""
    for i, row in report.iterrows():
        if i == 0:
            lines.append(
                f"    <tr><td>{row['scenario']}</td><td>{row['f1_mean_sd']}</td>"
                f"<td rowspan=\"{len(report)}\">{shared_test}</td></tr>"
            )
        else:
            lines.append(f"    <tr><td>{row['scenario']}</td><td>{row['f1_mean_sd']}</td></tr>")
    lines.extend(["  </tbody>", "</table>"])
    path.write_text("\n".join(lines) + "\n")


def compute_significance(scores: pd.DataFrame) -> pd.DataFrame:
    wide = scores.pivot_table(index="session_id", columns="feature_set", values="f1", aggfunc="mean")
    paired = wide[[FULL_TORUS_FEATURE, NO_R_FEATURE]].dropna()
    statistic = np.nan
    p_value = np.nan
    if len(paired) >= 5:
        try:
            statistic, p_value = stats.wilcoxon(
                paired[FULL_TORUS_FEATURE],
                paired[NO_R_FEATURE],
                zero_method="wilcox",
                alternative="two-sided",
            )
        except ValueError:
            statistic, p_value = np.nan, 1.0

    return pd.DataFrame(
        [
            {
                "comparison": "torus_features_15d vs no_r_geometry_11d",
                "feature_left": FULL_TORUS_FEATURE,
                "feature_right": NO_R_FEATURE,
                "n_pairs": int(len(paired)),
                "mean_left_f1": float(paired[FULL_TORUS_FEATURE].mean()) if len(paired) else np.nan,
                "mean_right_f1": float(paired[NO_R_FEATURE].mean()) if len(paired) else np.nan,
                "mean_difference_left_minus_right": float((paired[FULL_TORUS_FEATURE] - paired[NO_R_FEATURE]).mean())
                if len(paired)
                else np.nan,
                "wilcoxon_statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "significance": star_label(p_value),
            }
        ]
    )


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


def plot_ablation(summary: pd.DataFrame, significance: pd.DataFrame, suffix: str | None) -> None:
    sub = summary.set_index("feature_set").reindex(FEATURE_ORDER)
    means = sub["mean_f1"].to_numpy(float)
    stds = sub["std_f1"].fillna(0).to_numpy(float)
    labels = sub["feature_label"].tolist()
    colors = ["#8b0000", "#D85A30"]
    x = np.arange(len(FEATURE_ORDER))

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        alpha=0.88,
        edgecolor="#201715",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "ecolor": "#201715", "capthick": 1.1},
    )
    ax.axhline(1 / 3, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("F1")
    ax.set_title("Mouse EEG Torus Feature Ablation", fontweight="bold")
    ymax = max(0.82, float(np.nanmax(means + stds)) + 0.14)
    ax.set_ylim(0, min(1.0, ymax))
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.018, f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

    y = float(np.nanmax(means + stds)) + 0.055
    h = 0.025
    sig = str(significance.iloc[0]["significance"]) if not significance.empty else "n/a"
    p_value = significance.iloc[0]["p_value"] if not significance.empty else np.nan
    p_text = f"{sig}\np={p_value:.2g}" if np.isfinite(p_value) else sig
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
    ax.text(0.5, y + h + 0.01, p_text, ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(plot_path("eeg_torus_order_ablation_f1_barplot.png", suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (PLOT_DIR / "summary").mkdir(parents=True, exist_ok=True)

    scores = build_scores()
    summary = summarize_scores(scores)
    significance = compute_significance(scores)
    report = make_reporting_table(summary, significance)

    scores.to_csv(table_path("eeg_torus_order_ablation_f1.csv", args.output_suffix), index=False)
    summary.to_csv(table_path("eeg_torus_order_ablation_summary.csv", args.output_suffix), index=False)
    significance.to_csv(table_path("eeg_torus_order_ablation_significance.csv", args.output_suffix), index=False)
    report.to_csv(table_path("eeg_torus_order_ablation_report_table.csv", args.output_suffix), index=False)
    write_markdown_table(report, table_path("eeg_torus_order_ablation_report_table.md", args.output_suffix))
    plot_ablation(summary, significance, args.output_suffix)

    print(report.to_string(index=False))
    print()
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()
