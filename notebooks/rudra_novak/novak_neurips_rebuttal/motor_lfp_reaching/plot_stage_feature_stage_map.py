"""Plot a descriptive stage-by-feature map from frozen corrected geometry features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image

import plot_stage_feature_projections as frozen

OUTPUT = frozen.UNIT / "outputs/stage_tau3_feature_stage_map_v1"
SOURCE = frozen.GEOMETRY / "tables/features.csv"
STAGE_LABELS = frozen.LABELS
FEATURES = frozen.RADII + frozen.ORIENTATION + frozen.QUALITY
FEATURE_LABELS = (
    r"$R_1$", r"$R_2$", "Half-width",
    r"$n_x$", r"$n_y$", r"$n_z$",
    r"$u_x$", r"$u_y$", r"$u_z$",
    r"$v_x$", r"$v_y$", r"$v_z$",
    "MSE", "Mean error", "Inside",
)
CMAP = LinearSegmentedColormap.from_list(
    "stage_shift", ("#285D89", "#F8F9F8", "#B34C39"), N=256)


def summarize(table):
    if (len(table) != 1188 or not np.array_equal(table.window_index, np.arange(1188))
            or not set(FEATURES).issubset(table.columns)):
        raise ValueError("Unexpected corrected geometry table")
    if (table.groupby("stage_name").size().reindex(frozen.STAGES).tolist() != [198] * 6
            or table.trial_index.nunique() != 198
            or table.loc[table.stage_name.eq("post_TC"), "anchor_code"].ne(203).any()
            or table.loc[table.stage_name.eq("post_SC"), "anchor_code"].ne(206).any()):
        raise ValueError("Unexpected corrected cue-offset stages")
    values = table[list(FEATURES)].astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite geometric feature")
    missing = values.isna().all(axis=1)
    if (missing.sum() != 11 or not np.array_equal(missing, ~table.usable.to_numpy())
            or values[~missing].isna().any().any()):
        raise ValueError("Geometric fit validity changed")
    center = values.median()
    scale = (values.quantile(.75) - values.quantile(.25)) / 1.349
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("One or more features have no robust spread")
    medians = table.groupby("stage_name")[list(FEATURES)].median().reindex(frozen.STAGES)
    shifts = medians.sub(center).div(scale)
    if not np.isfinite(shifts.to_numpy()).all():
        raise ValueError("Nonfinite stage shift")
    counts = table.groupby("stage_name").usable.sum().reindex(frozen.STAGES)
    return medians, shifts, center, scale, counts


def plot(shifts, destination):
    frozen.style()
    fig, ax = plt.subplots(figsize=(12.2, 5.1), facecolor="white")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    image = ax.imshow(shifts.to_numpy(), cmap=CMAP, norm=norm, aspect="auto",
                      interpolation="nearest")
    ax.set_xticks(np.arange(len(FEATURES)), FEATURE_LABELS, fontsize=9)
    ax.set_yticks(np.arange(len(STAGE_LABELS)), STAGE_LABELS, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10)
    for row in range(len(STAGE_LABELS)):
        for column in range(len(FEATURES)):
            value = float(shifts.iloc[row, column])
            label = "0.0" if abs(value) < .05 else f"{value:+.1f}"
            ax.text(column, row, label, ha="center", va="center",
                    fontsize=8.3, color="white" if abs(value) >= .57 else frozen.INK)
    for column in (2.5, 11.5):
        ax.axvline(column, color="#626E76", linewidth=1.0)
    for row in np.arange(.5, len(STAGE_LABELS), 1):
        ax.axhline(row, color="white", linewidth=.8, alpha=.85)
    ax.text(1, 1.20, "SIZE", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=frozen.INK,
            transform=ax.get_xaxis_transform())
    ax.text(7, 1.20, "ORIENTATION", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=frozen.INK,
            transform=ax.get_xaxis_transform())
    ax.text(13, 1.20, "FIT QUALITY", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=frozen.INK,
            transform=ax.get_xaxis_transform())
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-.5, len(FEATURES) - .5)
    ax.set_ylim(len(STAGE_LABELS) - .5, -.5)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=.065,
                            pad=.14, aspect=35, ticks=[-1, -.5, 0, .5, 1])
    colorbar.set_label("Stage median relative to pooled median (robust SD units)",
                       fontsize=10, color=frozen.INK, labelpad=7)
    colorbar.ax.tick_params(labelsize=9, colors=frozen.INK)
    colorbar.outline.set_visible(False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.69)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                                         or box.x1 > fig.bbox.width + 1
                                         or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Clipped text: {item.get_text()}")
    fig.savefig(destination, dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(destination) as saved:
        if saved.size != (7320, 3060) or not np.any(np.asarray(saved.convert("RGB")) < 240):
            raise ValueError("Blank or wrong-sized stage map")


def main():
    source_hash = frozen.digest(SOURCE)
    config_path = frozen.GEOMETRY / "config.json"
    config_hash = frozen.digest(config_path)
    config = json.loads(config_path.read_text())
    if (config["m"], config["tau_ms"], config["points_per_cloud"]) != (3, 3, 294):
        raise ValueError("Not native tau=3 geometry")
    if tuple(config["stage_anchors"]) != (
            "TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO"):
        raise ValueError("Not corrected cue-offset stages")
    table = pd.read_csv(SOURCE, float_precision="round_trip")
    medians, shifts, center, scale, counts = summarize(table)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT / "stage_feature_map.png"
    plot(shifts, image_path)
    rows = []
    for i, stage in enumerate(frozen.STAGES):
        for feature in FEATURES:
            rows.append({"stage": stage, "feature": feature,
                         "n_usable": int(counts.iloc[i]),
                         "stage_median": float(medians.loc[stage, feature]),
                         "pooled_median": float(center[feature]),
                         "pooled_robust_sd": float(scale[feature]),
                         "standardized_median_shift": float(shifts.loc[stage, feature])})
    pd.DataFrame(rows).to_csv(OUTPUT / "stage_feature_map.csv", index=False,
                              float_format="%.17g")
    (OUTPUT / "README.md").write_text(
        "# Corrected-stage geometric feature map\n\n"
        "Monkey T, session y070316009-12, channel 7; 198 short-delay trials, "
        "six 300-ms corrected cue-offset stages per trial, native m=3 and tau=3 ms. "
        "[Stage map](stage_feature_map.png) shows each stage's median shift from "
        "the pooled median, divided by the pooled IQR/1.349 for that feature. "
        "The same zero-centered color range (-1 to +1 robust SD) is used for every cell. "
        "Annotations are rounded to one decimal; exact values are in "
        "[the table](stage_feature_map.csv).\n\n"
        "Only valid existing fits contribute: stage counts are "
        + ", ".join(f"{name}={int(n)}" for name, n in zip(frozen.LABELS, counts))
        + ". No imputation, new torus fitting, Ripser, projection, or decoder was run. "
        "These are descriptive marginal shifts; they do not measure classification "
        "accuracy or establish independent orientation effects. Per-window signal "
        "normalization was already applied in the source analysis.\n")
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "source": str(SOURCE.resolve()), "source_sha256": source_hash,
        "source_config_sha256": config_hash, "figure_sha256": frozen.digest(image_path),
        "features": list(FEATURES), "stages": list(frozen.STAGES),
        "scaling": "(stage median - pooled median) / (pooled IQR / 1.349)",
        "n_windows": 1188, "n_usable": int(table.usable.sum())}, indent=2) + "\n")
    if source_hash != frozen.digest(SOURCE) or config_hash != frozen.digest(config_path):
        raise AssertionError("Frozen source changed")
    print(f"Saved {image_path}; source hashes unchanged")


if __name__ == "__main__":
    main()
