"""Plot only frozen F1 and recall confusion from the six-stage tau sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent / "outputs/tau_sweep_v1"
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
INK = "#263238"
CMAP = LinearSegmentedColormap.from_list("stage_f1", ["#fff9f6", "#e9a18d", "#941f1a"])


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.grid": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(match=matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                             box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"Clipped labels in {path.name}: {outside}")
    fig.savefig(path, dpi=600)
    plt.close(fig)
    with Image.open(path) as image:
        return dict(filename=path.name, pixels=image.size, dpi=image.info.get("dpi"))


def f1_matrix(root, variant, overall, stage, winner):
    rows = []
    for name in STAGES:
        sub = stage[(stage.variant == variant) & (stage.stage_name == name)]
        rows.append(sub.set_index("tau_ms").reindex(range(1, 21)).f1.to_numpy())
    rows.append(overall[overall.variant == variant].set_index("tau_ms")
                .reindex(range(1, 21)).macro_f1.to_numpy())
    values = np.array(rows, dtype=float)
    if values.shape != (7, 20):
        raise ValueError("Wrong F1 matrix shape")
    fig, ax = plt.subplots(figsize=(11.7, 4.35))
    im = ax.imshow(values, cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(20), np.arange(1, 21))
    ax.set_yticks(np.arange(7), [*LABELS, "Macro-F1"])
    ax.set_xlabel("Lag delay (ms)", labelpad=8)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-.5, 20), minor=True)
    ax.set_yticks(np.arange(-.5, 7), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(7):
        for j in range(20):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.4,
                        color="white" if value > .58 else INK)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color=INK)
    # Baseline and observed winner are marked without altering their measured colors.
    from matplotlib.patches import Rectangle
    for tau, color, width in ((3, "#376d9b", 2.1), (winner, "#971f1a", 2.5)):
        ax.add_patch(Rectangle((tau - 1.5, -.5), 1, 7, fill=False,
                               edgecolor=color, linewidth=width, clip_on=False))
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.023, pad=.02, ticks=[0, .5, 1])
    cbar.ax.set_ylabel("F1", rotation=270, labelpad=13)
    cbar.ax.tick_params(length=2)
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=.12, right=.94, top=.96, bottom=.16)
    return fig, values


def recall_matrix(prediction_file):
    with np.load(prediction_file) as saved:
        y, p = saved["labels"], saved["predictions"]
    if y.shape != (1188,) or p.shape != (1188,) or not np.isin(y, range(6)).all() or not np.isin(p, range(6)).all():
        raise ValueError("Invalid saved held-out predictions")
    counts = np.bincount(y * 6 + p, minlength=36).reshape(6, 6)
    fractions = counts / counts.sum(axis=1, keepdims=True)
    if not np.allclose(fractions.sum(axis=1), 1):
        raise ValueError("Confusion rows are not normalized")
    return fractions


def confusion_figure(matrix):
    fig, ax = plt.subplots(figsize=(5.15, 4.45))
    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    for i in range(6):
        for j in range(6):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", va="center", ha="center", fontsize=8.3,
                    color="white" if value > .58 else INK)
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(6), LABELS)
    ax.set_xlabel("Predicted stage", labelpad=6)
    ax.set_ylabel("True stage", labelpad=6)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.025, ticks=[0, .5, 1])
    cbar.ax.set_ylabel("Prediction fraction", rotation=270, labelpad=13)
    cbar.ax.tick_params(length=2)
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=.16, right=.91, top=.96, bottom=.18)
    return fig


def render(root=ROOT):
    root = Path(root).resolve()
    if root != ROOT.resolve() or not (root / "validation.json").exists():
        raise ValueError("Plotting requires a validated frozen tau sweep")
    overall = pd.read_csv(root / "tables/overall_f1.csv")
    stage = pd.read_csv(root / "tables/stage_f1.csv")
    winners = pd.read_csv(root / "tables/winners.csv")
    style()
    exports = []
    for variant in ("native", "matched260"):
        selected = winners[winners.variant == variant]
        if len(selected) != 1:
            raise ValueError(f"Missing winner for {variant}")
        tau = int(selected.iloc[0].tau_ms)
        figure, values = f1_matrix(root, variant, overall, stage, tau)
        np.testing.assert_allclose(values[-1, tau-1], selected.iloc[0].macro_f1)
        exports.append(save(figure, root / "figures" / f"f1_matrix_{variant}.png"))
        fractions = recall_matrix(root / "predictions" / f"{variant}_tau_{tau:02d}.npz")
        expected = pd.read_csv(root / "tables" / f"confusion_recall_{variant}_tau_{tau:02d}.csv",
                               index_col=0).to_numpy()
        np.testing.assert_allclose(fractions, expected, atol=1e-12)
        exports.append(save(confusion_figure(fractions),
                            root / "figures" / f"recall_confusion_{variant}.png"))
    (root / "README.md").write_text(
        "# Six-stage tau sweep\n\n"
        "The four PNGs are in `figures/`: an F1 matrix and winner's recall confusion "
        "for each of the native and matched-260 sweeps. Blue outlines mark the earlier "
        "tau=3 setting; red outlines mark the highest observed macro-F1. "
        "The winner was selected from these held-out scores and is exploratory, not an "
        "independently validated optimum. Confusion diagonals are class recall, not F1. "
        "All per-tau predictions, fitted parameters, and failures are saved in the other folders.\n")
    (root / "figure_validation.json").write_text(json.dumps(exports, indent=2) + "\n")
    return exports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for result in render():
        print(result)
