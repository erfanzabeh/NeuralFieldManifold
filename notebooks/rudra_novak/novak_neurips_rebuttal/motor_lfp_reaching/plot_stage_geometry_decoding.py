"""Render only saved stage clouds, fit parameters, features, and held-out results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, STAGES, LABELS, COLORS, COLUMNS, scores

DISPLAY = dict(R1="Outer radius R1", R2="Outer radius R2", band_half_width="Band half-width",
               mse="Mean squared fit error", mean_error="Mean in-plane error", frac_inside="Fraction inside")
INK = "#263238"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
                         "axes.titlesize": 10, "axes.titleweight": "normal", "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "axes.edgecolor": INK, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, folder, name, exports):
    fig.canvas.draw()
    # Store pixel dimensions and artists outside the canvas for a reproducible layout audit.
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width > 0 and bbox.height > 0 and (
                bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1):
            outside.append(artist.get_text())
    path = folder/f"{name}.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    with Image.open(path) as im:
        dimensions = im.size
        dpi = im.info.get("dpi")
    exports.append(dict(file=path.name, pixels=dimensions, dpi=dpi, outside_text=outside, sha256=sha256(path)))


def distribution_axis(ax, table, column, component=None):
    rng = np.random.default_rng(891)
    plotted = []
    for s, color in enumerate(COLORS):
        sub = table[(table.stage == s) & table.usable]
        values = sub[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        values, sub = values[finite], sub.loc[finite]
        jitter = rng.uniform(-.23, .23, len(values))
        ax.scatter(s+jitter, values, s=7, alpha=.35, c=color, edgecolors="none", zorder=2)
        if len(values):
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            ax.plot([s, s], [q1, q3], color=INK, lw=2.2, zorder=3)
            ax.scatter(s, median, s=20, facecolor="white", edgecolor=INK, linewidth=.8, zorder=4)
        for row, value, j in zip(sub.itertuples(), values, jitter):
            plotted.append(dict(window_index=row.window_index, stage_name=row.stage_name,
                                feature=column, value=value, plotted_x=s+j))
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
    ax.set_xlim(-.55, 5.55)
    ax.tick_params(length=3, pad=3)
    if component:
        ax.set_ylim(-1.07, 1.07)
        ax.set_yticks([-1, 0, 1])
        ax.set_title(component)
    else:
        ax.set_ylabel(DISPLAY[column])
        ax.set_ylim(bottom=0)
        if column == "frac_inside":
            ax.set_ylim(-.03, 1.04)
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="upper"))
    return plotted


def feature_figures(root, table, exports):
    folder = root/"png"
    rows = []
    for column in ("R1", "R2", "band_half_width", "mse", "mean_error", "frac_inside"):
        fig, ax = plt.subplots(figsize=(4.3, 3.05))
        rows.extend(distribution_axis(ax, table, column))
        fig.subplots_adjust(left=.19, bottom=.23, right=.97, top=.96)
        save(fig, folder, f"distribution_{column}", exports)
    for prefix, name in (("normal", "Plane normal"), ("u", "Major-axis vector"), ("v", "Minor-axis vector")):
        fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.9), sharey=True)
        for axis, component in zip(axes, "xyz"):
            rows.extend(distribution_axis(axis, table, f"{prefix}_{component}", component=f"{name}: {component}"))
        axes[0].set_ylabel("Orientation component")
        fig.subplots_adjust(left=.07, right=.985, bottom=.26, top=.85, wspace=.17)
        save(fig, folder, f"orientation_{prefix}", exports)
    pd.DataFrame(rows).to_csv(root/"tables/plotted_feature_points.csv", index=False)


def outlines(result):
    fit = result["fit"]
    t = np.linspace(0, 2*np.pi, 241)
    center, u, v = [np.asarray(fit[k], dtype=float) for k in ("center", "u_axis", "v_axis")]
    return [center + (a*np.cos(t))[:, None]*u + (b*np.sin(t))[:, None]*v
            for a, b in ((fit["R1"], fit["R2"]), (fit["R1_in"], fit["R2_in"]))]


def cloud_axis(ax, cloud, result, stage, limit, overlay):
    ax.plot(*cloud.T, lw=.45, color=COLORS[stage], alpha=.6)
    ax.scatter(*cloud.T, s=2.5, c=COLORS[stage], alpha=.85, depthshade=False, linewidths=0)
    if overlay and result["usable"]:
        for line, ls in zip(outlines(result), ("-", "--")):
            ax.plot(*line.T, color=INK, lw=.9, ls=ls, alpha=.9)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3, prune="both"))
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
    ax.grid(False)
    ax.tick_params(labelsize=8, pad=0)
    ax.set_xlabel("x(t)", labelpad=0, fontsize=9)
    ax.set_ylabel("x(t - 3 ms)", labelpad=0, fontsize=9)
    ax.set_zlabel("x(t - 6 ms)", labelpad=1, fontsize=9)
    ax.set_title(LABELS[stage], pad=1)


def cloud_figures(root, table, exports):
    with np.load(root/"inputs/windows.npz") as z:
        clouds = z["clouds"]
    selected = table[table.original_trial_number.isin([6, 9, 10])]
    results = {}
    extent = [np.max(np.abs(clouds[selected.window_index]))]
    for row in selected.itertuples():
        r = json.loads((root/"checkpoints"/f"window_{row.window_index:04d}.json").read_text())
        r["usable"] = bool(row.usable)
        results[row.window_index] = r
        if r["usable"]:
            extent.append(np.max(np.abs(outlines(r))))
    limit = float(max(extent)*1.08)
    pd.DataFrame(dict(axis=["x", "y", "z"], minimum=-limit, maximum=limit)).to_csv(
        root/"tables/example_geometry_scales.csv", index=False)
    selected[["window_index", "original_trial_number", "stage_name", "usable"]].to_csv(
        root/"tables/example_selection.csv", index=False)
    for trial in (6, 9, 10):
        example = selected[selected.original_trial_number == trial].sort_values("stage")
        for overlay in (False, True):
            suffix = "fitted" if overlay else "observed"
            fig = plt.figure(figsize=(8.4, 6.05))
            for row in example.itertuples():
                # Event columns; pre/post event occupy the upper/lower row.
                position = (row.stage % 2)*3 + row.stage//2 + 1
                ax = fig.add_subplot(2, 3, position, projection="3d")
                cloud_axis(ax, clouds[row.window_index], results[row.window_index], row.stage, limit, overlay)
            fig.subplots_adjust(left=.015, right=.965, top=.94, bottom=.08, wspace=.03, hspace=.29)
            save(fig, root/"png", f"geometry_trial_{trial}_{suffix}", exports)
            for row in example.itertuples():
                fig = plt.figure(figsize=(3.0, 2.9))
                ax = fig.add_subplot(projection="3d")
                cloud_axis(ax, clouds[row.window_index], results[row.window_index], row.stage, limit, overlay)
                fig.subplots_adjust(left=.03, right=.87, top=.91, bottom=.13)
                save(fig, root/"png", f"geometry_trial_{trial}_{row.stage_name}_{suffix}", exports)


def decoding_figures(root, exports):
    with np.load(root/"heldout_predictions.npz") as z:
        computed = scores(z["labels"], z["predictions"])
    matrix = pd.read_csv(root/"tables/confusion_fraction.csv", index_col=0).to_numpy()
    np.testing.assert_allclose(matrix, computed["confusion_fraction"])
    fig, ax = plt.subplots(figsize=(4.1, 3.65))
    cmap = LinearSegmentedColormap.from_list("stage_recall", ["#FFF9F6", "#E9A18D", "#941F1A"])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > .6 else INK)
    ax.set_xticks(range(6), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(6), LABELS)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which="minor", color="white", lw=.8)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=.04, pad=.045, ticks=[0, .5, 1])
    cbar.set_label("Prediction fraction", fontsize=9)
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=.20, bottom=.25, right=.84, top=.98)
    save(fig, root/"png", "stage_confusion", exports)

    perf = pd.read_csv(root/"tables/per_stage_f1.csv")
    np.testing.assert_allclose(perf.f1, computed["per_stage_f1"])
    fig, ax = plt.subplots(figsize=(4.3, 3.05))
    for i, row in enumerate(perf.itertuples()):
        ax.plot([i, i], [row.ci_low, row.ci_high], color=INK, lw=1.2, zorder=2)
        ax.scatter(i, row.f1, s=38, c=COLORS[i], edgecolor=INK, linewidth=.5, zorder=3)
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
    ax.set_xlim(-.55, 5.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-stage F1")
    ax.set_yticks([0, .25, .5, .75, 1])
    fig.subplots_adjust(left=.16, bottom=.23, right=.97, top=.96)
    save(fig, root/"png", "stage_f1", exports)


def captions(root):
    text = """# Panel captions

Geometry: Original trials 6, 9, and 10 were selected by increasing original trial ID, independently of fit quality or classification. Every cloud contains 294 observed delay vectors from its 300-ms window, with m=3 and tau=3 ms. All examples use identical original-coordinate limits and viewpoints. Fitted panels overlay the outer solid and inner dashed annular boundaries only for usable fits; these outlines do not prove topology. No projection or dimensionality reduction is applied.

Feature distributions: Each dot is one usable trial-window fit. White-centered markers show medians; thick vertical segments show interquartile ranges. Stage colors are fixed, not assigned by performance. Radii and band half-width are in normalized signal units. Band half-width is (R1_out-R1_in+R2_out-R2_in)/4, not R1-R2. MSE has squared normalized units; mean error has normalized units; fraction inside is unitless. Unusable fits are absent only from descriptive distributions; decoding retains their windows using training-only imputation. Counts are in fit_accounting.csv.

Orientation: Components refer to the lag-coordinate system, not anatomical or reaching directions. Normals and ellipse axes use the same canonical ordering/sign convention as the prior decoder. Signed components may be discontinuous at canonical sign boundaries, and nearly circular fits have uncertain in-plane axes; no claim of stage separation should be based on these conventions alone.

Confusion: Row-normalized held-out prediction fractions from five folds, with all six stages of each original trial kept together. Diagonal values are stage recall, not F1. Each row contains 198 windows before normalization. The color scale is fixed to 0-1.

Per-stage F1: Pooled held-out per-class F1, with 95% conditional intervals from 2,000 whole-trial bootstrap resamples stratified by original reach direction. Every resampled trial contributes all six windows. Folds and windows are not independent experimental replicates. Overall macro-F1, accuracy, and the within-trial stage-label permutation test are in report.md.

Event windows: Pre is [-300,0) ms and post is [0,300) ms relative to actual recorded TC onset, SC onset, or GO. Post-TC/SC include subsequent delay activity; post-GO is a reaction/movement window rather than a pure reaching epoch. Filtering is zero-phase and local to each short window, so this is offline classification with potential edge effects, not causal online forecasting.
"""
    (root/"captions.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--panels", choices=("geometry", "features", "decoding", "all"), default="all")
    args = parser.parse_args()
    root = args.output.resolve()
    if root != OUTPUT.resolve():
        raise ValueError("Plot only the named stage experiment")
    (root/"png").mkdir(exist_ok=True)
    style()
    table = pd.read_csv(root/"tables/features.csv")
    assert table[COLUMNS].shape == (1188, 15)
    exports = []
    if args.panels in ("features", "all"):
        feature_figures(root, table, exports)
    if args.panels in ("geometry", "all"):
        cloud_figures(root, table, exports)
    if args.panels in ("decoding", "all"):
        decoding_figures(root, exports)
    captions(root)
    shutil.copy2(__file__, root/"source"/Path(__file__).name)
    write_json(root/f"plot_manifest_{args.panels}.json", dict(plot_source_sha256=sha256(__file__), exports=exports))
    print(json.dumps(dict(png_count=len(exports), files_with_outside_text=[e for e in exports if e["outside_text"]]), indent=2))


if __name__ == "__main__":
    main()
