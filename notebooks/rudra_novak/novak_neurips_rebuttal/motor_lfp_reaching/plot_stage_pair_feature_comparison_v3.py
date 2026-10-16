"""Side-by-side corrected stage-pair plots with prespecified paired tests."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from plot_stage_pair_geometry import (
    COLLECTION, ORIENTATION, PAIRS, QUALITY, RADIAL, TITLES, UNIT,
    load, pair_table, sha256, style, summary_rows,
)
from plot_stage_pair_feature_distributions_v2 import common_bandwidth, kde_pdf


OUTPUT = COLLECTION / "stage_pair_feature_comparison_v3"
GROUPS = {
    "radii_width": RADIAL,
    "plane_normal": ORIENTATION[:3],
    "major_axis": ORIENTATION[3:6],
    "minor_axis": ORIENTATION[6:9],
    "fit_quality": QUALITY,
}
INK = "#263238"
EVENTS = ("SC", "GO")


def save(fig, stem, manifest):
    png, pdf = (OUTPUT / f"{stem}.{suffix}" for suffix in ("png", "pdf"))
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    with Image.open(png) as image:
        pixels = image.size
        if min(pixels) < 1000 or np.std(np.asarray(image.convert("RGB"))) < 5:
            raise ValueError(f"Blank or unexpectedly small figure: {png}")
    manifest.append({"stem": stem, "png_pixels": list(pixels),
                     "png_sha256": sha256(png), "pdf_sha256": sha256(pdf)})


def matched_features(table):
    pairs = {}
    for event, (before, after, _, _) in PAIRS.items():
        pair = pair_table(table, before, after)
        pair.insert(0, "comparison", f"{before}_vs_{after}")
        pairs[event] = pair
    return pairs


def test_all_features(pairs):
    rows = []
    for event in EVENTS:
        pair = pairs[event].loc[pairs[event].paired_usable]
        for feature in (*RADIAL, *ORIENTATION, *QUALITY):
            delta = pair[f"delta_{feature}"].to_numpy(dtype=float)
            if len(delta) < 2 or not np.isfinite(delta).all():
                raise ValueError(f"Invalid paired differences for {event} {feature}")
            if np.count_nonzero(delta) == 0:
                statistic, p_raw = 0., 1.
            else:
                result = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
                statistic, p_raw = float(result.statistic), float(result.pvalue)
            rows.append({"event": event, "feature": feature, "n_matched_trials": len(delta),
                         "n_nonzero_differences": int(np.count_nonzero(delta)),
                         "median_post_minus_pre": float(np.median(delta)),
                         "wilcoxon_statistic": statistic, "p_raw": p_raw})
    tested = pd.DataFrame(rows)
    reject, corrected, _, _ = multipletests(tested.p_raw.to_numpy(), alpha=.05, method="holm")
    tested["p_holm_30_tests"] = corrected
    tested["significant_holm_0_05"] = reject
    return tested


def p_label(value):
    if value < 1e-4:
        return "Holm p < 1e-4"
    if value < .01:
        return f"Holm p = {value:.4f}"
    return f"Holm p = {value:.3f}"


def figure_axes():
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 8.1), constrained_layout=True)
    return fig, axes


def same_feature_limits(pairs, feature, padding=.12):
    values = []
    for event in EVENTS:
        matched = pairs[event].loc[pairs[event].paired_usable]
        values.extend((matched[f"{feature}_pre"].to_numpy(dtype=float),
                       matched[f"{feature}_post"].to_numpy(dtype=float)))
    pooled = np.concatenate(values)
    low, high = float(pooled.min()), float(pooled.max())
    span = max(high - low, 1e-6)
    low, high = low - span * padding, high + span * padding
    if feature in (*RADIAL, "mse", "mean_error", "frac_inside"):
        low = max(0, low)
    if feature == "frac_inside":
        high = min(1.02, high)
    return low, high


def box_figure(group, features, pairs, tests, manifest):
    fig, axes = figure_axes()
    for row, feature in enumerate(features):
        limits = same_feature_limits(pairs, feature)
        for column, event in enumerate(EVENTS):
            ax = axes[row, column]
            matched = pairs[event].loc[pairs[event].paired_usable]
            values = [matched[f"{feature}_pre"].to_numpy(dtype=float),
                      matched[f"{feature}_post"].to_numpy(dtype=float)]
            before, after, pre_color, post_color = PAIRS[event]
            for position, (data, color) in enumerate(zip(values, (pre_color, post_color))):
                rng = np.random.default_rng(1380 + row * 4 + column * 2 + position)
                ax.scatter(position + rng.uniform(-.12, .12, len(data)), data,
                           s=6, alpha=.25, color=color, linewidths=0,
                           rasterized=True, zorder=1)
            drawn = ax.boxplot(values, positions=(0, 1), widths=.28, patch_artist=True,
                               showfliers=False, whis=(5, 95),
                               boxprops={"color": INK, "linewidth": .8},
                               whiskerprops={"color": INK, "linewidth": .8},
                               capprops={"color": INK, "linewidth": .8},
                               medianprops={"color": INK, "linewidth": 1.3})
            for box, color in zip(drawn["boxes"], (pre_color, post_color)):
                box.set_facecolor(color)
                box.set_alpha(.55)
            p = float(tests.loc[(tests.event == event) & (tests.feature == feature),
                                "p_holm_30_tests"].iloc[0])
            ax.set_title(f"{TITLES[feature]}\n{p_label(p)}", fontsize=9, pad=4)
            ax.set_xticks((0, 1), (before.replace("_", "-"), after.replace("_", "-")))
            ax.set_xlim(-.42, 1.42)
            ax.set_ylim(*limits)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if column == 0:
                ax.set_ylabel("Feature value")
    save(fig, f"box_{group}_SC_GO", manifest)


def density_data(pairs, feature):
    values = {}
    for event in EVENTS:
        matched = pairs[event].loc[pairs[event].paired_usable]
        values[event] = (matched[f"{feature}_pre"].to_numpy(dtype=float),
                         matched[f"{feature}_post"].to_numpy(dtype=float))
    pooled = np.concatenate((*values["SC"], *values["GO"]))
    bandwidth = common_bandwidth(pooled[:len(pooled)//2], pooled[len(pooled)//2:])
    low, high = float(pooled.min() - 3 * bandwidth), float(pooled.max() + 3 * bandwidth)
    if feature in (*RADIAL, "mse", "mean_error", "frac_inside"):
        low = max(0, low)
    if feature == "frac_inside":
        high = min(1, high)
    grid = np.linspace(low, high, 512)
    curves = {event: (kde_pdf(pre, grid, bandwidth), kde_pdf(post, grid, bandwidth))
              for event, (pre, post) in values.items()}
    return grid, curves, bandwidth


def density_figure(group, features, pairs, manifest, rows):
    fig, axes = figure_axes()
    for row, feature in enumerate(features):
        grid, curves, bandwidth = density_data(pairs, feature)
        ymax = 1.12 * max(float(curve.max()) for event in EVENTS for curve in curves[event])
        for column, event in enumerate(EVENTS):
            ax = axes[row, column]
            colors = PAIRS[event][2:]
            for curve, color in zip(curves[event], colors):
                ax.fill_between(grid, curve, color=color, alpha=.17, linewidth=0)
                ax.plot(grid, curve, color=color, linewidth=1.7)
            ax.set_title(TITLES[feature], fontsize=9, pad=4)
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0, ymax)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlabel("Value")
            if column == 0:
                ax.set_ylabel("Probability density")
            pre, post = curves[event]
            rows.extend({"event": event, "feature": feature, "x": float(x),
                         "pre_pdf": float(a), "post_pdf": float(b),
                         "bandwidth": bandwidth} for x, a, b in zip(grid, pre, post))
    handles = [Line2D([0], [0], color=color, linewidth=1.7, label=label)
               for event in EVENTS for color, label in zip(PAIRS[event][2:],
                   (f"Pre-{event}", f"Post-{event}"))]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, 1.04), fontsize=8)
    save(fig, f"density_{group}_SC_GO", manifest)


def main():
    style()
    table, _clouds, source_paths = load()
    source_paths.extend((Path(__file__), UNIT / "plot_stage_pair_geometry.py",
                         UNIT / "plot_stage_pair_feature_distributions_v2.py"))
    before_hashes = {str(path): sha256(path) for path in source_paths}
    pairs = matched_features(table)
    tests = test_all_features(pairs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, curve_rows = [], []
    for group, features in GROUPS.items():
        box_figure(group, features, pairs, tests, manifest)
        density_figure(group, features, pairs, manifest, curve_rows)
    if any(sha256(path) != before_hashes[str(path)] for path in source_paths):
        raise ValueError("A protected source changed during plotting")
    density = pd.DataFrame(curve_rows)
    for _, group in density.groupby(["event", "feature"]):
        x = group.x.to_numpy()
        for column in ("pre_pdf", "post_pdf"):
            y = group[column].to_numpy()
            area = float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2))
            if not np.isclose(area, 1, atol=1e-3):
                raise ValueError("A plotted density does not integrate to one")
    density.to_csv(OUTPUT / "density_curves.csv", index=False)
    tests.to_csv(OUTPUT / "paired_wilcoxon_holm.csv", index=False)
    pd.concat(pairs.values(), ignore_index=True).to_csv(OUTPUT / "paired_feature_values.csv", index=False)
    pd.DataFrame([row for event in EVENTS for row in summary_rows(pairs[event], event)]).to_csv(
        OUTPUT / "feature_summary.csv", index=False)
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "source_hashes.json").write_text(json.dumps(before_hashes, indent=2) + "\n")
    n_sc, n_go = (int(pairs[event].paired_usable.sum()) for event in EVENTS)
    readme = f"""# Side-by-side geometric feature comparisons

Monkey T, session y070316009-12, channel 7. The corrected six-stage analysis
used 198 short-delay trials, 300-ms windows, m=3 and tau=3 ms. Post-SC starts
after the spatial instruction ends (event 206). Only saved feature values and
clouds were read; no fitter or decoder was run.

Each figure has SC in the left column and GO in the right column. Within each
column, pre and post are paired by original trial number. The five figure
groups contain all 15 features: `radii_width`, `plane_normal`, `major_axis`,
`minor_axis`, and `fit_quality`. For each group, `box_*_SC_GO` shows matched
trials as points with median/IQR boxes and 5th-95th percentile whiskers;
`density_*_SC_GO` shows probability densities, not cumulative curves.
Each figure is exported as a 600-dpi PNG and a PDF.

There are {n_sc} trials with both SC fits usable and {n_go} with both GO fits
usable. The same y-axis limits are used across the two columns for each boxplot
feature. Density plots use the same x/y ranges and a common Gaussian-kernel
bandwidth across all four stages for each feature; plotted coordinates are in
`density_curves.csv`.

The boxplot annotations are two-sided paired Wilcoxon signed-rank p-values,
Holm-corrected across all 30 prespecified tests (15 features x 2 comparisons).
Exact raw/adjusted p-values, matched counts, and median within-trial changes are
in `paired_wilcoxon_holm.csv`. Nonsignificant comparisons are reported too.
These are exploratory, within-session tests; they do not establish an effect
across animals or separate cue effects from the passage of trial time.
Each window was normalized independently, so radii are not raw LFP amplitudes.
The prior v1/v2 plots remain unchanged.
"""
    (OUTPUT / "README.md").write_text(readme)
    print(f"Saved {len(manifest)} combined box/density figure pairs to {OUTPUT}")
    print(f"Matched usable trials: SC={n_sc}, GO={n_go}")
    print(tests.loc[tests.significant_holm_0_05,
                    ["event", "feature", "median_post_minus_pre", "p_holm_30_tests"]].to_string(index=False))


if __name__ == "__main__":
    main()
