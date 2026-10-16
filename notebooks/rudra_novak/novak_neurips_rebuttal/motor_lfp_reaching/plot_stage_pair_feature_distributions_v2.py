"""Render boxplots and density estimates from saved corrected stage features."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from plot_stage_pair_geometry import (
    COLLECTION, ORIENTATION, PAIRS, QUALITY, RADIAL, SOURCE, TITLES, UNIT,
    load, pair_table, sha256, style, summary_rows,
)


OUTPUT = COLLECTION / "stage_pair_feature_distributions_v2"
GROUPS = {"radii_width": RADIAL, "orientation": ORIENTATION, "fit_quality": QUALITY}
INK = "#263238"


def save_figure(fig, stem, manifest):
    png = OUTPUT / f"{stem}.png"
    pdf = OUTPUT / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    with Image.open(png) as image:
        pixels = image.size
        rgb = np.asarray(image.convert("RGB"))
        if min(pixels) < 1000 or np.std(rgb) < 5:
            raise ValueError(f"Blank or unexpectedly small figure: {png}")
    manifest.append({"stem": stem, "png_pixels": list(pixels),
                     "png_sha256": sha256(png), "pdf_sha256": sha256(pdf)})


def axes_for(features):
    if len(features) == 9:
        return plt.subplots(3, 3, figsize=(9.4, 7.4), sharey=False, constrained_layout=True)
    return plt.subplots(1, 3, figsize=(9.4, 3.05), sharey=False, constrained_layout=True)


def boxes(event, group, features, pair, before, after, colors, manifest):
    matched = pair.loc[pair.paired_usable]
    fig, axes = axes_for(features)
    axes = np.ravel(axes)
    for index, (ax, feature) in enumerate(zip(axes, features)):
        values = [matched[f"{feature}_pre"].to_numpy(dtype=float),
                  matched[f"{feature}_post"].to_numpy(dtype=float)]
        for position, (data, color) in enumerate(zip(values, colors)):
            rng = np.random.default_rng(98240 + index * 2 + position)
            ax.scatter(position + rng.uniform(-.12, .12, len(data)), data, s=5,
                       color=color, alpha=.24, linewidths=0, rasterized=True, zorder=1)
        drawn = ax.boxplot(values, positions=(0, 1), widths=.28, patch_artist=True,
                           showfliers=False, whis=(5, 95),
                           boxprops={"color": INK, "linewidth": .8},
                           whiskerprops={"color": INK, "linewidth": .8},
                           capprops={"color": INK, "linewidth": .8},
                           medianprops={"color": INK, "linewidth": 1.3})
        for box, color in zip(drawn["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(.55)
        ax.set_xticks((0, 1), (before.replace("_", "-"), after.replace("_", "-")))
        ax.set_xlim(-.42, 1.42)
        ax.set_title(TITLES[feature])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if index % 3 == 0:
            ax.set_ylabel("Feature value")
        if feature == "frac_inside":
            ax.set_ylim(0, 1.02)
    save_figure(fig, f"{event.lower()}_box_{group}", manifest)


def common_bandwidth(pre, post):
    pooled = np.concatenate((pre, post))
    spread = float(np.std(pooled, ddof=1))
    if spread <= 0 or not np.isfinite(spread):
        raise ValueError("A constant feature cannot support a continuous density estimate")
    return 1.06 * spread * len(pooled) ** (-1 / 5)


def kde_pdf(data, grid, bandwidth):
    z = (grid[:, None] - data[None, :]) / bandwidth
    density = np.exp(-.5 * z * z).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    area = float(np.sum((density[:-1] + density[1:]) * np.diff(grid) / 2))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid density normalization")
    density /= area
    return density


def densities(event, group, features, pair, colors, manifest, density_rows):
    matched = pair.loc[pair.paired_usable]
    fig, axes = axes_for(features)
    axes = np.ravel(axes)
    for index, (ax, feature) in enumerate(zip(axes, features)):
        pre = matched[f"{feature}_pre"].to_numpy(dtype=float)
        post = matched[f"{feature}_post"].to_numpy(dtype=float)
        pooled = np.concatenate((pre, post))
        bandwidth = common_bandwidth(pre, post)
        lo, hi = float(pooled.min() - 3 * bandwidth), float(pooled.max() + 3 * bandwidth)
        if feature in (*RADIAL, "mse", "mean_error", "frac_inside"):
            lo = max(0., lo)
        if feature == "frac_inside":
            hi = min(1., hi)
        grid = np.linspace(lo, hi, 512)
        pre_pdf = kde_pdf(pre, grid, bandwidth)
        post_pdf = kde_pdf(post, grid, bandwidth)
        for density, color in ((pre_pdf, colors[0]), (post_pdf, colors[1])):
            ax.fill_between(grid, density, color=color, alpha=.18, linewidth=0)
            ax.plot(grid, density, color=color, linewidth=1.7)
        ax.set_title(TITLES[feature])
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if index % 3 == 0:
            ax.set_ylabel("Probability density")
        ax.set_xlabel("Value")
        density_rows.extend({"event": event, "feature": feature, "x": float(x),
                             "pre_pdf": float(a), "post_pdf": float(b),
                             "bandwidth": bandwidth, "n_matched_trials": len(matched)}
                            for x, a, b in zip(grid, pre_pdf, post_pdf))
    rows = 3 if len(features) == 9 else 1
    handles = [Line2D([0], [0], color=color, linewidth=1.7, label=label)
               for color, label in zip(colors, ("Pre", "Post"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.18 if rows == 1 else 1.04))
    save_figure(fig, f"{event.lower()}_density_{group}", manifest)


def main():
    style()
    table, _clouds, source_paths = load()
    source_paths.extend((Path(__file__), UNIT / "plot_stage_pair_geometry.py"))
    before_hashes = {str(path): sha256(path) for path in source_paths}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, summary_rows_all, pair_frames, density_rows = [], [], [], []
    for event, (before, after, pre_color, post_color) in PAIRS.items():
        pair = pair_table(table, before, after)
        pair.insert(0, "comparison", f"{before}_vs_{after}")
        pair_frames.append(pair)
        summary_rows_all.extend(summary_rows(pair, event))
        for group, features in GROUPS.items():
            boxes(event, group, features, pair, before, after,
                  (pre_color, post_color), manifest)
            densities(event, group, features, pair, (pre_color, post_color),
                      manifest, density_rows)
    if any(sha256(path) != before_hashes[str(path)] for path in source_paths):
        raise ValueError("A protected source changed during plotting")
    pd.concat(pair_frames, ignore_index=True).to_csv(OUTPUT / "paired_feature_values.csv", index=False)
    pd.DataFrame(summary_rows_all).to_csv(OUTPUT / "feature_summary.csv", index=False)
    density_table = pd.DataFrame(density_rows)
    density_table.to_csv(OUTPUT / "density_curves.csv", index=False)
    for (_, _), group in density_table.groupby(["event", "feature"]):
        x = group.x.to_numpy()
        for column in ("pre_pdf", "post_pdf"):
            y = group[column].to_numpy()
            area = np.sum((y[:-1] + y[1:]) * np.diff(x) / 2)
            if not np.isclose(area, 1, atol=1e-3):
                raise ValueError("Saved density does not integrate to one")
    (OUTPUT / "source_hashes.json").write_text(json.dumps(before_hashes, indent=2) + "\n")
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sc_count = int(pair_frames[0].paired_usable.sum())
    go_count = int(pair_frames[1].paired_usable.sum())
    readme = f"""# Corrected pre/post stage feature comparisons

Monkey T, session y070316009-12, channel 7. These are plot-only summaries of
the saved six-stage tau=3 ms, m=3, K=1 geometric fits. Post-SC starts after
the spatial instruction ends (event 206), not at spatial-cue onset.

Each comparison includes only trials with usable fits in both stages:
{sc_count} of 198 for pre/post-SC and {go_count} of 198 for pre/post-GO.
All 15 fitted features are included. No fitting, decoding, significance testing,
or feature redefinition was performed.

## Figure files

For each event prefix, `sc_` or `go_`:

| File suffix | Content |
| --- | --- |
| `box_radii_width` | R1, R2, and band half-width boxplots with all matched-trial points. |
| `box_orientation` | Nine plane-normal, major-axis, and minor-axis components. |
| `box_fit_quality` | MSE, mean in-plane error, and fraction inside. |
| `density_radii_width` | Probability-density curves for R1, R2, and width. |
| `density_orientation` | Probability-density curves for the nine orientation components. |
| `density_fit_quality` | Probability-density curves for the three fit-quality measures. |

Every figure is provided as a 600-dpi PNG and a PDF. The word `density`
means a probability-density estimate, not a cumulative distribution. For each
feature, pre/post curves use the same Gaussian-kernel bandwidth calculated from
their pooled matched-trial values. Curves are normalized over the displayed
range; `density_curves.csv` records all plotted coordinates and bandwidths.
Box centers are medians, boxes are interquartile ranges, and whiskers mark the
5th and 95th percentiles. `paired_feature_values.csv` retains every trial and
its within-trial difference; `feature_summary.csv` gives matched medians.

Each 300-ms window was independently normalized before fitting. Radii and width
are therefore in normalized signal units, not raw LFP amplitude. These are
descriptive comparisons from one recording. The earlier cloud examples remain
in `../stage_pair_geometry_v1/` and were not changed here.
"""
    (OUTPUT / "README.md").write_text(readme)
    print(f"Saved {len(manifest)} box/density figure pairs to {OUTPUT}")
    print(f"Matched usable trials: SC={sc_count}, GO={go_count}")


if __name__ == "__main__":
    main()
