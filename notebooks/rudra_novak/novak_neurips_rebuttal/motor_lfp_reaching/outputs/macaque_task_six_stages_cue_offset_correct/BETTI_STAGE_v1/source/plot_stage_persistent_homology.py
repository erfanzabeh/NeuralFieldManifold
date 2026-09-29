"""Publication PNGs from saved stage persistence diagrams and summaries only."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from run_stage_persistent_homology import (
    OUTPUT, STAGES, LABELS, COLORS, digest, write_json, verify_manifest,
)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def curve_panel(summary, dimension, convention, cohort="available"):
    table = summary[(summary.homology_dimension == dimension)
                    & (summary.distance_convention == convention) & (summary.cohort == cohort)]
    fig, ax = plt.subplots(figsize=(3.8, 3.15))
    fig.subplots_adjust(left=.17, right=.98, bottom=.19, top=.78)
    for stage, color in enumerate(COLORS):
        group = table[table.stage == stage].sort_values("grid_index")
        if group.empty or not group.n_windows.iloc[0]:
            continue
        ax.fill_between(group.distance, group.q25, group.q75, color=color, alpha=.13, lw=0, step="post")
        ax.plot(group.distance, group["mean"], color=color, lw=1.2, drawstyle="steps-post")
    xlabel = "Distance threshold" if convention == "raw" else "Distance / RMS radius"
    ax.set(xlabel=xlabel, ylabel=rf"Mean $\beta_{dimension}$", xlim=(0, table.distance.max()))
    upper = table[["mean", "q75"]].max().max()
    ax.set_ylim(0, max(float(upper)*1.08, .05) if np.isfinite(upper) else 1)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.legend(handles=[Line2D([], [], color=c, lw=1.5, label=label) for c, label in zip(COLORS, LABELS)],
               loc="upper center", bbox_to_anchor=(.55, .99), ncol=3, frameon=False,
               columnspacing=1., handlelength=1.5, handletextpad=.5)
    return fig, table.copy()


def distribution_panel(measures, column, matched=False):
    table = measures.copy()
    if matched:
        complete = table.groupby("trial_index").status.apply(lambda s: s.eq("success").all())
        table = table[table.trial_index.isin(complete.index[complete])]
    table = table[np.isfinite(table[column])].sort_values("window_index").copy()
    jitter = np.random.default_rng(230926).uniform(-.18, .18, len(table))
    table["plot_x"] = table.stage + jitter
    fig, ax = plt.subplots(figsize=(3.9, 3.15))
    fig.subplots_adjust(left=.19, right=.98, bottom=.22, top=.97)
    for stage, color in enumerate(COLORS):
        group = table[table.stage == stage]
        if group.empty:
            continue
        ax.scatter(group.plot_x, group[column], s=6, alpha=.45, color=color, edgecolors="none", zorder=2)
        q1, median, q3 = group[column].quantile([.25, .5, .75])
        ax.plot([stage, stage], [q1, q3], color="#292929", lw=2.2, solid_capstyle="butt", zorder=3)
        ax.scatter([stage], [median], s=20, facecolors="white", edgecolors="#292929", linewidths=.8, zorder=4)
    label = "Longest H1 lifetime" if column == "longest_H1_lifetime" else "Longest H1 lifetime / RMS radius"
    ax.set(xticks=range(6), xticklabels=LABELS, xlim=(-.5, 5.5), xlabel="Task stage", ylabel=label)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    high = float(table[column].max()) if len(table) else 1.
    ax.set_ylim(0, max(high*1.07, .01))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    return fig, table


def persistence_panel(intervals, stage, extent):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    fig.subplots_adjust(left=.19, right=.96, bottom=.19, top=.76)
    ax.plot([0, extent], [0, extent], color="#aaaaaa", ls="--", lw=.7)
    infinite_y = extent*1.12
    ax.axhline(infinite_y, color="#cccccc", lw=.6, ls=":")
    colors, markers = ("#a0a0a0", COLORS[stage], "#303030"), ("o", "o", "D")
    for h in range(3):
        group = intervals[intervals.homology_dimension == h]
        finite = group[np.isfinite(group.death)]
        ax.scatter(finite.birth, finite.death, s=8 if h == 0 else 13,
                   color=colors[h], marker=markers[h], alpha=.7, linewidths=0)
        infinite = group[np.isinf(group.death)]
        ax.scatter(infinite.birth, np.full(len(infinite), infinite_y), color=colors[h], marker="^", s=18, linewidths=0)
    ticks = np.linspace(0, extent, 4)
    ax.set(xlim=(-.025*extent, 1.025*extent), ylim=(-.025*extent, 1.20*extent),
           xlabel="Birth distance", ylabel="Death distance", xticks=ticks)
    ax.set_xticklabels([f"{v:.2g}" for v in ticks])
    ax.set_yticks([*ticks, infinite_y], [*(f"{v:.2g}" for v in ticks), r"$\infty$"])
    fig.legend(handles=[Line2D([], [], color=colors[h], marker=markers[h], lw=0,
                        markersize=4, label=f"H{h}") for h in range(3)],
               title=LABELS[stage], title_fontsize=9, ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(.56, .99), handletextpad=.3, columnspacing=1.)
    return fig


def barcode_panel(intervals, stage, extent, max_bars):
    table = intervals[intervals.homology_dimension == 1].copy()
    table = table.sort_values(["lifetime", "birth", "death"], ascending=[False, True, True])
    table["barcode_rank"] = np.arange(1, len(table)+1)
    fig, ax = plt.subplots(figsize=(3.4, max(3.15, .035*max_bars + .8)))
    fig.subplots_adjust(left=.19, right=.97, bottom=.19, top=.89)
    ax.hlines(table.barcode_rank, table.birth, table.death, color=COLORS[stage], lw=1.)
    ax.set(xlim=(0, extent*1.04), ylim=(max(max_bars, 1)+.5, .5),
           xlabel="Distance threshold", ylabel="H1 interval rank", title=LABELS[stage])
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    return fig, table


def save_panel(name, fig, data, caption, manifest):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    off_view_labels = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    off_view_labels.update((tick.label1, tick.label2))
    for text in fig.findobj(matplotlib.text.Text):
        if text in off_view_labels or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Text outside canvas for {name}: {outside}")
    target = OUTPUT / f"{name}.png"
    fig.savefig(target, dpi=600, metadata={"Software": "NeuralFieldManifold stage PH plot-only renderer"})
    plt.close(fig)
    data.to_csv(OUTPUT / "tables" / f"plot_{name}.csv", index=False)
    (OUTPUT / "captions" / f"{name}.txt").write_text(caption + "\n")
    with Image.open(target) as im:
        dpi = im.info.get("dpi")
        assert dpi and all(abs(v-600) < .1 for v in dpi)
        assert im.width >= 1800 and im.height >= 1800
        array = np.asarray(im.convert("RGB"))
        assert np.any(array < 240), "Blank panel"
        assert np.all(array[:3] > 245) and np.all(array[-3:] > 245)
        assert np.all(array[:, :3] > 245) and np.all(array[:, -3:] > 245)
        size = [im.width, im.height]
    manifest.append(dict(file=target.name, sha256=digest(target), pixels=size, dpi=list(dpi), outside_text=outside))


def write_report(measures, manifest):
    successes = measures.status.eq("success")
    counts = measures.groupby("stage_name").status.apply(lambda s: int(s.eq("success").sum()))
    summaries = pd.read_csv(OUTPUT / "tables/lifetime_summary.csv")
    norm = summaries[(summaries.cohort == "available") & (summaries.measure == "normalized_H1_lifetime")].set_index("stage_name")
    raw = summaries[(summaries.cohort == "available") & (summaries.measure == "longest_H1_lifetime")].set_index("stage_name")
    by_stage = "; ".join(f"{label} {norm.loc[stage, 'median']:.3f}" for stage, label in zip(STAGES, LABELS))
    positive = int((measures.loc[successes, "longest_H1_lifetime"] > 0).sum())
    h2 = int((measures.loc[successes, "H2_intervals"] > 0).sum())
    text = f"""# Six-Stage Persistent Homology

Completed **{int(successes.sum())}/1,188** clouds from **198 short-delay trials** in Monkey T,
session y070316009-12, channel 7. Successful windows by stage:
{'; '.join(f'{label}: {counts[stage]}' for stage, label in zip(STAGES, LABELS))}.
All calculations used the saved 294 observed points, m=3, tau=3 ms, and corrected
300-ms windows. Post-TC starts at tone offset; post-SC starts at spatial-instruction
end (distractor onset). No signals, geometric fits, or decoding results changed.

Ripser 0.6.15 computed full H0/H1/H2 persistence with Euclidean distances and
coefficients in F2, without subsampling or an assumed topology.
**{positive}** successful clouds contain H1 intervals; **{h2}** contain H2 intervals.
Stage median longest H1 lifetimes range from **{raw['median'].min():.3f} to {raw['median'].max():.3f}**
in preprocessed-coordinate distance units. RMS-normalized medians are: {by_stage}.
Stage distributions overlap substantially; size-normalized Betti curves are visually
similar. Betti counts are not the oscillatory-mode count K.

Curves show stage means and interquartile bands, not confidence intervals.
Distribution dots are individual trial windows; markers show median/IQR.
Normalized summaries divide diagram distances by each cloud's centered RMS radius;
they do not require transformed clouds or another Ripser run. Trial 6 supplies all
six example diagrams and barcodes, without selection for separation.

Failures: **{int((~successes).sum())}**. Undefined RMS normalizations:
**{int((~measures.normalization_valid).sum())}**. Clouds with unusable geometry fits
remain included ({int((~measures.geometry_usable & successes).sum())} successful PH calculations).
The six stages are repeated measurements from the same trials, not independent recordings.

These are descriptive, scale-dependent measurements from one selected recording.
Short filtered windows, finite sampling, and noise can affect persistence. H1/H2
intervals alone do not establish clean torus topology or stage-specific separation.
No threshold optimization, significance testing, or topology decoding was performed.
"""
    (OUTPUT / "REPORT.md").write_text(text)
    links = "\n".join(f"- [{item['file']}]({item['file']})" for item in manifest)
    (OUTPUT / "README.md").write_text(f"""# Betti Analysis: Corrected Six Stages

This folder contains topology-only results, not a new decoder.
All source geometry fits and decoding outputs remain unchanged.

## Main Panels
- [Loop-count curves](01_betti_h1.png)
- [Size-normalized loop-count curves](02_betti_h1_rms_normalized.png)
- [Longest loop lifetime](03_longest_h1_lifetime.png)
- [Size-normalized longest lifetime](04_longest_h1_lifetime_rms_normalized.png)

## All Standalone PNGs
{links}

## Documentation
- [Short report](REPORT.md)
- [Calculation validation](validation/numerical_checks.json)
- [Plot validation](validation/plot_checks.json)
- [Per-window measurements](tables/window_measurements.csv)
- [Stage lifetime summaries](tables/lifetime_summary.csv)
- [Stage Betti-curve summaries](tables/betti_summary.csv)

`diagrams/` stores full H0/H1/H2 intervals, including infinity. `tables/` contains
measurements, compressed CSVs of intervals and per-window curves, shared grids,
and the exact values used in each PNG. `captions/` provides methods outside the artwork.
`inputs/` preserves trial/stage metadata; the original frozen clouds remain in the
source experiment identified in `config.json`. All raw-distance axes use the existing
preprocessed-coordinate units, not raw voltage. RMS-normalized axes are dimensionless.

Examples always use original trial 6; H1 barcode rows are ranked by lifetime within
each stage, not matched loop identities across stages. All its H1 intervals are shown.
H0's essential infinite interval is displayed separately above the finite diagram scale.

Regeneration uses the saved tables and intervals only:
`python plot_stage_persistent_homology.py` from the motor-LFP analysis directory.
The independent runner supports `--phase pilot`, `--phase all --resume`, and
`--phase verify --resume`. Neither entrypoint calls a geometric fitter or decoder.
""")


def main():
    style()
    assert json.loads((OUTPUT / "validation/numerical_checks.json").read_text())["passed"]
    verify_manifest(OUTPUT / "provenance/input_source_hashes.json")
    verify_manifest(OUTPUT / "provenance/table_hashes.json")
    analysis_paths = [p for directory in ("diagrams", "checkpoints", "inputs") for p in (OUTPUT / directory).rglob("*") if p.is_file()]
    analysis_hashes = {str(p): digest(p) for p in analysis_paths}
    measures = pd.read_csv(OUTPUT / "tables/window_measurements.csv")
    summary = pd.read_csv(OUTPUT / "tables/betti_summary.csv")
    intervals = pd.read_csv(OUTPUT / "tables/persistence_intervals.csv.gz")
    manifests = []
    for name, h, mode in (("01_betti_h1", 1, "raw"), ("02_betti_h1_rms_normalized", 1, "rms_normalized"),
                          ("05_betti_h0", 0, "raw"), ("06_betti_h2", 2, "raw")):
        fig, data = curve_panel(summary, h, mode)
        save_panel(name, fig, data,
            f"Stage-wise mean beta{h} with interquartile bands across successful trial clouds. "
            f"Distance convention: {mode}. Count birth <= distance < death on a shared 512-point grid. "
            "Bands show trial variability, not confidence intervals; see data for stage counts.", manifests)
    for name, col in (("03_longest_h1_lifetime", "longest_H1_lifetime"),
                      ("04_longest_h1_lifetime_rms_normalized", "normalized_H1_lifetime")):
        fig, data = distribution_panel(measures, col)
        save_panel(name, fig, data, "One dot per successful trial window; median and interquartile range. "
            "Zero denotes an empty successful H1 diagram, never a failed calculation. "
            "RMS normalization is applied only to derived distances, not to Ripser inputs.", manifests)
    example = measures[measures.original_trial_number == 6].sort_values("stage")
    assert len(example) == 6
    selected = intervals[intervals.original_trial_number == 6]
    deaths = selected.loc[np.isfinite(selected.death), "death"]
    extent = float(deaths.max()) if len(deaths) else 1.
    h1 = selected[selected.homology_dimension == 1]
    h1_extent = float(h1.death.max()) if len(h1) else extent
    max_bars = int(h1.groupby("stage_name").size().max()) if len(h1) else 0
    for row in example.itertuples():
        if row.status != "success":
            continue  # Failure is reported, not replaced by a more attractive trial.
        data = selected[selected.window_index == row.window_index].copy()
        fig = persistence_panel(data, row.stage, extent)
        save_panel(f"07_persistence_trial6_{row.stage_name}", fig, data,
            f"Original trial 6, {LABELS[row.stage]}. Full H0/H1/H2 diagram; infinity is plotted "
            "above the shared finite scale. No topology imposed and no intervals removed.", manifests)
        fig, bars = barcode_panel(data, row.stage, h1_extent, max_bars)
        save_panel(f"08_h1_barcode_trial6_{row.stage_name}", fig, bars,
            f"Original trial 6, {LABELS[row.stage]}. All H1 intervals, ranked by decreasing lifetime. "
            "Shared distance and rank scales across the six stages; rows are not cross-stage matched loops.", manifests)
    if measures.status.ne("success").any():
        for mode in ("raw", "rms_normalized"):
            fig, data = curve_panel(summary, 1, mode, "matched_six_stages")
            save_panel(f"09_matched_betti_h1_{mode}", fig, data,
                "Sensitivity: only trials with six successful PH calculations; mean and interquartile bands.", manifests)
        for col in ("longest_H1_lifetime", "normalized_H1_lifetime"):
            fig, data = distribution_panel(measures, col, matched=True)
            save_panel(f"10_matched_{col}", fig, data,
                "Sensitivity: only trials with six successful PH calculations; trial dots and median/IQR.", manifests)
    write_report(measures, sorted(manifests, key=lambda m: m["file"]))
    for path, expected in analysis_hashes.items():
        assert digest(path) == expected
    verify_manifest(OUTPUT / "provenance/input_source_hashes.json")
    verify_manifest(OUTPUT / "provenance/table_hashes.json")
    protected_count = verify_manifest(OUTPUT / "provenance/protected_hashes.json")
    assert "ripser" not in sys.modules, "Plot-only process imported Ripser"
    assert not any(name.startswith("NeuralFieldManifold.fits") for name in sys.modules)
    shutil.copy2(__file__, OUTPUT / "source" / Path(__file__).name)
    write_json(OUTPUT / "validation/plot_checks.json", dict(passed=True, png_count=len(manifests),
        protected_files_unchanged=protected_count, analysis_files_unchanged=len(analysis_hashes),
        imported_ripser=False, fitter_calls=0, decoder_calls=0,
        source_sha256=digest(__file__), panels=manifests))
    print(json.dumps(dict(pngs=len(manifests), folder=str(OUTPUT), validated=True), indent=2))


if __name__ == "__main__":
    main()
