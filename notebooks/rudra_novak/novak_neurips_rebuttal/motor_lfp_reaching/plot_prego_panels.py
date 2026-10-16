#!/usr/bin/env python
"""Render standalone manuscript panels exclusively from checksum-verified tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "prego_single_channel"
PANELS = ["A", "B", "C", "D", "E", "F", "S1", "S2"]
METHODS = ["peak_power", "peak_frequency", "power_frequency", "all_bands", "torus"]
LABELS = dict(peak_power="Peak\npower", peak_frequency="Peak\nfrequency",
              power_frequency="Power +\nfrequency", all_bands="All band\npowers",
              torus="Torus\nfeatures", torus_pca="Torus\nPCA")
COLORS = dict(peak_power="#D98565", peak_frequency="#E9B8A8", power_frequency="#668397",
              all_bands="#858585", torus="#9B302B", torus_pca="#B97639",
              torus_power_frequency="#8D5551", torus_all_bands="#436E68")
INK = "#203442"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "svg.fonttype": "none",
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "figure.facecolor": "white", "axes.facecolor": "white"})


def load_tables(output):
    manifest = json.loads((output / "frozen_manifest.json").read_text())
    if manifest["status"] != "frozen":
        raise ValueError("Results are not frozen")
    tables = {}
    for name, expected in manifest["table_sha256"].items():
        path = output / "tables" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Table checksum mismatch: {name}")
        tables[path.stem] = pd.read_csv(path)
    return tables


def selected_panels(value):
    if value == "all":
        return PANELS
    if value not in PANELS:
        raise ValueError(value)
    return [value]


def save(fig, output, name, data, caption):
    folder = output / "panels"
    folder.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=600, facecolor="white")
    data.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.caption.txt").write_text(caption + "\n")
    plt.close(fig)
    print(name, flush=True)


def axes_style(ax, ylabel, ylim=None):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=.5)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)


def primary(tables, table, monkey=None):
    frame = tables[table]
    frame = frame[frame.analysis == "short"] if "analysis" in frame else frame
    return frame[frame.monkey == monkey] if monkey else frame


def cohort_text(tables, monkey):
    row = primary(tables, "summary", monkey).iloc[0]
    return f"Monkey {monkey}: {int(row.n_lfps)} LFP recordings from {int(row.n_days)} recording days, {int(row.n_trials)} balanced trial observations; one animal."


def method_text(monkey):
    dims = "2, 2, 4, 5, and 15" if monkey == "M" else "1, 1, 2, 5, and 15"
    bands = "12-25 and 25-40 Hz" if monkey == "M" else "12-40 Hz"
    return f"Feature dimensions (peak power, peak frequency, joint power/frequency, all bands, torus): {dims}. Spectral peak bands: {bands}."


def p_text(value):
    if value >= .01:
        return f"p = {value:.3f}"
    return f"p = {value:.4f}" if value >= .0001 else f"p = {value:.1e}"


def bracket(ax, x1, x2, y, label, height):
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], c=INK, lw=.7)
    ax.text((x1+x2)/2, y+height*1.5, label, ha="center", va="bottom", fontsize=8)


def panel_a(t, output):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.02, right=.99, top=.97, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    target = fig.add_axes([.03, .39, .20, .47])
    target.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal"); target.axis("off")
    target.add_patch(Circle((0, 0), .12, color=INK))
    for direction in range(1, 7):
        theta = np.deg2rad(60 - 60*(direction-1))
        xy = (np.cos(theta), np.sin(theta))
        target.add_patch(Circle(xy, .16, facecolor="#CB9852" if direction == 1 else "white", edgecolor=INK, lw=1))
        target.text(1.33*xy[0], 1.33*xy[1], str(direction), ha="center", va="center", fontsize=9)
    target.annotate("", xy=(.42, .73), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#9B302B", lw=1.4))
    ax.text(.12, .93, "Six instructed directions", ha="center", fontweight="bold")
    events = [("Temporal\ncue", .32), ("Spatial\ncue", .59), ("GO", .86), ("Reach", .97)]
    ax.annotate("", xy=(.995, .67), xytext=(.28, .67), arrowprops=dict(arrowstyle="->", color=INK))
    ax.add_patch(Rectangle((.72, .60), .14, .14, facecolor="#E8C5C2", edgecolor="none"))
    for text, x in events:
        ax.plot([x, x], [.63, .71], color=INK, lw=1)
        ax.text(x, .77, text, ha="center", va="bottom", fontsize=9)
    ax.text(.455, .69, "D1", ha="center", va="bottom")
    ax.text(.685, .69, "D2", ha="center", va="bottom")
    ax.text(.79, .56, "Final 500 ms", ha="center", color="#862C29", fontweight="bold")
    ax.text(.59, .43, "Target-direction\ninstruction", ha="center", fontsize=8)
    ax.text(.80, .34, "Short D1 and D2: M 1,000 ms; T 700 ms", ha="center", fontsize=8)
    ax.text(.63, .24, "Schematic timeline; not to scale", ha="center", fontsize=8, color="#60666A")
    pipeline = [("One LFP\nchannel", .04, .18), ("Spectral or geometric\nfeatures", .29, .26),
                ("LDA", .63, .10), ("Predicted\ndirection", .82, .16)]
    for label, x, width in pipeline:
        ax.add_patch(Rectangle((x, .01), width, .16, facecolor="#F2F5F7", edgecolor="#9AA9B2", lw=.7))
        ax.text(x+width/2, .09, label, ha="center", va="center", fontsize=9)
    for start, end in ((.22, .29), (.55, .63), (.73, .82)):
        ax.annotate("", xy=(end-.006, .09), xytext=(start+.006, .09), arrowprops=dict(arrowstyle="->", color=INK, lw=.9))
    data = t["recording_audit"][t["recording_audit"].main_included]
    caption = ("Six-direction instructed delayed-reaching task. Temporal cue specifies short/long delays; spatial cue specifies target direction. "
        "Only short-delay trials enter main panels; each trial supplies a single-channel 500-ms pre-GO segment. "
        "D1 and D2 nominal short durations: M 1,000 ms, T 700 ms. Timeline and target layout are schematic. "
        "GO alignment comes from raw event code 207 (zero-based sample M=5000, T=4500), not the old converted constant. "
        "Recordings can share behavioral trials, so summed trial observations are not independent behavioral trials. "
        + " ".join(cohort_text(t, m) for m in "MT") + " Five-fold stratified LDA is evaluated independently within each recording.")
    save(fig, output, "A_task_window", data, caption)


def radius_plot(ax, frame, radius):
    points = [frame.loc[frame.direction == d, radius].dropna().values for d in range(1, 7)]
    ax.boxplot(points, positions=np.arange(1, 7), widths=.5, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#ECD3D0", edgecolor="#7A3D39", linewidth=.7),
        medianprops=dict(color=INK, linewidth=1), whiskerprops=dict(color="#7A3D39", linewidth=.7),
        capprops=dict(color="#7A3D39", linewidth=.7))
    rng = np.random.default_rng(42)
    for d, values in enumerate(points, 1):
        ax.scatter(d+rng.uniform(-.18, .18, len(values)), values, s=4, alpha=.45, c="#963C35", linewidths=0, zorder=3)
    axes_style(ax, f"{radius} (normalized units)")
    ax.set_xlabel("Reach direction")
    ax.set_xticks(range(1, 7))


def panel_b(t, output):
    for monkey in "MT":
        frame = t["example_geometry"][t["example_geometry"].monkey == monkey]
        record = frame.recording.iloc[0]
        caption = (f"Descriptive geometry example, Monkey {monkey}, {record}. Selected nearest the animal's median label-blind fit MSE, "
            "with recording-ID tie break; decoding scores are not used. All displayed trials use the reference embedding learned "
            "from training trials of fold 0. Points are trials; boxes span quartiles with median and 1.5-IQR whiskers. "
            "Failed fits are omitted only from this descriptive plot. These three radii are a subset of the 15 decoder features, "
            "not evidence of direction separation in every recording. Geometry uses robust trial normalization; radii are not voltage amplitudes.")
        for radius in ("R1", "R2", "r"):
            fig, ax = plt.subplots(figsize=(3.3, 2.45), layout="constrained")
            radius_plot(ax, frame, radius); ax.set_title(f"Monkey {monkey}: {radius}")
            save(fig, output, f"B_geometry_{monkey}_{radius}", frame[["recording", "raw_trial_index", "direction", "fold", radius]], caption)
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6), layout="constrained")
        for ax, radius in zip(axes, ("R1", "R2", "r")):
            radius_plot(ax, frame, radius); ax.set_title(f"Monkey {monkey}: {radius}")
        save(fig, output, f"B_preview_{monkey}", frame, caption)


def panel_c(t, output):
    for monkey in "MT":
        for method in ("power_frequency", "torus"):
            data = primary(t, "mean_confusions", monkey)
            data = data[data.method == method]
            matrix = data.pivot(index="true_direction", columns="predicted_direction", values="fraction").reindex(index=range(1, 7), columns=range(1, 7)).values
            fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="constrained")
            im = ax.pcolormesh(np.arange(7)-.5, np.arange(7)-.5, matrix, cmap="Reds", vmin=0, vmax=1,
                               shading="flat", rasterized=False)
            ax.set_aspect("equal")
            ax.set_ylim(5.5, -.5)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if matrix[i,j] > .55 else "black")
            ax.set(xticks=range(6), yticks=range(6), xticklabels=range(1, 7), yticklabels=range(1, 7),
                   xlabel="Predicted direction", ylabel="True direction", title=f"Monkey {monkey}: {LABELS[method].replace(chr(10), ' ')}")
            colorbar = fig.colorbar(im, ax=ax, shrink=.82, pad=.04)
            colorbar.set_label("Prediction fraction")
            colorbar.solids.set_rasterized(False)
            caption = (cohort_text(t, monkey) + " Short-delay pre-GO held-out predictions; each recording's confusion matrix is row-normalized "
                "before equal-recording averaging. All matrices share direction order 1-6 and color scale 0-1. Diagonal entries are class recall, not F1.")
            save(fig, output, f"C_confusion_{monkey}_{method}", data, caption)


def panel_d(t, output):
    all_summary = primary(t, "summary")
    selected = all_summary[all_summary.method.isin(METHODS)]
    height = float((selected["mean"]+selected.sd).max())
    limit = max(.3, np.ceil((height+.14)*10)/10)
    for monkey in "MT":
        data = primary(t, "summary", monkey).set_index("method").loc[METHODS].reset_index()
        null = t["null_summary"][t["null_summary"].monkey == monkey].set_index("method").loc[METHODS]
        data = data.merge(null[["mean", "low", "high"]].add_prefix("null_").reset_index(), on="method")
        fig, ax = plt.subplots(figsize=(3.6, 3.55))
        fig.subplots_adjust(left=.14, right=.995, bottom=.24, top=.87)
        x = np.arange(5)
        ax.bar(x, data["mean"], yerr=data.sd, color=[COLORS[m] for m in METHODS], width=.67,
               edgecolor="#383838", linewidth=.5, capsize=2, error_kw=dict(elinewidth=.8))
        center = data.null_mean.values
        ax.errorbar(x+.16, center, yerr=np.vstack([center-data.null_low, data.null_high-center]),
                    fmt="o", color="#333333", markersize=3, elinewidth=.8, capsize=2, label="Shuffled labels (95% null interval)")
        comparisons = t["planned_comparisons"]
        comparisons = comparisons[(comparisons.monkey == monkey) & (comparisons.panel == "D")]
        for offset, baseline in enumerate(("power_frequency", "all_bands")):
            p = comparisons.loc[comparisons.baseline == baseline, "p_holm"].iloc[0]
            bracket(ax, METHODS.index(baseline), 4, height+.025+offset*.06, p_text(p), .006)
        axes_style(ax, "Macro-F1", (0, limit))
        ax.set_xlim(-.5, 4.5)
        ax.set_xticks(x, [LABELS[m] for m in METHODS]); ax.set_title(f"Monkey {monkey}")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), frameon=False, handletextpad=.3)
        caption = (cohort_text(t, monkey) + " Short-delay trials, final 500 ms pre-GO. Five-fold stratified, shrinkage LDA; identical trials/folds across methods. "
            "Bars are recording-level mean macro-F1 +/- SD. Gray markers and intervals summarize the across-recording mean under 200 within-fold label permutations (2.5-97.5 percentiles), not a theoretical 1/6 F1 threshold. "
            "Paired tests use recording-day mean differences, with Holm correction over all eight planned D/E comparisons. " + method_text(monkey))
        save(fig, output, f"D_decoding_{monkey}", data, caption)
        comparisons.to_csv(output / "panels" / f"D_decoding_{monkey}.tests.csv", index=False)


def panel_e(t, output):
    all_pairs = t["paired_differences"]
    all_pairs = all_pairs[all_pairs.enhanced.str.startswith("torus_")]
    low, high = min(-.05, all_pairs.difference.min()-.02), max(.05, all_pairs.difference.max()+.025)
    for monkey in "MT":
        data = all_pairs[all_pairs.monkey == monkey]
        tests = t["planned_comparisons"]
        tests = tests[(tests.monkey == monkey) & (tests.panel == "E")]
        fig, ax = plt.subplots(figsize=(3.45, 3.2), layout="constrained")
        ax.axhline(0, color="#626262", ls="--", lw=.8)
        rng = np.random.default_rng(42)
        for x, baseline in enumerate(("power_frequency", "all_bands")):
            points = data[data.baseline == baseline].difference
            row = tests[tests.baseline == baseline].iloc[0]
            ax.scatter(x+rng.uniform(-.17, .17, len(points)), points, s=9, color=COLORS[baseline], alpha=.5, linewidths=0)
            ax.vlines(x+.23, row.ci_low, row.ci_high, color=INK, linewidth=1.5)
            ax.hlines([row.ci_low, row.ci_high], x+.18, x+.28, color=INK, linewidth=1)
            ax.scatter([x+.23], [row.mean_difference], marker="D", s=25, c=INK, zorder=4)
            ax.text(x, high-.008, p_text(row.p_holm), ha="center", va="top", fontsize=8)
        ax.set_xlim(-.48, 1.55)
        ax.set_xticks([0, 1], ["Add torus to\npower + frequency", "Add torus to\nall band powers"])
        axes_style(ax, "Change in macro-F1", (low, high)); ax.set_title(f"Monkey {monkey}")
        caption = (cohort_text(t, monkey) + " Each point is an LFP's paired macro-F1 difference using identical short-delay trials and folds. "
            "Diamonds show the mean recording difference with a 10,000-draw recording-day-cluster bootstrap 95% interval; zero indicates no improvement. "
            "Two-sided Wilcoxon tests use day-averaged paired differences; displayed p values are Holm-corrected over eight planned tests. "
            "Positive values favor adding geometry; nonsignificant comparisons are retained. "
            f"Fused dimensions: {19 if monkey == 'M' else 17}D for torus plus power/frequency, and 20D for torus plus five band powers.")
        save(fig, output, f"E_added_value_{monkey}", data, caption)
        tests.to_csv(output / "panels" / f"E_added_value_{monkey}.statistics.csv", index=False)


def panel_f(t, output):
    all_data = primary(t, "direction_summary")
    selected = all_data[all_data.method.isin(["power_frequency", "torus"])]
    limit = min(1., max(.4, np.ceil((selected["mean"]+selected["std"]).max()*10)/10+.05))
    for monkey in "MT":
        data = selected[selected.monkey == monkey]
        fig, ax = plt.subplots(figsize=(3.7, 2.8), layout="constrained")
        for offset, method in zip((-.18, .18), ("power_frequency", "torus")):
            values = data[data.method == method].sort_values("direction")
            ax.bar(np.arange(1, 7)+offset, values["mean"], yerr=values["std"], width=.34,
                   color=COLORS[method], label=LABELS[method].replace("\n", " "), capsize=1.5, error_kw=dict(elinewidth=.7))
        axes_style(ax, "Per-direction F1", (0, limit))
        ax.set(xticks=range(1, 7), xlabel="Reach direction", title=f"Monkey {monkey}")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=1)
        caption = (cohort_text(t, monkey) + " Per-direction F1 calculated from held-out predictions within each recording, then averaged equally across recordings. "
            "Bars show mean +/- SD. Same short-delay cohort, pre-GO window, and folds as D; common direction order and axis limits across animals. "
            "Descriptive comparison only: no direction-specific hypothesis tests.")
        save(fig, output, f"F_direction_f1_{monkey}", data, caption)


def panel_s1(t, output):
    all_data = t["summary"]
    selected = all_data[all_data.analysis.isin(["matched_short", "matched_long"]) & all_data.method.isin(["power_frequency", "all_bands", "torus"])]
    limit = max(.4, np.ceil((selected["mean"]+selected.sd).max()*10)/10+.05)
    for monkey in "MT":
        data = selected[selected.monkey == monkey]
        fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
        methods = ["power_frequency", "all_bands", "torus"]
        for offset, analysis in zip((-.18, .18), ("matched_short", "matched_long")):
            rows = data[data.analysis == analysis].set_index("method").loc[methods]
            ax.bar(np.arange(3)+offset, rows["mean"], yerr=rows.sd, width=.34,
                color=[COLORS[m] for m in methods], alpha=1 if analysis.endswith("short") else .45,
                hatch=None if analysis.endswith("short") else "///", label=analysis.split("_")[1].capitalize(),
                edgecolor="#3F3F3F", linewidth=.5, capsize=2, error_kw=dict(elinewidth=.8))
        axes_style(ax, "Macro-F1", (0, limit))
        ax.set_xticks(range(3), [LABELS[m] for m in methods]); ax.set_title(f"Monkey {monkey}")
        ax.legend(frameon=False, ncol=2)
        row = data.iloc[0]
        caption = (f"Monkey {monkey}: {int(row.n_lfps)} matched LFPs, {int(row.n_days)} days, one animal. "
            "Separate within-delay decoding, not cross-delay transfer. Each recording contributes equal numbers of short and long trials per direction "
            "(minimum available count across all 12 conditions, at least 15 per condition). Both use the final 500 ms pre-GO and all methods share folds "
            "within each delay. Bars are mean +/- SD across the same recordings. Long D1/D2: M 2,000 ms, T 1,500 ms; short: M 1,000 ms, T 700 ms. "
            "This matched subset can differ from the main cohort; this supporting comparison has no additional inferential test.")
        save(fig, output, f"S1_delay_robustness_{monkey}", data, caption)


def panel_s2(t, output):
    methods = ["power_frequency", "torus", "torus_pca"]
    selected = primary(t, "summary")
    selected = selected[selected.method.isin(methods)]
    limit = max(.4, np.ceil((selected["mean"]+selected.sd).max()*10)/10+.05)
    for monkey in "MT":
        dim = 4 if monkey == "M" else 2
        data = selected[selected.monkey == monkey].set_index("method").loc[methods].reset_index()
        fig, ax = plt.subplots(figsize=(3.35, 2.9), layout="constrained")
        ax.bar(range(3), data["mean"], yerr=data.sd, width=.64, color=[COLORS[m] for m in methods], capsize=3,
               edgecolor="#3F3F3F", linewidth=.5, error_kw=dict(elinewidth=.8))
        axes_style(ax, "Macro-F1", (0, limit)); ax.set_title(f"Monkey {monkey}")
        ax.set_xticks(range(3), [f"Power + frequency\n({dim}D)", "Torus\n(15D)", f"Torus PCA\n({dim}D)"])
        caption = (cohort_text(t, monkey) + f" Same short-delay cohort and held-out folds as main panels. Full 15D torus features are compared with "
            f"PCA-reduced torus features matched to the {dim}D spectral baseline. Median imputation, scaling, and feature PCA are fitted on training trials only; "
            "the frozen transform is applied to held-out trials. Bars are recording means +/- SD. This is a feature-dimensionality control, not a refitted lower-order geometry model.")
        save(fig, output, f"S2_dimensionality_{monkey}", data, caption)


def write_index(tables, output):
    import fitz

    rows = ["# Pre-GO Single-Channel Decoding", "",
        "Main panels: short-delay, six-direction decoding from the final 500 ms before GO.",
        "All panels are separate PDF, editable SVG, and 600-dpi PNG files. Use the PDF/SVG at their native size; labels are at least 8 pt.",
        "Each has an underlying CSV and caption; D/E have additional test tables. Panel letters are not baked into the artwork.", "",
        "## Cohort", ""]
    rows += [cohort_text(tables, monkey) for monkey in "MT"]
    rows += ["", "Trial observations can include the same behavioral trial on different channels. No pseudopopulations, multichannel decoder, or cross-animal transfer.",
        "", "## Timing Audit", "",
        "The raw GO event (207) is at MATLAB sample 5001 for M and 4501 for T. The old converted metadata uses 4500 zero-based for both. "
        "This run uses the raw event table, giving M [4500:5000] and T [4000:4500]. Old results remain unchanged; old M timing needs a separate review.",
        "", "## Individual Panels", "", "| Artwork | PDF | SVG | PNG | Data | Caption |", "|---|---|---|---|---|---|"]
    for path in sorted((output / "panels").glob("*.pdf")):
        name = path.stem
        rows.append("| "+name+" | "+" | ".join(f"[{label}](panels/{name}.{suffix})" for label, suffix in
                    (("PDF", "pdf"), ("SVG", "svg"), ("PNG", "png"), ("CSV", "csv"), ("Caption", "caption.txt")))+" |")
    rows += ["", "## Audit Tables", "",
        "- [Held-out predictions](tables/heldout_predictions.csv)",
        "- [Recording scores](tables/recording_scores.csv)",
        "- [Mean and SD summaries](tables/summary.csv)",
        "- [All eight planned comparisons](tables/planned_comparisons.csv)",
        "- [Recording eligibility and GO audit](tables/recording_audit.csv)",
        "- [Example selection](tables/example_selection.csv)",
        "- [Training-fold embedding selections](tables/embedding_selection.csv)",
        "- [Feature-fit exclusions](tables/exclusions_geometry.csv)",
        "- [Failed trial fits retained via training-median imputation](tables/fit_failures.csv)",
        "- [Numerical/export validation](validation.json)",
        "- [Methods, source references, and reproduction commands](../../PREGO_ANALYSIS.md)",
        "", "Fold-specific feature CSVs, model-selection metadata, and trial IDs are under cache/<recording>/<analysis>/.",
        "The filename 'short' means the primary cohort. 'matched_short' and 'matched_long' are S1 only.",
        "", "## Interpretation", "",
        "Macro-F1 is averaged across six direction-specific F1 values. Confusion diagonals are recall, not F1. "
        "D/F and supporting bars show recording means and SDs. E shows actual paired recording differences with day-cluster bootstrap confidence intervals. "
        "Planned p values use paired day means and Holm correction across eight tests. Null intervals are shuffled-label references, not statistical tests against 1/6.",
        "", "This adapts the 2012 peak-power/frequency representation, not the original population LVQ decoder or its accuracy benchmark. "
        "The descriptive examples are selected by label-blind fit error, never decoding scores.",
        "", "## Preview", "", "The contact sheet is for navigation only; the standalone files above are the manuscript deliverables.",
        "", "![Preview](contact_sheet.png)", ""]
    (output / "README.md").write_text("\n".join(rows))
    names = ["A_task_window", "B_preview_M", "B_preview_T"]
    names += [f"C_confusion_{m}_{method}" for m in "MT" for method in ("power_frequency", "torus")]
    names += [f"{panel}_{name}_{m}" for panel, name in (("D", "decoding"), ("E", "added_value"),
               ("F", "direction_f1"), ("S1", "delay_robustness"), ("S2", "dimensionality")) for m in "MT"]
    fig, axes = plt.subplots(6, 3, figsize=(13.5, 17.4), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        with fitz.open(output / "panels" / (name+".pdf")) as doc:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            ax.imshow(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
        ax.set_title(name.replace("_", " "), fontsize=9)
    fig.suptitle("Pre-GO single-channel decoding | preview only", fontsize=13)
    fig.savefig(output / "contact_sheet.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel", choices=PANELS+["all"], default="all")
    args = parser.parse_args()
    tables = load_tables(args.output)
    for panel in selected_panels(args.panel):
        globals()["panel_"+panel.lower()](tables, args.output)
    if args.panel == "all":
        write_index(tables, args.output)


if __name__ == "__main__":
    main()
