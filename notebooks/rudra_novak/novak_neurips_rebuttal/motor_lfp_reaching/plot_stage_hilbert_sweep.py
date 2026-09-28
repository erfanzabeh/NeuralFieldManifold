"""Plot-only rendering of frozen Hilbert-envelope decoding results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from plot_stage_normalization_geometry import COLORS, LABELS, UNIT
from run_stage_additive_spectral import COLORS as METHOD_COLORS, LABELS as METHOD_LABELS, METHODS
from stage_geometry_decoding import STAGES, scores

ROOT = UNIT / "outputs/stage_hilbert_envelope_tau_sweep_v1"


def save(fig, name):
    fig.savefig(ROOT / name, dpi=600, facecolor="white")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def render():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "font.family": "DejaVu Sans"})
    selection = json.loads((ROOT / "selection.json").read_text())
    stages = pd.read_csv(ROOT / "tables/stage_f1.csv")
    table = stages[stages.tau_ms.between(1, 25)].pivot(index="stage_name", columns="tau_ms", values="f1").loc[list(STAGES), range(1, 26)]
    fig, ax = plt.subplots(figsize=(8.0, 2.7), layout="constrained")
    artist = ax.imshow(table, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_yticks(range(6), LABELS)
    ax.set_xticks(range(25), range(1, 26))
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Task stage")
    fig.colorbar(artist, ax=ax, label="F1", shrink=.85)
    for tau, linestyle, color in ((3, ":", "#222222"), (selection["f1_tau_ms"], "--", "#222222"), (selection["ami_tau_ms"], "-", "#376D9B")):
        if tau is not None and 1 <= tau <= 25:
            ax.axvline(tau - 1, color=color, ls=linestyle, lw=1.2)
    ax.set_title("Six-stage decoding across delays")
    save(fig, "stage_f1_tau_heatmap.png")

    curves = pd.read_csv(ROOT / "tables/ami_curves.csv")
    curve = curves[(curves.scope == "pooled") & (curves.bins == 48)]
    fig, ax = plt.subplots(figsize=(5.6, 2.8), layout="constrained")
    ax.plot(curve.tau_ms, curve.ami_nats, color="#B5B5B5", lw=1, label="AMI")
    ax.plot(curve.tau_ms, curve.smoothed_ami_nats, color="#376D9B", lw=1.6, label="Smoothed AMI")
    if selection["ami_tau_ms"] is not None:
        ax.axvline(selection["ami_tau_ms"], color="#376D9B", ls="--", lw=1,
                   label=f"AMI delay: {selection['ami_tau_ms']} ms")
    ax.set(xlabel="Delay (ms)", ylabel="AMI (nats)")
    ax.legend(frameon=False)
    clean(ax)
    save(fig, "ami_delay_selection.png")

    for tau in sorted({t for t in (selection["f1_tau_ms"], selection["ami_tau_ms"]) if t is not None}):
        out = ROOT / "selected" / f"tau_{tau:02d}"
        matrix = pd.read_csv(out / "recall_geometry.csv", index_col=0).to_numpy()
        with np.load(out / "predictions.npz") as saved:
            metric = scores(saved["labels"], saved["predictions"][:, METHODS.index("geometry")])
        np.testing.assert_allclose(matrix, metric["confusion_fraction"], atol=1e-12)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.)
        fig, ax = plt.subplots(figsize=(4.1, 3.6), layout="constrained")
        artist = ax.imshow(matrix, vmin=0, vmax=1, cmap="Reds")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i,j] > .5 else "#222222", fontsize=8)
        ax.set_xticks(range(6), LABELS, rotation=45, ha="right")
        ax.set_yticks(range(6), LABELS)
        ax.set(xlabel="Predicted stage", ylabel="True stage", title=f"Geometry decoding, delay {tau} ms")
        fig.colorbar(artist, ax=ax, label="Prediction fraction", shrink=.8, ticks=[0,.5,1])
        save(fig, f"tau_{tau:02d}_recall_confusion.png")

        stage = pd.read_csv(out / "per_stage_f1.csv")
        np.testing.assert_allclose(stage.f1, metric["per_stage_f1"], atol=1e-12)
        fig, ax = plt.subplots(figsize=(5.2, 2.9), layout="constrained")
        positions = np.arange(6)
        ax.bar(positions, stage.f1, color=COLORS, width=.68)
        ax.vlines(positions, stage.ci_low, stage.ci_high, color="#222222", lw=1)
        ax.hlines(stage.ci_low, positions-.08, positions+.08, color="#222222", lw=1)
        ax.hlines(stage.ci_high, positions-.08, positions+.08, color="#222222", lw=1)
        ax.set_xticks(positions, LABELS)
        ax.set(ylim=(0,1), ylabel="F1", title=f"Per-stage decoding, delay {tau} ms")
        clean(ax)
        save(fig, f"tau_{tau:02d}_per_stage_f1.png")

        summary = pd.read_csv(out / "summary.csv").set_index("method").loc[list(METHODS)]
        with np.load(out / "bootstrap.npz") as saved:
            low, high = np.quantile(saved["macro_f1"], [.025, .975], axis=0)
            np.testing.assert_allclose(summary.ci_low, low, atol=1e-12)
            np.testing.assert_allclose(summary.ci_high, high, atol=1e-12)
        for j, name in enumerate(METHODS):
            with np.load(out / "predictions.npz") as saved:
                np.testing.assert_allclose(summary.loc[name, "macro_f1"], scores(saved["labels"], saved["predictions"][:,j])["macro_f1"], atol=1e-12)
        fig, ax = plt.subplots(figsize=(8.2, 3.2), layout="constrained")
        positions = np.arange(len(METHODS))
        ax.bar(positions, summary.macro_f1, color=[METHOD_COLORS[m] for m in METHODS], width=.68, edgecolor="#444444", linewidth=.5)
        ax.vlines(positions, summary.ci_low, summary.ci_high, color="#222222", lw=1)
        ax.hlines(summary.ci_low, positions-.07, positions+.07, color="#222222", lw=1)
        ax.hlines(summary.ci_high, positions-.07, positions+.07, color="#222222", lw=1)
        for i, value in enumerate(summary.macro_f1):
            ax.text(i, max(value, summary.iloc[i].ci_high) + .016, f"{value:.3f}", ha="center", fontsize=8)
        ax.axhline(1/6, ls="--", color="#888888", lw=1)
        ax.set_xticks(positions, [METHOD_LABELS[m] for m in METHODS])
        ax.set(ylim=(0, max(.5, float(summary.ci_high.max())+.09)), ylabel="F1",
               title=f"Six-stage decoding, delay {tau} ms")
        clean(ax)
        save(fig, f"tau_{tau:02d}_feature_comparison.png")

    overall = pd.read_csv(ROOT / "tables/overall_f1.csv").set_index("tau_ms")
    report = ["# Hilbert-envelope stage decoding", "",
              "198 short-delay trials contributed six corrected 300-ms windows each. Geometry used m=3, K=1, and the unchanged 15-feature LDA pipeline.", "",
              f"The exploratory F1-selected delay was **{selection['f1_tau_ms']} ms**, with macro-averaged F1 **{selection['f1']:.4f}**.",
              f"The pooled AMI first-minimum delay was **{selection['ami_tau_ms']} ms**." if selection['ami_tau_ms'] is not None else "AMI had no interior minimum in 1-100 ms; no alternative delay was invented."]
    if selection["ami_tau_ms"] is not None:
        report.append(f"At the AMI delay, geometric F1 was **{overall.loc[selection['ami_tau_ms'], 'macro_f1']:.4f}**.")
    report += [f"New envelope-normalized tau=3 F1: **{overall.loc[3, 'macro_f1']:.4f}**; previous MAD tau=3: **0.2928**.", ""]
    for tau in sorted({t for t in (selection["f1_tau_ms"], selection["ami_tau_ms"]) if t is not None}):
        out = ROOT / "selected" / f"tau_{tau:02d}"
        summary = pd.read_csv(out / "summary.csv").set_index("method")
        record = json.loads((out / "metrics.json").read_text())
        usable = pd.read_csv(out / "fit_accounting.csv").unusable.sum()
        report.append(f"Delay {tau} ms: geometry 95% conditional interval {summary.loc['geometry','ci_low']:.3f}-{summary.loc['geometry','ci_high']:.3f}; nominal fixed-delay permutation p={record['permutation_p']:.4f}; {usable} unusable fits. Complete-trial sensitivity: {record['sensitivity']}.")
    report += ["", "Spectral baselines retain their original raw-signal definition. Bootstrap intervals resample whole trials, stratified by reach direction. The F1 winner is selected on these held-out scores, so its score, interval, and nominal permutation p-value are not selection-corrected or independently validated. AMI is a heuristic, not evidence of optimal topology.", "",
               "Full-trial envelopes use temporal context outside the windows: this is offline, single-recording analysis. No previous outputs were modified. No Betti calculation, PCA, PINN, or geometric parameter tuning was performed."]
    (ROOT / "REPORT.md").write_text("\n".join(report)+"\n")
    pngs = sorted(ROOT.glob("*.png"))
    (ROOT / "README.md").write_text("# Figure index\n\n" + "\n".join(f"- [{p.name}]({p.name})" for p in pngs) +
        "\n\nHeatmap: dotted line = original 3 ms, dashed black = F1 selection, solid blue = AMI selection when inside the sweep. F1 is macro-averaged; matrix diagonals are recall. Error intervals are conditional whole-trial bootstrap intervals. See REPORT.md.\n")
    (ROOT / "captions.md").write_text("F1 heatmap: per-stage F1 for native embeddings, same grouped folds at every delay.\n\nAMI: raw and smoothed pooled histogram AMI, first interior minimum.\n\nRecall matrices: held-out prediction fractions, row-normalized; diagonals are recall, not F1.\n\nPer-stage F1 and method comparison: conditional 95% whole-trial bootstrap intervals; six-stage macro averaging for method comparisons. The dashed 1/6 line is a balanced-class nominal reference, not an exact expected finite-sample F1.\n")
    dimensions = {p.name: Image.open(p).size for p in pngs}
    (ROOT / "figure_validation.json").write_text(json.dumps(dict(pngs=dimensions,
        plotted_scores_match_predictions=True, plotted_intervals_match_bootstrap=True,
        plotting_invokes_no_fitting_ami_or_decoding=True), indent=2)+"\n")


if __name__ == "__main__":
    render()
