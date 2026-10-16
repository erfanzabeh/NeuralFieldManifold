"""Plot saved stage AMI curves only; no estimator, fitter, or decoder imports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "outputs/stage_ami_v1"


def selection(selections, scope, bins):
    row = selections[(selections.scope == scope) & (selections.bins == bins)]
    assert len(row) == 1
    value = row.iloc[0].selected_tau_ms
    return None if pd.isna(value) else int(value)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlim=(1, 100), xlabel="Delay (ms)", ylabel="AMI (nats)")
    ax.tick_params(length=3)
    ax.grid(False)


def save(fig, stem):
    fig.savefig(ROOT / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    curves = pd.read_csv(ROOT / "tables/ami_curves.csv")
    selected = pd.read_csv(ROOT / "tables/selected_delays.csv")
    config = json.loads((ROOT / "config.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "figure.facecolor": "white", "text.color": "#222222"})
    primary = curves[(curves.scope == "pooled") & (curves.bins == 48)]
    tau = selection(selected, "pooled", 48)
    fig, ax = plt.subplots(figsize=(5.7, 3.3), layout="constrained")
    ax.plot(primary.tau_ms, primary.ami_nats, color="#BBBBBB", lw=.9, label="Histogram estimate")
    ax.plot(primary.tau_ms, primary.smoothed_ami_nats, color="#263E4C", lw=1.5, label="Smoothed AMI")
    ax.axvline(3, color="#376D9B", ls="--", lw=1., label="Current delay: 3 ms")
    if tau is not None:
        ax.axvline(tau, color="#A32E26", lw=1., label=f"First minimum: {tau} ms")
        point = primary[primary.tau_ms == tau]
        ax.scatter(point.tau_ms, point.smoothed_ami_nats, c="#A32E26", s=22, zorder=5)
    style(ax)
    ax.set_title("Pooled task-stage windows")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "ami_pooled")

    stages = curves[(curves.bins == 48) & curves.scope.isin(config["stage_names"])]
    ymin, ymax = stages.smoothed_ami_nats.min(), stages.smoothed_ami_nats.max()
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.9), sharex=True, sharey=True,
                             layout="constrained")
    for ax, name, label, color in zip(axes.flat, config["stage_names"],
                                    config["stage_labels"], config["stage_colors"]):
        rows = stages[stages.scope == name]
        ax.plot(rows.tau_ms, rows.smoothed_ami_nats, color=color, lw=1.5)
        ax.axvline(3, color="#888888", ls="--", lw=.7)
        stage_tau = selection(selected, name, 48)
        if stage_tau is not None:
            point = rows[rows.tau_ms == stage_tau]
            ax.scatter(point.tau_ms, point.smoothed_ami_nats, c=color,
                       edgecolor="#333333", lw=.5, s=20, zorder=5)
        style(ax)
        ax.set_ylim(max(0., ymin-.03), ymax+.06)
        ax.set_title(f"{label}" + (f"  |  {stage_tau} ms" if stage_tau is not None else ""))
    save(fig, "ami_by_stage")

    fig, ax = plt.subplots(figsize=(5.7, 3.3), layout="constrained")
    for bins, color in ((32, "#38816A"), (48, "#263E4C"), (64, "#BE5A47")):
        rows = curves[(curves.scope == "pooled") & (curves.bins == bins)]
        bin_tau = selection(selected, "pooled", bins)
        ax.plot(rows.tau_ms, rows.smoothed_ami_nats, color=color, lw=1.3,
                label=f"{bins} bins" + (f": {bin_tau} ms" if bin_tau is not None else ": no minimum"))
        if bin_tau is not None:
            point = rows[rows.tau_ms == bin_tau]
            ax.scatter(point.tau_ms, point.smoothed_ami_nats, c=color, s=22, zorder=5)
    style(ax)
    ax.set_title("Histogram-bin sensitivity")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "ami_bin_sensitivity")
    (ROOT / "captions.md").write_text(
        "Pooled AMI: histogram estimate (48 bins) and sigma=1-ms smoothed curve; "
        "the first interior local minimum is marked alongside the adopted 3-ms delay. "
        "All 1,188 frozen windows contribute within-window lagged pairs.\n\n"
        "Stage curves: each curve pools 198 windows; filled markers indicate descriptive "
        "stage-specific first minima. Dashed lines mark 3 ms. Shared axis limits are used. "
        "These minima are not stage-specific parameters applied to the existing clouds.\n\n"
        "Bin sensitivity: 32, 48, and 64 bins, otherwise identical estimation and selection. "
        "Markers show each curve's first minimum. AMI is in nats. No curves are confidence "
        "intervals, and no refitting or decoding is performed by this plotting entrypoint.\n")
    (ROOT / "README.md").write_text(
        "# Independent stage AMI diagnostic\n\n"
        "- `ami_pooled.png`: primary pooled curve and suggested delay.\n"
        "- `ami_by_stage.png`: descriptive curves for all six corrected stages.\n"
        "- `ami_bin_sensitivity.png`: estimator sensitivity, without retuning.\n"
        "- `REPORT.md`: findings, fold choices, and limitations.\n"
        "- `tables/`: exact curves, selections, frozen window identities, and existing F1 context.\n\n"
        "Analysis: `stage_ami_analysis.py`; plot-only: `plot_stage_ami.py`. "
        "All previous outputs remain unchanged.\n")
    assert len(curves) == 1400 and np.isfinite(curves.smoothed_ami_nats).all()
    print(ROOT)


if __name__ == "__main__":
    main()
