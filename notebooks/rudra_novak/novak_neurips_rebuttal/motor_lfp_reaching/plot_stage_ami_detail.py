"""Magnified views of saved AMI curves; no analysis or parameter selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_stage_ami import ROOT, save, selection, style


def main():
    curves = pd.read_csv(ROOT / "tables/ami_curves.csv")
    chosen = pd.read_csv(ROOT / "tables/selected_delays.csv")
    config = json.loads((ROOT / "config.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10})
    primary = curves[(curves.scope == "pooled") & (curves.bins == 48)]
    selected = selection(chosen, "pooled", 48)
    fig, ax = plt.subplots(figsize=(5.7, 3.3), layout="constrained")
    ax.plot(primary.tau_ms, primary.ami_nats, color="#BBBBBB", lw=.9, label="Histogram estimate")
    ax.plot(primary.tau_ms, primary.smoothed_ami_nats, color="#263E4C", lw=1.5, label="Smoothed AMI")
    if selected is not None:
        ax.axvline(selected, color="#A32E26", lw=1., label=f"First minimum: {selected} ms")
        point = primary[primary.tau_ms == selected]
        ax.scatter(point.tau_ms, point.smoothed_ami_nats, c="#A32E26", s=22, zorder=5)
    style(ax)
    ax.set(xlim=(10, 40), ylim=(0, .14), title="Pooled AMI (10-40 ms)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "ami_pooled_detail")

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.9), sharex=True, sharey=True,
                             layout="constrained")
    for ax, name, label, color in zip(axes.flat, config["stage_names"],
                                    config["stage_labels"], config["stage_colors"]):
        rows = curves[(curves.scope == name) & (curves.bins == 48)]
        ax.plot(rows.tau_ms, rows.ami_nats, color="#BBBBBB", lw=.7)
        ax.plot(rows.tau_ms, rows.smoothed_ami_nats, color=color, lw=1.5)
        tau = selection(chosen, name, 48)
        if tau is not None:
            point = rows[rows.tau_ms == tau]
            ax.scatter(point.tau_ms, point.smoothed_ami_nats, c=color,
                       edgecolor="#333333", lw=.5, s=20, zorder=5)
        style(ax)
        ax.set(xlim=(10, 70), ylim=(0, .25), title=label + (f"  |  {tau} ms" if tau else ""))
    save(fig, "ami_by_stage_detail")
    path = Path(__file__)
    (ROOT / "detail_plot_provenance.json").write_text(json.dumps({
        "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pooled_limits": {"delay_ms": [10, 40], "ami_nats": [0, .14]},
        "stage_limits": {"delay_ms": [10, 70], "ami_nats": [0, .25]},
        "input": "tables/ami_curves.csv", "estimation_called": False,
    }, indent=2) + "\n")
    print(ROOT / "ami_pooled_detail.png")


if __name__ == "__main__":
    main()
