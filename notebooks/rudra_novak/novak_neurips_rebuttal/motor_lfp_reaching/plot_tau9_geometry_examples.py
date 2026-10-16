"""Plot saved native tau=9 stage clouds and fitted annular boundaries; no fitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from motor_lfp_utils import lag_embed
from prego_geometric_fits import array_hash, sha256, write_json
from stage_geometry_decoding import COLORS, LABELS, OUTPUT as SOURCE, STAGES, UNIT

ROOT = UNIT / "outputs/tau_sweep_v1"
DESTINATION = ROOT / "examples_tau9"
ORIGINAL_TRIALS = (6, 9, 10)
TAU = 9


def outlines(fit):
    angle = np.linspace(0, 2 * np.pi, 361)
    center = np.asarray(fit["center"])
    major = np.asarray(fit["u_axis"])
    minor = np.asarray(fit["v_axis"])
    return [center + (a * np.cos(angle))[:, None] * major
            + (b * np.sin(angle))[:, None] * minor
            for a, b in ((fit["R1"], fit["R2"]),
                         (fit["R1_in"], fit["R2_in"]))]


def inputs():
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv")
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        processed = saved["processed"]
    entries = []
    for trial in ORIGINAL_TRIALS:
        rows = metadata[metadata.original_trial_number == trial].set_index("stage_name")
        if set(rows.index) != set(STAGES) or len(rows) != 6:
            raise ValueError(f"Incomplete original trial {trial}")
        for stage in STAGES:
            row = rows.loc[stage]
            index = int(row.window_index)
            points = lag_embed(processed[index], dim=3, tau=TAU)
            fit_path = ROOT / "checkpoints/native/tau_09" / f"window_{index:04d}.json"
            result = json.loads(fit_path.read_text())
            if result["cloud_sha256"] != array_hash(points) or result["tau_ms"] != TAU:
                raise ValueError(f"Cloud/fit mismatch for window {index}")
            if points.shape != (282, 3) or result["status"] != "returned":
                raise ValueError(f"Unusable example fit for window {index}")
            entries.append(dict(trial=trial, stage=stage, index=index,
                                points=points, boundaries=outlines(result["fit"]),
                                normalized_error=result["summary"]["normalized_3d_set_error"],
                                fit_path=fit_path, cloud_sha256=array_hash(points)))
    return entries


def plot_trial(trial, entries, limit):
    fig = plt.figure(figsize=(11.4, 7.7), facecolor="white")
    fig.suptitle(f"Monkey T | Original trial {trial} | tau = 9 ms", x=.5, y=.975,
                 fontsize=15, color="#263238")
    for stage_index, stage in enumerate(STAGES):
        row = next(item for item in entries if item["trial"] == trial and item["stage"] == stage)
        ax = fig.add_subplot(2, 3, stage_index + 1, projection="3d", proj_type="ortho")
        points = row["points"]
        color = COLORS[stage_index]
        ax.plot(*points.T, lw=.72, color=color, alpha=.46, zorder=1)
        ax.scatter(*points.T, s=3.3, color=color, alpha=.75, depthshade=False,
                   linewidths=0, zorder=2)
        outer, inner = row["boundaries"]
        ax.plot(*outer.T, color="#202c34", lw=1.45, zorder=4)
        ax.plot(*inner.T, color="#202c34", lw=1.1, linestyle="--", zorder=4)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()
        ax.set_title(LABELS[stage_index], fontsize=11, color="#263238", pad=0)
    handles = [Line2D([0], [0], color="#71818a", linewidth=1.5, marker="o", markersize=4,
                      label="Observed lag trajectory"),
               Line2D([0], [0], color="#202c34", linewidth=1.5,
                      label="Fitted outer boundary"),
               Line2D([0], [0], color="#202c34", linewidth=1.2, linestyle="--",
                      label="Fitted inner boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .015), fontsize=9)
    fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.09, wspace=.02, hspace=.07)
    return fig


def main():
    if not (ROOT / "validation.json").exists():
        raise ValueError("The tau sweep must be validated before examples are rendered")
    entries = inputs()
    limit = max(float(np.max(np.abs(points)))
                for row in entries for points in [row["points"], *row["boundaries"]]) * 1.05
    DESTINATION.mkdir(exist_ok=True)
    records = []
    for trial in ORIGINAL_TRIALS:
        fig = plot_trial(trial, entries, limit)
        path = DESTINATION / f"tau9_native_trial_{trial}.png"
        fig.savefig(path, dpi=600, facecolor="white")
        plt.close(fig)
        print(path)
        records.extend(dict(original_trial_number=trial, stage_name=row["stage"],
                            window_index=row["index"], point_count=282,
                            normalized_3d_set_error=row["normalized_error"],
                            cloud_sha256=row["cloud_sha256"])
                       for row in entries if row["trial"] == trial)
    pd.DataFrame(records).to_csv(DESTINATION / "examples.csv", index=False)
    write_json(DESTINATION / "provenance.json", dict(tau_ms=TAU, variant="native",
               original_trial_numbers=list(ORIGINAL_TRIALS), common_axis_limit=limit,
               processed_windows_sha256=sha256(SOURCE / "inputs/windows.npz"),
               fit_checkpoints_verified=True, refit=False,
               figure_files=[f"tau9_native_trial_{trial}.png" for trial in ORIGINAL_TRIALS]))


if __name__ == "__main__":
    main()
