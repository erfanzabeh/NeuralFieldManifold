"""Compact macaque task and middle-lag orientation panels for a two-row slide."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from stage_geometry_decoding import ANCHORS, COLORS, LABELS, OUTPUT, STAGES


SOURCE = OUTPUT.parent / "macaque_task_six_stages_cue_offset_v2"
DENSITIES = OUTPUT / "feature_distributions_v1/density_curves.csv"
DESTINATION = OUTPUT.parent / "macaque_task_six_stages_cue_offset_correct/slide_panels_v1"
INK, MUTED = "#202C32", "#58646A"


def styling() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK})


def export(fig, stem) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and (
                box.x0 < -1 or box.y0 < -1 or
                box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Clipped text: {artist.get_text()}")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600)
    plt.close(fig)


def task_panel() -> None:
    labels = pd.read_csv(SOURCE / "stage_window_labels.csv")
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    assert provenance["n_trials"] == 198 and provenance["n_windows"] == 1188
    assert list(labels.stage_name) == list(STAGES)
    assert list(labels.anchor) == list(ANCHORS)
    assert labels.begin_relative_ms.tolist() == [-300, 0, -300, 0, -300, 0]
    assert labels.end_relative_ms.tolist() == [0, 300, 0, 300, 0, 300]

    fig = plt.figure(figsize=(5.6, 4.1))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 5.6), ylim=(0, 4.1))
    ax.set_axis_off()
    ax.imshow(plt.imread(SOURCE / "assets/macaque_screen.png"),
              extent=(.12, 1.48, 2.94, 3.89), interpolation="lanczos", zorder=1)
    ax.set_aspect("auto")
    text_artists = []

    def text(x, y, value, size=9, weight="normal", color=INK, ha="center"):
        text_artists.append(ax.text(x, y, value, fontsize=size, fontweight=weight,
                                    color=color, ha=ha, va="center", zorder=5))

    def line(x, y, width=.9, color=INK, **kwargs):
        ax.plot(x, y, lw=width, color=color, zorder=2, **kwargs)

    text(1.63, 3.68, "Six-stage reaching task", 12, "bold", ha="left")
    text(1.63, 3.37, "Monkey T  |  short delay  |  198 trials  |  channel 7",
         8.5, color=MUTED, ha="left")

    scale = .0019
    start = .35
    tc_on = start + 700 * scale
    tc_off = tc_on + 200 * scale
    sc_on = tc_off + 700 * scale
    sc_off = sc_on + 55 * scale
    go = sc_off + 700 * scale
    anchors = (tc_on, tc_off, sc_on, sc_off, go, go)
    axis_y = 2.48
    ax.annotate("", xy=(5.53, axis_y), xytext=(.28, axis_y),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.15,
                            "mutation_scale": 10, "shrinkA": 0, "shrinkB": 0}, zorder=3)
    for position in (start, go):
        line([position, position], [axis_y-.10, axis_y+.10], 1.0)
    for left, right in ((tc_on, tc_off), (sc_on, sc_off)):
        ax.add_patch(Rectangle((left, axis_y-.043), right-left, .086,
                               facecolor=INK, edgecolor="none", zorder=4))
    text(start, 2.78, "Start", 9.5, "bold")
    text((tc_on+tc_off)/2, 2.80, "Temporal cue (TC)", 8.5, "bold")
    text((tc_on+tc_off)/2, 2.64, "200 ms tone", 7.5, color=MUTED)
    text((sc_on+sc_off)/2, 2.80, "Spatial cue (SC)", 8.5, "bold")
    text((sc_on+sc_off)/2, 2.64, "55 ms cue", 7.5, color=MUTED)
    text(go, 2.80, "GO", 10, "bold")

    def delay(left, right, label):
        y = 2.22
        line([left, right], [y, y], .7)
        for position in (left, right):
            line([position, position], [y-.035, y+.035], .7)
        text((left+right)/2, 2.02, label, 8.5)
        text((left+right)/2, 1.82, "700 ms", 9, "bold")

    delay(start, tc_on, "Initial hold")
    delay(tc_off, sc_on, "Delay 1")
    delay(sc_off, go, "Delay 2")
    for anchor in dict.fromkeys(anchors):
        line([anchor, anchor], [1.27, axis_y-.10], .55, "#B3B9BB",
             linestyle=(0, (2, 3)))
    for stage, (name, color) in enumerate(zip(LABELS, COLORS)):
        anchor = anchors[stage]
        left = anchor + int(labels.iloc[stage].begin_relative_ms) * scale
        right = anchor + int(labels.iloc[stage].end_relative_ms) * scale
        assert np.isclose((right-left)/scale, 300)
        ax.add_patch(Rectangle((left, 1.30), right-left, .24,
                               facecolor=color, edgecolor="none", zorder=3))
        midpoint = (left+right)/2
        text(midpoint, 1.08, name, 8.5, "bold")
        text(midpoint, .87, "300 ms", 8)
    text(.35, .46, "Post-TC/SC windows start at cue offset; post-GO starts at GO.",
         8.5, color=MUTED, ha="left")
    text(.35, .23, "Cue and delay durations are nominal; windows use recorded events.",
         8, color=MUTED, ha="left")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(artist.get_text(), artist.get_window_extent(renderer))
             for artist in text_artists]
    for index, (value, box) in enumerate(boxes):
        for other_value, other in boxes[index+1:]:
            if box.overlaps(other):
                raise ValueError(f"Overlapping labels: {value} / {other_value}")
    export(fig, DESTINATION / "task_six_stages_compact")


def geometry_panel() -> None:
    densities = pd.read_csv(DENSITIES)
    features = ("normal_y", "u_y")
    selected = densities[densities.feature.isin(features)]
    assert len(selected) == 2 * 6 * 512
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2))
    for ax, feature, title in zip(
            axes, features,
            (r"Annulus-plane normal, $n_2$", r"Major-axis direction, $u_2$")):
        values = selected[selected.feature == feature]
        grid = None
        peak = 0.0
        for stage, color in enumerate(COLORS):
            curve = values[values.stage == stage].sort_values("x")
            assert len(curve) == 512
            x, y = curve.x.to_numpy(), curve.density.to_numpy()
            assert np.isfinite(x).all() and np.isfinite(y).all() and (y >= 0).all()
            if grid is None:
                grid = x
            else:
                np.testing.assert_allclose(grid, x, rtol=0, atol=1e-12)
            peak = max(peak, float(y.max()))
            ax.fill_between(x, y, color=color, alpha=.075, linewidth=0)
            ax.plot(x, y, color=color, lw=1.35)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, peak*1.07)
        ax.set_title(title, fontsize=9.5, pad=7)
        ax.set_xlabel("Unit-vector component", fontsize=8.5)
        ticks = MaxNLocator(4, min_n_ticks=3).tick_values(grid[0], grid[-1])
        ax.set_xticks(ticks[(ticks >= grid[0]) & (ticks <= grid[-1])])
        ax.tick_params(length=2.5, labelsize=8)
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.grid(False)
    axes[0].set_ylabel("Density", fontsize=8.5)
    fig.suptitle(r"Orientation along $x(t-3\,\mathrm{ms})$",
                 y=.975, fontsize=10.5, fontweight="bold")
    handles = [Line2D([0], [0], color=color, lw=1.6, label=label)
               for color, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False,
               bbox_to_anchor=(.5, .875), fontsize=7.5,
               handlelength=1.4, columnspacing=.9)
    fig.subplots_adjust(left=.105, right=.97, bottom=.17, top=.64, wspace=.28)
    export(fig, DESTINATION / "geometry_middle_lag")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    styling()
    task_panel()
    geometry_panel()
    print(DESTINATION)


if __name__ == "__main__":
    main()
