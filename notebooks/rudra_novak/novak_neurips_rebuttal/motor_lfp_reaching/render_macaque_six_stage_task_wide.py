"""Wide, compact version of the corrected six-stage macaque task timeline."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from stage_geometry_decoding import ANCHORS, COLORS, LABELS, OUTPUT, STAGES


WIDTH, HEIGHT = 18.0, 3.35
INK, MUTED = "#171717", "#555555"
SOURCE = OUTPUT.parent / "macaque_task_six_stages_cue_offset_v2"
DESTINATION = OUTPUT.parent / "macaque_task_six_stages_cue_offset_correct/wide_timeline_v1"


def main() -> None:
    labels = pd.read_csv(SOURCE / "stage_window_labels.csv")
    assert list(labels.stage_name) == list(STAGES)
    assert list(labels.anchor) == list(ANCHORS)
    assert labels.begin_relative_ms.tolist() == [-300, 0, -300, 0, -300, 0]
    assert labels.end_relative_ms.tolist() == [0, 300, 0, 300, 0, 300]
    original = json.loads((SOURCE / "provenance.json").read_text())
    assert original["n_trials"] == 198 and original["n_windows"] == 1188

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.image_inline": True,
                         "pdf.fonttype": 42, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "text.color": INK,
                         "lines.solid_capstyle": "butt"})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()
    ax.imshow(plt.imread(SOURCE / "assets/macaque_screen.png"),
              extent=(.12, 2.62, 1.20, 2.66), interpolation="lanczos", zorder=1)
    ax.set_aspect("auto")
    texts = []

    def text(x, y, value, size=10, weight="normal", color=INK, ha="center"):
        texts.append(ax.text(x, y, value, fontsize=size, fontweight=weight,
                             color=color, ha=ha, va="center", zorder=5))

    def line(x, y, width=1, color=INK, **kwargs):
        ax.plot(x, y, lw=width, color=color, zorder=2, **kwargs)

    def interval(left, right, name):
        y = 1.96
        line([left, right], [y, y], .85)
        for x in (left, right):
            line([x, x], [y-.045, y+.045], .85)
        text((left+right)/2, 1.75, name, 10)
        text((left+right)/2, 1.56, "700 ms", 10, "bold")

    scale = .005
    start = 3.05
    tc_on = start + 700 * scale
    tc_off = tc_on + 200 * scale
    sc_on = tc_off + 700 * scale
    sc_off = sc_on + 55 * scale
    go = sc_off + 700 * scale
    anchors = (tc_on, tc_off, sc_on, sc_off, go, go)
    y = 2.46
    ax.annotate("", xy=(17.58, y), xytext=(2.95, y),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4,
                            "mutation_scale": 11, "shrinkA": 0, "shrinkB": 0}, zorder=3)
    for x in (start, go):
        line([x, x], [y-.12, y+.12], 1.2)
    for left, right in ((tc_on, tc_off), (sc_on, sc_off)):
        ax.add_patch(Rectangle((left, y-.045), right-left, .09,
                               facecolor=INK, edgecolor="none", zorder=4))
        for x in (left, right):
            line([x, x], [y-.085, y+.085], .8)
    text(start, 3.07, "Start", 11, "bold")
    text((tc_on+tc_off)/2, 3.13, "Temporal cue (TC)", 11, "bold")
    text((tc_on+tc_off)/2, 2.88, "200 ms tone", 9, color=MUTED)
    text((sc_on+sc_off)/2, 3.13, "Spatial cue (SC)", 11, "bold")
    text((sc_on+sc_off)/2, 2.88, "55 ms target cue", 9, color=MUTED)
    text(go, 3.07, "GO", 11, "bold")
    text(16.7, 3.07, "Reach", 11, "bold")
    interval(start, tc_on, "Initial hold")
    interval(tc_off, sc_on, "Delay 1 (D1)")
    interval(sc_off, go, "Delay 2 (D2)")
    text(.18, .80, "Monkey T | Short-delay trials", 9, color=MUTED, ha="left")
    text(.18, .59, "198 trials | Channel 7", 9, color=MUTED, ha="left")

    for anchor in dict.fromkeys(anchors):
        line([anchor, anchor], [1.12, y-.13], .65, "#AAAAAA", linestyle=(0, (2, 3)))
    for stage, (name, color) in enumerate(zip(LABELS, COLORS)):
        anchor = anchors[stage]
        left = anchor + int(labels.iloc[stage].begin_relative_ms) * scale
        right = anchor + int(labels.iloc[stage].end_relative_ms) * scale
        assert np.isclose((right-left)/scale, 300)
        ax.add_patch(Rectangle((left, 1.10), right-left, .26,
                               facecolor=color, edgecolor="none", zorder=3))
        if stage % 2:
            line([anchor, anchor], [1.10, 1.36], .7, "white")
        middle = (left+right)/2
        text(middle, .89, name, 9, "bold")
        text(middle, .69, "300 ms", 8.5)
        description = "before onset" if stage % 2 == 0 else \
                      "after cue end" if stage in (1, 3) else "after GO"
        text(middle, .49, description, 8, color=MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(artist.get_text(), artist.get_window_extent(renderer)) for artist in texts]
    for i, (value, box) in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), value
        for other_value, other in boxes[i+1:]:
            assert not box.overlaps(other), (value, other_value)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    stem = DESTINATION / "macaque_task_six_stages_wide"
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=600)
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        assert image.size == (int(WIDTH*600), int(HEIGHT*600))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
