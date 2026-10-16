"""Compact, cue-offset-correct task panel for the macaque stage figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/macaque_panel_E_task_v3"
MONKEY = ROOT / "outputs/macaque_task_six_stages_cue_offset_v2/assets/macaque_screen.png"
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")

INK = "#1C2C37"
MUTED = "#58636A"
RULE = "#A5AFB3"
FIG_W, FIG_H = 6.6, 3.7


def main() -> None:
    # Nominal protocol timing is only used to draw the schematic; analysis used recorded events.
    start, tc_on, tc_off, sc_on, sc_off, go, end = 0, 700, 900, 1600, 1655, 2355, 2655
    windows = ((400, tc_on), (tc_off, 1200), (1300, sc_on),
               (sc_off, 1955), (2055, go), (go, end))
    assert len(windows) == len(STAGES) == len(LABELS) == len(COLORS) == 6
    assert all(right - left == 300 for left, right in windows)
    assert all(windows[i][1] <= windows[i + 1][0] for i in range(5))
    assert windows[1][0] == tc_off and windows[3][0] == sc_off

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 8.2), ylim=(0, 3.45))
    ax.set_axis_off()
    ax.set_aspect("auto")
    texts = []

    def label(x: float, y: float, value: str, size: float = 9,
              weight: str = "normal", color: str = INK, ha: str = "center") -> None:
        texts.append(ax.text(x, y, value, ha=ha, va="center", color=color,
                             fontsize=size, fontweight=weight, zorder=5))

    def rule(x0: float, x1: float, y: float, color: str = INK,
             width: float = 0.9) -> None:
        ax.plot([x0, x1], [y, y], color=color, lw=width, zorder=2)

    def pos(ms: float) -> float:
        return 0.52 + 7.23 * ms / end

    # Keep the animal as a small task cue; the measured windows carry the composition.
    ax.imshow(plt.imread(MONKEY), extent=(0.13, 1.86, 2.14, 3.32),
              interpolation="lanczos", zorder=1)
    label(2.0, 3.10, "Delayed reaching task", 12.5, "bold", ha="left")
    label(2.0, 2.77, "Monkey T  |  198 short-delay trials  |  one LFP channel",
          8.8, color=MUTED, ha="left")

    y_axis = 1.93
    ax.annotate("", xy=(7.86, y_axis), xytext=(pos(start), y_axis),
                arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": INK,
                            "mutation_scale": 10, "shrinkA": 0, "shrinkB": 0},
                zorder=2)
    for ms in (start, go):
        x = pos(ms)
        ax.plot([x, x], [y_axis - .12, y_axis + .12], color=INK, lw=1.1, zorder=3)
    for left, right in ((tc_on, tc_off), (sc_on, sc_off)):
        ax.add_patch(Rectangle((pos(left), y_axis - .055), pos(right) - pos(left),
                               .11, facecolor=INK, edgecolor="none", zorder=4))

    label(pos(start), 2.20, "Start", 8.9, "bold")
    label(pos((tc_on + tc_off) / 2), 2.35, "Temporal cue", 9.1, "bold")
    label(pos((tc_on + tc_off) / 2), 2.16, "200 ms tone", 8.2, color=MUTED)
    label(pos((sc_on + sc_off) / 2), 2.35, "Spatial cue", 9.1, "bold")
    label(pos((sc_on + sc_off) / 2), 2.16, "55 ms target", 8.2, color=MUTED)
    label(pos(go), 2.21, "GO", 9.1, "bold")
    label(7.55, 2.21, "Reach", 8.9, "bold")

    for left, right, name in ((start, tc_on, "Initial hold"),
                              (tc_off, sc_on, "Delay 1"),
                              (sc_off, go, "Delay 2")):
        x0, x1 = pos(left), pos(right)
        rule(x0, x1, 1.67, RULE, 0.8)
        for x in (x0, x1):
            ax.plot([x, x], [1.64, 1.70], color=RULE, lw=0.8, zorder=2)
        label((x0 + x1) / 2, 1.48, name, 8.5)
        label((x0 + x1) / 2, 1.28, "700 ms", 8.6, "bold")

    for ms in (tc_on, tc_off, sc_on, sc_off, go):
        x = pos(ms)
        ax.plot([x, x], [1.04, y_axis - .09], color=RULE, lw=0.65,
                linestyle=(0, (2, 2)), zorder=1)

    for (left, right), name, color in zip(windows, LABELS, COLORS):
        x0, x1 = pos(left), pos(right)
        ax.add_patch(Rectangle((x0, .80), x1 - x0, .28,
                               facecolor=color, edgecolor="none", zorder=3))
        label((x0 + x1) / 2, .58, name, 8.6, "bold")

    label(4.11, .22, "Six 300-ms windows aligned to recorded event times",
          8.4, color=MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(artist.get_text(), artist.get_window_extent(renderer)) for artist in texts]
    for i, (value, box) in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), value
        for other_value, other in boxes[i + 1:]:
            assert not box.overlaps(other), (value, other_value)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "macaque_panel_E_task"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600)
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
