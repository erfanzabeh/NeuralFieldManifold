"""Update the native task diagram with the six frozen 300-ms stage windows."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, PREVIOUS, STAGES, LABELS, COLORS

DESTINATION = OUTPUT.parent / "macaque_task_six_stages_v1"
ILLUSTRATION = OUTPUT.parent / "macaque_task_timeline_v3/assets/macaque_screen.png"
WIDTH, HEIGHT = 14.8, 4.45
INK, MUTED = "#171717", "#555555"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source_paths = [ILLUSTRATION, PREVIOUS / "inputs/alignment.json",
                    OUTPUT / "inputs/window_metadata.csv", OUTPUT / "inputs/event_samples.csv",
                    OUTPUT / "tables/features.csv", OUTPUT / "heldout_predictions.npz", OUTPUT / "config.json"]
    protected = {str(p): sha256(p) for p in source_paths}
    alignment = json.loads(source_paths[1].read_text())
    assert alignment["sampling_rate_hz"] == 1000 and alignment["short_delay_ms"] == 700
    assert alignment["recording"] == "monkeyT_session-y070316009-12_lfp-7"
    windows = pd.read_csv(source_paths[2])
    events = pd.read_csv(source_paths[3])
    assert len(windows) == 1188 and windows.original_trial_number.nunique() == 198
    for i, stage in enumerate(STAGES):
        rows = windows[windows.stage_name == stage]
        assert len(rows) == 198 and rows.stage.eq(i).all()
        assert (rows.end_sample_exclusive - rows.start_sample).eq(300).all()
        assert (rows.start_sample - rows.event_sample).eq(-300 if i % 2 == 0 else 0).all()
        assert (rows.end_sample_exclusive - rows.event_sample).eq(0 if i % 2 == 0 else 300).all()
        anchor = events.set_index("original_trial_number")[("TC_sample", "SC_sample", "GO_sample")[i//2]]
        np.testing.assert_array_equal(rows.event_sample, anchor.loc[rows.original_trial_number])
    (DESTINATION / "assets").mkdir(exist_ok=True)
    asset = DESTINATION / "assets/macaque_screen.png"
    shutil.copy2(ILLUSTRATION, asset)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none", "svg.image_inline": True, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "text.color": INK, "lines.solid_capstyle": "butt"})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()
    ax.imshow(plt.imread(asset), extent=(.08, 2.72, 1.78, 3.54), interpolation="lanczos", zorder=1)
    ax.set_aspect("auto")
    text_artists = []

    def label(x, y, text, size=12, bold=False, color=INK, ha="center"):
        text_artists.append(ax.text(x, y, text, fontsize=size,
                                   fontweight="bold" if bold else "normal", color=color,
                                   ha=ha, va="center", zorder=5))

    def line(x, y, lw=1.1, color=INK, **kwargs):
        ax.plot(x, y, lw=lw, color=color, zorder=2, **kwargs)

    def interval(left, right, name):
        y = 2.50
        line([left, right], [y, y], .9)
        for x in (left, right):
            line([x, x], [y-.05, y+.05], .9)
        label((left+right)/2, 2.24, name, 12)
        label((left+right)/2, 1.99, "700 ms", 12, bold=True)

    # Preserve nominal protocol intervals; data extraction uses the recorded onsets above.
    scale = .003
    start = 3.20
    tc_on = start + 700 * scale
    tc_off = tc_on + 200 * scale
    sc_on = tc_off + 700 * scale
    sc_off = sc_on + 55 * scale
    go = sc_off + 700 * scale
    y = 2.97
    ax.annotate("", xy=(14.30, y), xytext=(3.05, y),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6,
                            "mutation_scale": 13, "shrinkA": 0, "shrinkB": 0}, zorder=3)
    for x in (start, go):
        line([x, x], [y-.14, y+.14], 1.35)
    for left, right in ((tc_on, tc_off), (sc_on, sc_off)):
        ax.add_patch(Rectangle((left, y-.055), right-left, .11, facecolor=INK, edgecolor="none", zorder=4))
        for x in (left, right):
            line([x, x], [y-.11, y+.11], .9)
    label(start, 3.64, "Start", 14, bold=True)
    label((tc_on+tc_off)/2, 3.86, "Temporal cue (TC)", 14, bold=True)
    label((tc_on+tc_off)/2, 3.55, "200 ms tone", 12, color=MUTED)
    label((sc_on+sc_off)/2, 3.86, "Spatial cue (SC)", 14, bold=True)
    label((sc_on+sc_off)/2, 3.55, "55 ms target cue", 12, color=MUTED)
    label(go, 3.64, "GO", 14, bold=True)
    label(12.65, 3.64, "Reach", 14, bold=True)
    interval(start, tc_on, "Initial hold")
    interval(tc_off, sc_on, "Delay 1 (D1)")
    interval(sc_off, go, "Delay 2 (D2)")
    label(1.40, 1.52, "Monkey T | Short-delay trials", 10.5, color=MUTED)
    label(1.40, 1.25, "198 trials | Channel 7", 10, color=MUTED)

    sampling_rows = []
    for event, anchor in zip(("TC", "SC", "GO"), (tc_on, sc_on, go)):
        line([anchor, anchor], [1.18, y-.17], .8, "#A0A0A0", linestyle=(0, (2, 3)))
        label(anchor, .24, f"{event} onset", 9, color=MUTED)
    for stage, (name, color) in enumerate(zip(LABELS, COLORS)):
        anchor = (tc_on, sc_on, go)[stage//2]
        begin_ms = -300 if stage % 2 == 0 else 0
        end_ms = 0 if stage % 2 == 0 else 300
        left, right = anchor + begin_ms*scale, anchor + end_ms*scale
        ax.add_patch(Rectangle((left, 1.18), right-left, .32, facecolor=color, edgecolor="none", zorder=3))
        # Mark the shared event boundary without shifting or shortening either sampling window.
        if stage % 2 == 1:
            line([anchor, anchor], [1.18, 1.50], .8, "white")
        label((left+right)/2, .94, name, 11, bold=True)
        label((left+right)/2, .68, "300 ms", 10.5)
        sampling_rows.append(dict(stage=stage, stage_name=STAGES[stage], event=("TC", "SC", "GO")[stage//2],
                                  begin_relative_ms=begin_ms, end_relative_ms=end_ms,
                                  drawing_left=left, drawing_right=right, color=color))
    label(14.30, .24, "Nominal task timing; reach schematic", 8.5, color=MUTED, ha="right")
    for row in sampling_rows:
        assert np.isclose((row["drawing_right"]-row["drawing_left"])/scale, 300)
    for a, b in zip(sampling_rows[:-1], sampling_rows[1:]):
        assert a["drawing_right"] <= b["drawing_left"] + 1e-12
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(a.get_text(), a.get_window_extent(renderer)) for a in text_artists]
    for i, (text, box) in enumerate(boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), text
        for other_text, other in boxes[i+1:]:
            assert not box.overlaps(other), (text, other_text)
    stem = DESTINATION / "macaque_task_six_stages"
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=600)
    fig.savefig(DESTINATION / "preview.png", dpi=180)
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as im:
        assert im.size == (8880, 2670)
        assert all(abs(dpi - 600) < .1 for dpi in im.info["dpi"])
    with fitz.open(stem.with_suffix(".pdf")) as document:
        assert len(document) == 1
        text = document[0].get_text()
        assert all(label in text for label in LABELS) and text.count("300 ms") == 6
    assert protected == {p: sha256(p) for p in protected}, "Protected source changed"
    assert sha256(asset) == sha256(ILLUSTRATION)
    pd.DataFrame(sampling_rows).to_csv(DESTINATION / "stage_window_labels.csv", index=False)
    shutil.copy2(__file__, DESTINATION / Path(__file__).name)
    (DESTINATION / "caption.md").write_text(
        "Monkey T short-delay reaching task with the six analyzed 300-ms windows: immediately before "
        "and after temporal-cue onset, spatial-cue onset, and GO. Pre windows are [-300,0) ms; "
        "post windows are [0,300) ms. Every trial contributes all six windows. Colors match the "
        "stage geometry and decoding panels. Data: session y070316009-12, channel 7, 198 trials.\n\n"
        "Task durations shown are nominal protocol timings: 700-ms initial hold, 200-ms temporal cue, "
        "700-ms delay 1 from TC offset to SC onset, 55-ms spatial cue, and 700-ms delay 2 from SC "
        "offset to GO. Actual windows were aligned to recorded onsets, not these nominal intervals. "
        "Measured TC-to-SC onset intervals range 878-895 ms; SC-to-GO intervals range 737-747 ms. "
        "Reach position is schematic; the post-GO window can contain reaction and movement. "
        "Post-TC and post-SC windows include activity after their respective cues have ended. "
        "The original illustration is reused unchanged; text, lines, and windows are native artwork. "
        "No preprocessing, fitting, feature extraction, or decoding was changed.\n")
    write_json(DESTINATION / "provenance.json", dict(
        source_sha256=sha256(__file__), source_hashes=protected, source_unchanged=True,
        illustration_unchanged=True, recording=alignment["recording"], n_trials=198, n_windows=1188,
        nominal_durations_ms=dict(initial_hold=700, temporal_cue=200, delay_1=700, spatial_cue=55, delay_2=700),
        recorded_interval_ms=dict(TC_to_SC=[int((events.SC_sample-events.TC_sample).min()),
                                           int((events.SC_sample-events.TC_sample).max())],
                                  SC_to_GO=[int((events.GO_sample-events.SC_sample).min()),
                                            int((events.GO_sample-events.SC_sample).max())]),
        sampling_windows=sampling_rows, data_windows_validated=True, text_overlap_checked=True,
        fitting_called=False, decoding_called=False,
        exports=[dict(path=str(stem.with_suffix(f".{ext}")), sha256=sha256(stem.with_suffix(f".{ext}")))
                 for ext in ("png", "svg", "pdf")]))
    print(f"Saved and validated six-stage task diagram: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
