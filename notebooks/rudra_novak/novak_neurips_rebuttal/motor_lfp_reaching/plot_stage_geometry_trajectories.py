"""Line-only visual variants of frozen stage geometry; no fitting or decoding."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_stage_geometry_decoding import cloud_axis, save, style
from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT

DESTINATION = OUTPUT/"trajectory_style_v1"


def trajectory_axis(ax, cloud, result, stage, limit, overlay):
    cloud_axis(ax, cloud, result, stage, limit, overlay)
    for dots in list(ax.collections):
        dots.remove()
    observed = ax.lines[0]
    observed.set_linewidth(1.0)
    observed.set_alpha(1.0)
    observed.set_solid_capstyle("round")
    observed.set_solid_joinstyle("round")
    observed.set_antialiased(True)
    # Preserve the actual time-ordered samples; rounded strokes are not smoothing.
    np.testing.assert_array_equal(np.column_stack(observed.get_data_3d()), cloud)
    assert not ax.collections


def main():
    folder = DESTINATION/"png"
    folder.mkdir(parents=True, exist_ok=True)
    protected = {str(p): sha256(p) for p in sorted(OUTPUT.rglob("*"))
                 if p.is_file() and not p.is_relative_to(DESTINATION)}
    table = pd.read_csv(OUTPUT/"tables/features.csv")
    selected = pd.read_csv(OUTPUT/"tables/example_selection.csv")
    scales = pd.read_csv(OUTPUT/"tables/example_geometry_scales.csv")
    limit = float(scales.maximum.iloc[0])
    np.testing.assert_allclose(scales.maximum, limit)
    np.testing.assert_allclose(scales.minimum, -limit)
    table = table[table.window_index.isin(selected.window_index)]
    assert set(table.original_trial_number) == {6, 9, 10} and len(table) == 18
    with np.load(OUTPUT/"inputs/windows.npz") as saved:
        clouds = saved["clouds"]
    results = {}
    for row in table.itertuples():
        result = json.loads((OUTPUT/"checkpoints"/f"window_{row.window_index:04d}.json").read_text())
        result["usable"] = bool(row.usable)
        results[row.window_index] = result
    style()
    exports = []
    for trial in (6, 9, 10):
        example = table[table.original_trial_number == trial].sort_values("stage")
        for overlay in (False, True):
            suffix = "fitted" if overlay else "observed"
            fig = plt.figure(figsize=(8.4, 6.05))
            for row in example.itertuples():
                position = (row.stage % 2)*3+row.stage//2+1
                ax = fig.add_subplot(2, 3, position, projection="3d")
                trajectory_axis(ax, clouds[row.window_index], results[row.window_index], row.stage, limit, overlay)
            fig.subplots_adjust(left=.015, right=.965, top=.94, bottom=.08, wspace=.03, hspace=.29)
            save(fig, folder, f"geometry_trial_{trial}_{suffix}", exports)
            for row in example.itertuples():
                fig = plt.figure(figsize=(3.0, 2.9))
                ax = fig.add_subplot(projection="3d")
                trajectory_axis(ax, clouds[row.window_index], results[row.window_index], row.stage, limit, overlay)
                fig.subplots_adjust(left=.03, right=.87, top=.91, bottom=.13)
                save(fig, folder, f"geometry_trial_{trial}_{row.stage_name}_{suffix}", exports)
        print(f"Rendered trial {trial}: observed and fitted variants", flush=True)
    assert len(exports) == 42 and not any(e["outside_text"] for e in exports)
    assert all(sha256(p) == h for p, h in protected.items()), "An existing result changed"
    shutil.copy2(__file__, DESTINATION/Path(__file__).name)
    (DESTINATION/"caption.md").write_text(
        "The same observed clouds and saved annular fits as the original six-stage panels, "
        "rendered with continuous 1-point, opaque, stage-colored trajectories instead of dot markers. "
        "Every trajectory joins the 294 observed samples in temporal order without interpolation, "
        "smoothing, resampling, or changes to coordinates. The fitted outer solid and inner dashed "
        "boundaries are unchanged. Stage colors, grid-free axes, limits, viewpoints, and example "
        "trials (6, 9, 10) are unchanged. These lines are not AR-predicted signals. No fitting, "
        "feature extraction, or decoding was run. Earlier outputs are preserved.\n")
    write_json(DESTINATION/"provenance.json", dict(source_sha256=sha256(__file__),
               base_plot_source_sha256=sha256(Path(__file__).with_name("plot_stage_geometry_decoding.py")),
               protected_files=protected, previous_files_unchanged=True, exports=exports,
               style=dict(line_width_points=1., alpha=1., markers=False, smoothing=False,
                          grid=False, axis_limits=[-limit, limit], elevation=24, azimuth=-58)))
    print(f"Validated {len(exports)} PNGs; preserved {len(protected)} previous files", flush=True)


if __name__ == "__main__":
    main()
