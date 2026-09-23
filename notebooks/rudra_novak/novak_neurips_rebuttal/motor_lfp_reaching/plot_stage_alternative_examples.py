"""Additional fixed-seed trial examples from existing stage clouds and fits."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_stage_geometry_decoding import outlines, save, style
from plot_stage_geometry_trajectories import trajectory_axis
from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT

DESTINATION = OUTPUT/"alternative_examples_v1"
SEED = 20260922


def select_trials(features):
    complete = features.groupby("original_trial_number").usable.all()
    eligible = complete[complete].index.to_numpy()
    eligible = eligible[~np.isin(eligible, [6, 9, 10])]
    return np.sort(np.random.default_rng(SEED).choice(eligible, 6, replace=False))


def main():
    folder = DESTINATION/"png"
    folder.mkdir(parents=True, exist_ok=True)
    source_paths = [OUTPUT/"tables/features.csv", OUTPUT/"inputs/windows.npz",
                    OUTPUT/"heldout_predictions.npz", OUTPUT/"config.json", OUTPUT/"tables/summary.csv"]
    features = pd.read_csv(source_paths[0])
    trials = select_trials(features)
    selected = features[features.original_trial_number.isin(trials)].copy()
    assert len(selected) == 36 and selected.usable.all()
    source_paths += [OUTPUT/"checkpoints"/f"window_{i:04d}.json" for i in selected.window_index]
    protected = {str(p): sha256(p) for p in source_paths}
    with np.load(OUTPUT/"inputs/windows.npz") as z:
        clouds = z["clouds"]
    results = {}
    extent = [np.max(np.abs(clouds[selected.window_index]))]
    for row in selected.itertuples():
        result = json.loads((OUTPUT/"checkpoints"/f"window_{row.window_index:04d}.json").read_text())
        result["usable"] = True
        results[row.window_index] = result
        extent.append(np.max(np.abs(outlines(result))))
    limit = float(max(extent)*1.08)
    selected[["window_index", "original_trial_number", "stage", "stage_name", "usable"]].to_csv(
        DESTINATION/"example_selection.csv", index=False)
    style()
    exports = []
    for trial in trials:
        example = selected[selected.original_trial_number == trial].sort_values("stage")
        for overlay in (False, True):
            fig = plt.figure(figsize=(8.4, 6.05))
            for row in example.itertuples():
                position = (row.stage % 2)*3 + row.stage//2 + 1
                ax = fig.add_subplot(2, 3, position, projection="3d")
                trajectory_axis(ax, clouds[row.window_index], results[row.window_index], row.stage, limit, overlay)
            fig.subplots_adjust(left=.015, right=.965, top=.94, bottom=.08, wspace=.03, hspace=.29)
            suffix = "fitted" if overlay else "observed"
            save(fig, folder, f"geometry_trial_{trial}_{suffix}", exports)
        print(f"Rendered trial {trial}", flush=True)
    assert len(exports) == 12 and not any(e["outside_text"] for e in exports)
    assert protected == {str(p): sha256(p) for p in source_paths}
    shutil.copy2(__file__, DESTINATION/Path(__file__).name)
    (DESTINATION/"caption.md").write_text(
        "Six additional original trials selected without replacement using random seed 20260922, "
        "from trials with six usable stage fits, excluding earlier examples 6, 9, and 10. "
        "No visual-separation or decoder-performance criterion was used. Each gallery compares "
        "all six stages of the same trial. These are illustrative complete-fit examples, not "
        "a new cohort or a claim of stage separation. All figures retain the existing m=3, "
        "tau=3 ms observed clouds, preprocessing, stage colors, camera angle, and saved annular fits. "
        "The axis limits are shared across these new examples and include every plotted point and "
        "fitted outline without clipping. No fitting, feature extraction, or decoding was rerun.\n")
    write_json(DESTINATION/"provenance.json", dict(seed=SEED, original_trial_ids=trials.tolist(),
               selection="random six complete-fit trials excluding original examples 6,9,10",
               protected_inputs=protected, inputs_unchanged=True, m=3, tau_ms=3,
               axis_limits=[-limit, limit], camera=dict(elevation=24, azimuth=-58),
               source_sha256=sha256(__file__), exports=exports))
    print(f"Validated {len(exports)} PNGs; saved data and decoding unchanged", flush=True)


if __name__ == "__main__":
    main()
