"""Render per-stage F1 as a one-row heatmap from frozen held-out predictions."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from plot_stage_geometry_decoding import INK, save, style
from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, STAGES, LABELS, scores


def main():
    destination = OUTPUT/"f1_heatmap_v1"
    destination.mkdir(exist_ok=True)
    paths = [OUTPUT/"heldout_predictions.npz", OUTPUT/"tables/per_stage_f1.csv",
             OUTPUT/"config.json", OUTPUT/"tables/summary.csv"]
    before = {str(path): sha256(path) for path in paths}
    table = pd.read_csv(paths[1]).set_index("stage_name").loc[list(STAGES)]
    with np.load(paths[0]) as saved:
        y, pred = saved["labels"], saved["predictions"]
    values = scores(y, pred)["per_stage_f1"]
    np.testing.assert_allclose(values, table.f1)
    np.testing.assert_allclose(values, f1_score(y, pred, labels=np.arange(6), average=None, zero_division=0))
    table.to_csv(destination/"stage_f1_values.csv")

    style()
    fig, ax = plt.subplots(figsize=(6.2, 1.6))
    cmap = LinearSegmentedColormap.from_list("stage_f1", ["#FFF9F6", "#E9A18D", "#941F1A"])
    im = ax.imshow(values[None, :], vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
    for j, value in enumerate(values):
        ax.text(j, 0, f"{value:.3f}", ha="center", va="center", fontsize=11,
                color="white" if value > .6 else INK)
    ax.set_xticks(range(6), LABELS, fontsize=9)
    ax.set_yticks([0], ["F1"], fontsize=10)
    ax.tick_params(length=0, pad=8)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.grid(which="minor", axis="x", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.022, pad=.035, ticks=[0, .5, 1])
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=.075, right=.945, bottom=.30, top=.80)
    exports = []
    save(fig, destination, "stage_f1_heatmap", exports)
    assert not exports[0]["outside_text"]
    after = {str(path): sha256(path) for path in paths}
    assert before == after, "Source results changed"
    (destination/"caption.md").write_text(
        "Per-stage F1 from the saved pooled held-out predictions of the 15D geometric stage decoder. "
        "Each cell combines precision and recall for that stage: F1 = 2TP/(2TP+FP+FN). "
        "This is a per-class performance heatmap, not a true-versus-predicted confusion matrix; "
        "there are no pairwise off-diagonal F1 entries. The color scale is fixed to 0-1. "
        "Five-fold evaluation grouped all six windows of each trial together. "
        "Conditional bootstrap intervals remain available in stage_f1_values.csv. "
        "No geometric fitting or classifier training was run for this plot.\n")
    shutil.copy2(__file__, destination/Path(__file__).name)
    write_json(destination/"provenance.json", dict(input_hashes=before, source_sha256=sha256(__file__),
               inputs_unchanged=True, sklearn_f1_verified=True, exports=exports))
    print(table.f1.to_string())
    print(destination/"stage_f1_heatmap.png")


if __name__ == "__main__":
    main()
