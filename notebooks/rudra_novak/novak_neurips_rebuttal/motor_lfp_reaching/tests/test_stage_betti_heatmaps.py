"""Checks for rectangular, plot-only stage Betti contrasts."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_stage_betti_heatmaps as plot


def example():
    stage = np.tile(np.arange(6), 2)
    metadata = pd.DataFrame(dict(window_index=np.arange(12), trial_index=np.repeat([0, 1], 6),
        stage=stage, stage_name=list(plot.STAGES)*2, status="success"))
    curves = np.array([[0, s+1, 0] for s in stage], dtype=float)
    return curves, np.array([0., .5, 1.]), metadata


def test_pooled_mean_is_at_each_distance_not_row_normalization():
    curves, grid, metadata = example()
    table = plot.deviation_table(curves, grid, metadata, "raw")
    np.testing.assert_allclose(table.pooled_mean_betti, np.tile([0, 3.5, 0], 6))
    np.testing.assert_allclose(table.loc[table.grid_index == 1, "delta_betti"], np.arange(6)-2.5)
    assert table.loc[table.grid_index != 1, "delta_betti"].eq(0).all()


def test_pooled_reference_weights_windows_not_stage_means():
    curves, grid, metadata = example()
    curves, metadata = curves[:-1], metadata.iloc[:-1]
    table = plot.deviation_table(curves, grid, metadata, "raw")
    np.testing.assert_allclose(table.loc[table.grid_index == 1, "pooled_mean_betti"], 36/11)


@pytest.mark.parametrize("invalid", ["order", "duplicate", "failure", "nonfinite", "label"])
def test_reject_invalid_inputs(invalid):
    curves, grid, metadata = example()
    if invalid == "order":
        metadata = metadata.iloc[::-1]
    elif invalid == "duplicate":
        metadata.loc[6, "trial_index"] = 0
    elif invalid == "failure":
        metadata.loc[0, "status"] = "failed"
    elif invalid == "nonfinite":
        curves[0, 1] = np.nan
    else:
        metadata.loc[0, "stage_name"] = "post_TC"
    with pytest.raises(ValueError):
        plot.deviation_table(curves, grid, metadata, "raw")


def test_saved_results_and_display_zoom(tmp_path):
    plot.style()
    tables = plot.load_tables()
    limit = max(1., float(np.ceil(max(t.delta_betti.abs().max() for t in tables.values()))))
    for mode, table in tables.items():
        assert len(table) == 6*512 and table.n_windows.eq(198).all()
        full, ax, mesh = plot.heatmap_panel(table, limit)
        zoom, zoom_ax, zoom_mesh = plot.heatmap_panel(table, limit, zoom=True)
        np.testing.assert_array_equal(mesh.get_array(), zoom_mesh.get_array())
        assert mesh.get_clim() == zoom_mesh.get_clim() == (-limit, limit)
        assert ax.get_xlim()[1] == table.distance.max()
        assert zoom_ax.get_xlim() == (0., 1.)
        assert [tick.get_text() for tick in ax.get_yticklabels()] == list(plot.LABELS)
        info = plot.save_panel(zoom, tmp_path / f"{mode}.png")
        assert info["outside_text"] == []
        plt.close(full)
    assert "ripser" not in sys.modules
