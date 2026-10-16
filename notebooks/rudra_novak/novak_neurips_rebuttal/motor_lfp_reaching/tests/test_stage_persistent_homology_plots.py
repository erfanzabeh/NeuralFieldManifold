"""Plot-only checks, with no computation or alteration of the frozen experiment."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_stage_persistent_homology as plot
import run_stage_persistent_homology as ph


def test_curve_panel_uses_saved_means():
    rows = []
    for stage in range(6):
        for j, distance in enumerate(np.linspace(0, 3, 512)):
            mean = (stage+1)*np.exp(-distance)
            rows.append(dict(stage=stage, stage_name=ph.STAGES[stage], grid_index=j,
                distance=distance, mean=mean, q25=mean*.8, q75=mean*1.2, n_windows=198,
                homology_dimension=1, distance_convention="raw", cohort="available"))
    table = pd.DataFrame(rows)
    plot.style()
    fig, used = plot.curve_panel(table, 1, "raw")
    assert len(fig.axes[0].lines) == 6
    for stage, line in enumerate(fig.axes[0].lines):
        np.testing.assert_array_equal(line.get_ydata(), table.loc[table.stage == stage, "mean"])
    assert len(used) == len(table)
    plt.close(fig)


def test_png_legibility_and_no_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "OUTPUT", tmp_path)
    monkeypatch.setattr(ph, "compute_diagrams", lambda *_: (_ for _ in ()).throw(AssertionError("Ripser called")))
    (tmp_path / "tables").mkdir()
    (tmp_path / "captions").mkdir()
    plot.style()
    measures = pd.DataFrame(dict(window_index=np.arange(1188), stage=np.tile(np.arange(6), 198),
        longest_H1_lifetime=np.linspace(.05, 1.4, 1188)))
    fig, used = plot.distribution_panel(measures, "longest_H1_lifetime")
    manifest = []
    plot.save_panel("distribution", fig, used, "test", manifest)
    assert manifest[0]["outside_text"] == []
    assert len(used) == 1188
    intervals = pd.DataFrame(dict(homology_dimension=[0, 0, 1, 1, 2],
        birth=[0., 0., .1, .3, .8], death=[np.inf, .8, 1.2, .5, .85]))
    intervals["lifetime"] = intervals.death - intervals.birth
    fig = plot.persistence_panel(intervals, 2, 1.2)
    plot.save_panel("diagram", fig, intervals, "test", manifest)
    fig, bars = plot.barcode_panel(intervals, 2, 1.2, 20)
    assert len(bars) == 2
    np.testing.assert_array_equal(bars.barcode_rank, [1, 2])
    assert bars.lifetime.is_monotonic_decreasing
    plot.save_panel("barcode", fig, bars, "test", manifest)
    assert len(manifest) == 3


def test_summary_failure_and_normalization(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "SOURCE", tmp_path)
    (tmp_path / "tables").mkdir()
    table = pd.DataFrame(dict(window_index=np.arange(12), trial_index=np.repeat([0, 1], 6),
        original_trial_number=np.repeat([6, 9], 6), stage=np.tile(np.arange(6), 2),
        stage_name=list(ph.STAGES)*2, direction=np.ones(12, dtype=int)))
    pd.DataFrame(dict(window_index=np.arange(12), usable=np.ones(12, dtype=bool))).to_csv(
        tmp_path / "tables/fit_parameters_diagnostics.csv", index=False)
    cloud = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    diagrams = [np.array([[0., 1.], [0., np.inf]]), np.array([[1., 2.]]), np.empty((0, 2))]
    def read(index, *_):
        return (dict(status="failed" if index == 0 else "success", reason="failure" if index == 0 else "",
                     elapsed_seconds=.1), None if index == 0 else diagrams)
    monkeypatch.setattr(ph, "read_checkpoint", read)
    frames, arrays, grids = ph.summarize_data(np.repeat(cloud[None], 12, axis=0), table)
    m = frames["window_measurements"]
    assert np.isnan(m.iloc[0].longest_H1_lifetime)
    assert np.isnan(arrays["raw_H1"][0]).all()
    np.testing.assert_allclose(m.iloc[1].normalized_H1_lifetime, np.sqrt(2))
    summary = frames["betti_summary"]
    matched = summary[summary.cohort == "matched_six_stages"]
    assert matched.n_windows.eq(1).all()
    assert grids["raw_H2"]["fallback_to_H0"]
    assert not arrays["raw_H2"][1:].any()
    assert set(frames["lifetime_summary"].cohort) == {"available", "matched_six_stages"}
