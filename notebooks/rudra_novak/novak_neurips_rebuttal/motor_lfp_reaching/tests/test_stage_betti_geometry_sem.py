"""Check exact Betti integration and the saved cohort before rendering."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_stage_betti_geometry_sem as plot


def test_exact_integral_handles_infinite_component_and_common_endpoint():
    meta = pd.DataFrame(dict(window_index=[0, 1], stage_name=["pre_TC", "post_TC"],
        original_trial_number=[6, 6], centered_rms=[2., 1.],
        H0_intervals=[1, 1], H1_intervals=[1, 1], H2_intervals=[0, 0]))
    intervals = pd.DataFrame(dict(window_index=[0, 0, 1, 1],
        stage_name=["pre_TC", "pre_TC", "post_TC", "post_TC"],
        original_trial_number=[6]*4, homology_dimension=[0, 1, 0, 1],
        birth=[0., .2, 0., .2], death=[np.inf, .8, np.inf, .8]))
    raw = plot.average_betti_counts(intervals, meta, 1., normalized=False)
    scaled = plot.average_betti_counts(intervals, meta, 1., normalized=True)
    np.testing.assert_allclose(raw, [[1., .6, 0.], [1., .6, 0.]])
    np.testing.assert_allclose(scaled, [[1., .3, 0.], [1., .6, 0.]])


def test_stage_sem_uses_actual_stage_count():
    data = pd.DataFrame(dict(stage=[0, 0, 0, 1, 1, 1], score=[1., 2., 3., 4., 5., 6.]))
    # Populate the remaining stages to exercise the exact six-stage contract.
    data = pd.concat([data, *[pd.DataFrame(dict(stage=[s]*3, score=[1., 2., 3.]))
                             for s in range(2, 6)]], ignore_index=True)
    summary = plot.stage_summary(data, ["score"])
    assert summary.n.eq(3).all()
    np.testing.assert_allclose(summary["sem"], np.full(6, 1/np.sqrt(3)))


def test_frozen_cohort_and_exports(tmp_path, monkeypatch):
    meta = pd.read_csv(plot.PH_ROOT / "tables/window_measurements.csv")
    feature = pd.read_csv(plot.FEATURE_ROOT / "tables/features.csv")
    plot.validate_cohort(meta, feature)
    assert len(plot.FEATURES) == 15
    monkeypatch.setattr(plot, "OUTPUT", tmp_path)
    plot.style()
    fig = plot.feature_panel(plot.stage_summary(feature[feature.usable], plot.FEATURES), "R1")
    assert plot.export_png(fig, tmp_path / "R1.png")["pixels"] == [2490, 1890]
