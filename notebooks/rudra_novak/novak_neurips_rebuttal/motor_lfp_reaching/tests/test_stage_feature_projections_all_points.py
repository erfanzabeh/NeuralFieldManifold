"""Check full-cohort descriptive projections retain all stage windows."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_stage_feature_projections_all_points as projections


def test_all_point_preprocessing_uses_every_row():
    values = np.array([[1., np.nan], [3., 8.], [5., 10.]])
    scaled, imputer, scaler = projections.preprocess(values)
    np.testing.assert_allclose(imputer.statistics_, [3., 9.])
    np.testing.assert_allclose(scaled.mean(axis=0), [0., 0.], atol=1e-12)
    assert scaler.n_samples_seen_ == 3


def test_projection_coordinates_contain_all_windows_without_split_labels():
    geometry, values, _, _ = projections.frozen.load_frozen_data()
    scaled, _, _ = projections.preprocess(values)
    xy, details = projections.project(scaled, geometry.stage.to_numpy(), "pca")
    assert details["uses_stage_labels"] is False
    table = projections.coordinate_table(geometry, xy)
    projections.validate_coordinates(table, geometry)
    assert "split" not in table.columns
    assert table.groupby("stage").size().tolist() == [198] * 6


def test_lda_is_marked_supervised():
    values = np.array([[-2., 0., 1.], [-1., .1, 0.], [-1.5, -.1, .5],
                       [1., .3, .1], [2., .4, .5], [1.5, .2, 1.],
                       [0., 2., 1.], [.3, 3., .6], [-.3, 2.5, .3]])
    stages = np.repeat(np.arange(3), 3)
    scaled, _, _ = projections.preprocess(values)
    xy, details = projections.project(scaled, stages, "lda")
    assert xy.shape == (9, 2)
    assert details["uses_stage_labels"] is True
