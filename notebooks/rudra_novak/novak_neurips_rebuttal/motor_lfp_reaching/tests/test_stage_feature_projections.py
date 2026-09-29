"""Check corrected-stage joins and training-only projection preprocessing."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_stage_feature_projections as projections


def test_frozen_inputs_and_feature_families():
    geometry, values, betti, columns = projections.load_frozen_data()
    sets = projections.feature_sets(values, betti, columns)
    expected = dict(geometry_all=15, radii=3, orientation=9, quality=3,
                    betti_raw=3, betti_rms=3, combined_raw=18, combined_rms=18)
    assert {name: matrix.shape[1] for name, (matrix, _) in sets.items()} == expected
    assert len(geometry) == 1188
    assert geometry.groupby("stage").size().tolist() == [198] * 6
    assert geometry.loc[geometry.stage_name.eq("post_TC"), "anchor_code"].eq(203).all()
    assert geometry.loc[geometry.stage_name.eq("post_SC"), "anchor_code"].eq(206).all()
    assert np.isnan(values).all(axis=1).sum() == 11
    np.testing.assert_array_equal(sets["combined_raw"][0][:, :15], values)
    np.testing.assert_array_equal(sets["combined_raw"][0][:, 15:], betti["raw"])


def test_heldout_values_cannot_change_fitted_preprocessing():
    values = np.array([[1., 5., np.nan], [3., 7., 2.], [5., 9., 4.],
                       [100., 300., np.nan], [200., 400., 10.]])
    train = np.array([True, True, True, False, False])
    scaled, imputer, scaler = projections.training_preprocess(values, train)
    modified = values.copy()
    modified[~train] = [-999., 999., 10000.]
    scaled_modified, imputer_modified, scaler_modified = projections.training_preprocess(modified, train)
    np.testing.assert_array_equal(imputer.statistics_, imputer_modified.statistics_)
    np.testing.assert_array_equal(scaler.mean_, scaler_modified.mean_)
    np.testing.assert_array_equal(scaled[train], scaled_modified[train])
    np.testing.assert_array_equal(imputer.statistics_, [3., 7., 3.])
    assert not np.array_equal(scaled[~train], scaled_modified[~train])


def test_saved_fold_zero_is_whole_trial_split():
    geometry, values, _, columns = projections.load_frozen_data()
    train = geometry.heldout_fold_zero_based.ne(0).to_numpy()
    assert int(train.sum()) == 948
    assert geometry.loc[~train, "trial_index"].nunique() == 40
    assert geometry.loc[train, "trial_index"].nunique() == 158
    scaled, _, _ = projections.training_preprocess(values, train)
    xy, details = projections.project(scaled, geometry.stage.to_numpy(), train, "pca")
    assert len(details["explained_variance_ratio"]) == 2
    table = projections.coordinate_table(geometry, xy)
    projections.validate_coordinates(table, geometry)
