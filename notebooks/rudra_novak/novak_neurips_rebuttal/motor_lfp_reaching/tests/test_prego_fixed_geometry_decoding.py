import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def core():
    return importlib.import_module("prego_fixed_geometry_decoding")


def example_result():
    return dict(status="returned", summary=dict(flags="", optimizer_success=True), fit=dict(
        R1=3., R2=2., R1_in=1.5, R2_in=1., minor_radius=.625,
        mse=.1, mean_error=.2, frac_inside=.8,
        direction=np.array([0., 0., -1.]), u_axis=np.array([-1., 0., 0.]),
        v_axis=np.array([0., 1., 0.])))


def synthetic_inputs():
    rng = np.random.default_rng(9)
    y = np.repeat(np.arange(1, 7), 10)
    folds = np.tile(np.arange(5), 12)
    x = rng.normal(size=(60, 14)) + y[:, None]/5
    x[0] = np.nan
    return dict(geometry=x), y, folds


def test_exact_fourteen_columns_exclude_width_and_center(core):
    result = example_result()
    features, reason = core.geometry_features(result)
    assert reason == "" and features.shape == (14,)
    np.testing.assert_allclose(features[:5], [3, 2, .1, .2, .8])
    np.testing.assert_allclose(features[5:], [0, 0, 1, 1, 0, 0, 0, 1, 0])
    result["fit"]["minor_radius"] = 900
    result["fit"]["center"] = np.array([100, 200, 300])
    np.testing.assert_array_equal(core.geometry_features(result)[0], features)
    assert len(core.GEOMETRY_COLUMNS) == 14
    assert not any("width" in n or "minor" in n or "center" in n for n in core.GEOMETRY_COLUMNS)


def test_equivalent_axis_signs_and_radius_order_give_same_features(core):
    result = example_result()
    before = core.geometry_features(result)[0]
    changed = copy.deepcopy(result)
    changed["fit"]["direction"] *= -1
    changed["fit"]["u_axis"] *= -1
    np.testing.assert_array_equal(core.geometry_features(changed)[0], before)
    changed["fit"]["R1"], changed["fit"]["R2"] = 2., 3.
    changed["fit"]["R1_in"], changed["fit"]["R2_in"] = 1., 1.5
    changed["fit"]["u_axis"], changed["fit"]["v_axis"] = changed["fit"]["v_axis"], changed["fit"]["u_axis"]
    np.testing.assert_array_equal(core.geometry_features(changed)[0], before)


@pytest.mark.parametrize("kind", ["failed", "nonconverged", "nonfinite", "invalid_radius", "zero_axis"])
def test_unusable_fit_returns_missing_row_not_dropped_trial(core, kind):
    result = example_result()
    if kind == "failed":
        result = dict(status="failed", reason="test failure")
    elif kind == "nonconverged":
        result["summary"]["optimizer_success"] = False
    elif kind == "nonfinite":
        result["fit"]["mse"] = np.nan
    elif kind == "invalid_radius":
        result["fit"]["R1_in"] = 4.
    else:
        result["fit"]["direction"][:] = 0
    values, reason = core.geometry_features(result)
    assert np.isnan(values).all() and values.shape == (14,) and reason


def test_boundary_and_circular_flags_are_retained(core):
    result = example_result()
    result["summary"]["flags"] = "active_optimizer_bound;nearly_circular_orientation"
    assert np.isfinite(core.geometry_features(result)[0]).all()


def test_heldout_predictions_cover_every_trial_without_preprocessing_leakage(core):
    features, y, folds = synthetic_inputs()
    pred, models, audit = core.decode(features, y, folds, retain_models=True)
    assert pred.shape == (60, 1) and set(pred[:, 0]) <= set(range(1, 7))
    assert len(audit) == 5
    model = models[("geometry", 0)]
    assert list(model.named_steps) == ["impute", "scale", "lda"]
    assert model.named_steps["lda"].solver == "lsqr"
    assert model.named_steps["lda"].shrinkage == "auto"
    np.testing.assert_allclose(model.named_steps["impute"].statistics_,
                               np.nanmedian(features["geometry"][folds != 0], axis=0))
    changed = {"geometry": features["geometry"].copy()}
    changed["geometry"][folds == 0] = 1e9
    _, modified, _ = core.decode(changed, y, folds, retain_models=True)
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, modified[("geometry", 0)].named_steps["scale"].mean_)
    np.testing.assert_array_equal(model.named_steps["lda"].coef_, modified[("geometry", 0)].named_steps["lda"].coef_)


def test_all_missing_training_column_stops_without_dropping_features(core):
    features, y, folds = synthetic_inputs()
    features["geometry"][folds != 0, 2] = np.nan
    with pytest.raises(ValueError, match="training"):
        core.decode(features, y, folds)


def test_scores_match_sklearn_including_zero_recall_class(core):
    y = np.repeat(np.arange(1, 7), 8)
    pred = np.random.default_rng(4).integers(1, 6, size=(len(y), 2))
    scores = core.score_predictions(y, pred)
    for i in range(2):
        assert scores["macro_f1"][i] == pytest.approx(f1_score(y, pred[:, i], average="macro", labels=np.arange(1, 7), zero_division=0))
        assert scores["accuracy"][i] == pytest.approx(accuracy_score(y, pred[:, i]))
        np.testing.assert_allclose(scores["per_direction_f1"][i], f1_score(y, pred[:, i], average=None, labels=np.arange(1, 7), zero_division=0))
        np.testing.assert_allclose(scores["confusion_fraction"][i], confusion_matrix(y, pred[:, i], labels=np.arange(1, 7), normalize="true"))


def test_bootstrap_preserves_class_counts_and_reproduces_from_indices(core):
    y = np.repeat(np.arange(1, 7), 8)
    pred = np.random.default_rng(4).integers(1, 7, size=(len(y), 2))
    result = core.bootstrap_predictions(y, pred, n_bootstrap=12, seed=7)
    assert result["indices"].shape == (12, len(y))
    for b, idx in enumerate(result["indices"]):
        np.testing.assert_array_equal(np.bincount(y[idx])[1:], np.full(6, 8))
        scores = core.score_predictions(y[idx], pred[idx])
        np.testing.assert_array_equal(scores["macro_f1"], result["macro_f1"][b])


def test_permutation_refits_lda_and_scores_against_permuted_labels(core):
    from prego_decoding import permute_within_folds
    features, y, folds = synthetic_inputs()
    result = core.permutation_result(3, features, y, folds, seed=81)
    shuffled = permute_within_folds(y, folds, seed=84)
    np.testing.assert_array_equal(result["labels"], shuffled)
    expected, _, _ = core.decode(features, shuffled, folds)
    np.testing.assert_array_equal(result["predictions"], expected)
    np.testing.assert_array_equal(result["macro_f1"], core.score_predictions(shuffled, expected)["macro_f1"])
    for fold in range(5):
        np.testing.assert_array_equal(np.sort(y[folds == fold]), np.sort(shuffled[folds == fold]))
