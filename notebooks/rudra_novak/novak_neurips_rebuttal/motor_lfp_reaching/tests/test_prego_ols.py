import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def core():
    return importlib.import_module("prego_ols")


def test_trial_local_lags_and_targets(core):
    x = np.vstack([np.arange(500), 10000 + np.arange(500)])
    design, targets = core.design(x, [0, 5, 10], [1, 10])
    np.testing.assert_array_equal(design[0, 0], [200, 195, 190])
    np.testing.assert_array_equal(design[1, -1], [10489, 10484, 10479])
    np.testing.assert_array_equal(targets[0, -1], [490, 499])
    assert design.shape == (2, 290, 3)
    with pytest.raises(ValueError):
        core.design(x, [0, 201], [10])


def test_ols_coefficients_in_original_units(core):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(7, 20, 3)) * [1, 10, .01] + [4, -3, .6]
    weights = np.array([[2, -1], [.3, .2], [-.1, .5], [4, -2]])
    y = weights[0] + x @ weights[1:]
    model = core.fit_ols(x, y)
    assert model["status"] == "ok"
    np.testing.assert_allclose(model["coefficients"], weights, atol=1e-10)
    np.testing.assert_allclose(core.predict(model, x), y, atol=1e-10)


def test_singular_fit_is_flagged_not_regularized(core):
    x = np.ones((4, 10, 2))
    model = core.fit_ols(x, np.ones((4, 10, 1)))
    assert model["status"] == "rank_deficient"
    assert model["rank"] < 3


def test_recursive_forecast_matches_independent_loop(core):
    rng = np.random.default_rng(3)
    history = rng.normal(size=(5, 6, 3))
    coefficients = np.array([.3, .5, -.2, .1])
    actual = core.recursive_prediction(coefficients, history, 10)
    expected = np.empty((5, 6))
    for i in range(5):
        for j in range(6):
            values = history[i, j].tolist()
            for _ in range(10):
                new = coefficients[0] + np.dot(coefficients[1:], values)
                values = [new] + values[:-1]
            expected[i, j] = new
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_metrics_and_persistence(core):
    truth = np.array([[0., 1., 2.], [0., 2., 4.]])
    predicted = np.zeros_like(truth)
    baseline = truth + 1
    scores = core.trial_metrics(truth, predicted, baseline)
    np.testing.assert_allclose(scores["nmse"], [2.5, 2.5])
    np.testing.assert_allclose(scores["r2"], [-1.5, -1.5])
    np.testing.assert_allclose(scores["nmse_persistence"], [1.5, .375])
    assert np.isnan(scores["pearson"]).all()
    assert np.isnan(core.trial_metrics(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)))["nmse"]).all()


def test_one_se_prefers_simpler_orders(core):
    rows = pd.DataFrame([
        dict(p=1, m=0, tau=1, mean_nmse=.15, se_nmse=.01, valid=True),
        dict(p=2, m=0, tau=1, mean_nmse=.11, se_nmse=.02, valid=True),
        dict(p=3, m=0, tau=1, mean_nmse=.10, se_nmse=.02, valid=True),
        dict(p=4, m=0, tau=1, mean_nmse=.01, se_nmse=.001, valid=False),
    ])
    chosen = core.select_candidate(rows, "ar")
    assert chosen["p"] == 2
    assert chosen["one_se_threshold"] == pytest.approx(.12)


def test_one_se_embedding_dimension_then_delay_error(core):
    rows = pd.DataFrame([
        dict(p=0, m=2, tau=1, mean_nmse=.115, se_nmse=.01, valid=True),
        dict(p=0, m=2, tau=10, mean_nmse=.11, se_nmse=.01, valid=True),
        dict(p=0, m=3, tau=1, mean_nmse=.10, se_nmse=.02, valid=True),
    ])
    chosen = core.select_candidate(rows, "embedding")
    assert (chosen["m"], chosen["tau"]) == (2, 10)


def test_ar_poles_follow_recurrence_convention(core):
    radius, frequency = .98, 20.
    angle = 2 * np.pi * frequency / 1000
    poles = core.ar_poles([0., 2 * radius * np.cos(angle), -radius**2])
    np.testing.assert_allclose(poles["magnitude"], radius, atol=1e-12)
    np.testing.assert_allclose(np.abs(poles["frequency_hz"]), frequency, atol=1e-10)


def test_embedding_grid_never_exceeds_epoch_support(core):
    config = core.default_config()
    candidates = core.candidates(config, "embedding")
    assert any(c["m"] == 6 for c in candidates)
    assert all((c["m"] - 1) * c["tau"] <= 200 for c in candidates)
    assert not any(c["m"] == 9 and c["tau"] == 100 for c in candidates)


def test_outer_test_values_never_change_inner_selection(core):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(60, 500))
    labels = np.tile(np.arange(1, 7), 10)
    folds = np.repeat(np.arange(5), 12)
    config = core.default_config()
    config.update(orders=[1, 2], dimensions=[2, 3], delays=[1, 5])
    original = core.audit_fold(x, labels, folds, 0, config, seed=8)
    changed = x.copy()
    changed[folds == 0] = rng.normal(1000, 100, changed[folds == 0].shape)
    second = core.audit_fold(changed, labels, folds, 0, config, seed=8)
    pd.testing.assert_frame_equal(original["validation"], second["validation"])
    assert original["selection"] == second["selection"]
    for split in original["inner_splits"]:
        assert not np.intersect1d(split["train"], np.flatnonzero(folds == 0)).size
        assert not np.intersect1d(split["validation"], np.flatnonzero(folds == 0)).size


def test_ar_predictions_do_not_refresh_from_future_truth(core):
    coefficients = np.array([.1, .8, -.2])
    x = np.arange(500.)[None, :]
    design, _ = core.design(x, [0, 1], [1, 10])
    first = core.recursive_prediction(coefficients, design[:, :1], 10)
    x[:, 201:] = -999
    design, _ = core.design(x, [0, 1], [1, 10])
    second = core.recursive_prediction(coefficients, design[:, :1], 10)
    np.testing.assert_array_equal(first, second)
