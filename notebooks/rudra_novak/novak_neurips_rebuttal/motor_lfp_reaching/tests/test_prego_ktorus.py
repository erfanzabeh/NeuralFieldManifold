from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def core():
    return importlib.import_module("prego_ktorus")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_full_dimension_synthetic_fit(core, k):
    rng = np.random.default_rng(12)
    m = 2*k + 1
    times = np.arange(400)/1000
    freq = np.array([11.3, 23.7, 37.1])[:k]
    coefficients = rng.normal(size=(1+2*k, m))
    cloud = core.phase_design(times, freq) @ coefficients
    result = core.fit_cloud(cloud, times, freq + .15)
    assert result["success"]
    assert result["coefficients"].shape == (1+2*k, m)
    assert result["prediction"].shape == cloud.shape
    assert result["normalized_error"] < 1e-10
    np.testing.assert_allclose(result["frequencies"], freq, atol=1e-4)
    features = core.geometry_features(result, cloud)
    assert len(features) == k*m*(m+1)//2 + m + 1
    assert len(features) == len(core.feature_names(k, m))


def test_higher_coordinates_affect_objective(core):
    times = np.arange(400)/1000
    rng = np.random.default_rng(19)
    cloud = core.phase_design(times, [13., 31.]) @ rng.normal(size=(5, 5))
    original = core.fit_cloud(cloud, times, [13., 31.])
    changed = cloud.copy()
    changed[:, 4] += 2*np.sin(2*np.pi*47*times)
    perturbed = core.fit_cloud(changed, times, [13., 31.])
    assert original["normalized_error"] < 1e-12
    assert perturbed["normalized_error"] > .01
    assert perturbed["rmse_by_coordinate"][4] > .5


def test_mode_shape_is_phase_origin_invariant(core):
    rng = np.random.default_rng(34)
    coeff = rng.normal(size=(5, 5))
    rotated = coeff.copy()
    for k, phase in enumerate([.8, 2.1]):
        rotation = np.array([[np.cos(phase), -np.sin(phase)],
                             [np.sin(phase), np.cos(phase)]])
        rotated[1+2*k:3+2*k] = rotation @ coeff[1+2*k:3+2*k]
    np.testing.assert_allclose(core.mode_shapes(coeff), core.mode_shapes(rotated), atol=1e-12)


def test_zero_modes_and_bad_embedding_are_explicit(core):
    result = core.choose_embedding([], np.arange(1, 101), np.zeros(100), 500)
    assert result["status"] == "no_supported_peaks"
    assert result["K"] == 0
    lags = np.arange(1, 101)
    ami = (lags-90.)**2
    result = core.choose_embedding([12., 24., 36.], lags, ami, 500)
    assert result["status"] == "unresolved_embedding"
    assert result["dimension"] == 7
    assert result["tau"] is None


def test_embedding_has_no_silent_default_or_dimension_reduction(core):
    lags = np.arange(1, 101)
    result = core.choose_embedding([11., 29.], lags, (lags-17.)**2, 500)
    assert result["status"] == "ok"
    assert result["K"] == 2 and result["dimension"] == 5
    assert result["tau"] == 17
    assert result["condition"] <= 100
    assert core.choose_embedding([11.], lags, -lags.astype(float), 500)["status"] == "unresolved_embedding"
    assert core.delay_map([11., 29.], 5, 17).shape == (5, 4)
    assert core.harmonic_flags([10., 20., 37.])


def test_cloud_retains_exact_delay_coordinates(core):
    from motor_lfp_utils import lag_embed
    from select_trace_embedding_parameters import preprocess_segments
    times = np.arange(500)/1000
    signal = np.sin(2*np.pi*13*times) + .4*np.cos(2*np.pi*29*times)
    embedding = dict(dimension=5, tau=17, K=2, frequencies=[13., 29.], status="ok")
    cloud, times = core.trial_cloud(signal, embedding)
    np.testing.assert_array_equal(cloud, lag_embed(preprocess_segments(signal[None])[0], 5, 17))
    assert cloud.shape == (432, 5)
    assert times.shape == (432,)


def test_selection_is_label_blind_and_training_only(core):
    t = np.arange(500)/1000
    train = np.stack([np.sin(2*np.pi*17*t + i) + .7*np.cos(2*np.pi*37*t+i/3) for i in range(18)])
    first = core.learn_geometry(train, seed=9, bootstraps=8)
    second = core.learn_geometry(train.copy(), seed=9, bootstraps=8)
    assert first == second
    assert first["n_training_trials"] == 18
    assert "center" not in first and "components" not in first
    assert first["dimension"] == 2*first["K"]+1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_psd_recovers_resolved_synthetic_mode_count(core, k):
    t = np.arange(500)/1000
    rng = np.random.default_rng(9)
    freqs = [10, 26, 42][:k]
    train = np.array([sum(np.sin(2*np.pi*f*t+rng.uniform(0, 6)) for f in freqs)
                      + .02*rng.normal(size=500) for _ in range(30)])
    selected = core.learn_geometry(train, bootstraps=20)
    assert selected["K"] == k
    assert selected["status"] == "ok"
    np.testing.assert_allclose(selected["frequencies"], freqs, atol=2.)


def test_no_pca_in_classifier_and_variable_fold_width(core, monkeypatch):
    from sklearn.decomposition import PCA
    from prego_decoding import fold_assignments
    monkeypatch.setattr(PCA, "fit", lambda *a, **k: pytest.fail("PCA was called"))
    monkeypatch.setattr(PCA, "fit_transform", lambda *a, **k: pytest.fail("PCA was called"))
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(1, 7), 15)
    folds = fold_assignments(y)
    power = rng.normal(size=(90, 2))
    freq = rng.normal(size=(90, 2))
    bands = rng.normal(size=(90, 5))
    geometry = [rng.normal(size=(90, width)) for width in [10, 36, 10, 36, 10]]
    predictions = core.decode(power, freq, bands, geometry, y, folds)
    assert set(predictions) == set(core.METHODS)
    for prediction in predictions.values():
        assert prediction.shape == (90,)
        assert set(prediction) <= set(range(1, 7))
    assert "pca" not in core.estimator().named_steps


def test_failed_fit_returns_missing_not_zero(core):
    with pytest.raises(ValueError):
        core.fit_cloud(np.ones((20, 5)), np.arange(20)/1000, [])
    embedding = dict(dimension=5, tau=17, K=2, frequencies=[13., 29.], status="ok")
    feature, info = core.fit_trial(np.full(500, np.nan), embedding)
    assert not info["success"]
    assert np.isnan(feature).all()
