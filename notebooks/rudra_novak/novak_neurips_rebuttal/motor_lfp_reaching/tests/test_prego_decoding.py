from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def core():
    assert importlib.util.find_spec("prego_decoding") is not None, "Pre-GO decoding implementation is missing"
    return importlib.import_module("prego_decoding")


def test_epoch_excludes_go_and_post_go_and_rejects_invalid_signal(core):
    x = np.tile(np.arange(1100, dtype=float), (3, 1))
    x[1, 650] = np.nan
    x[2] = 4
    segments, keep = core.extract_prego(x, fs=1000, go_sample=700)
    np.testing.assert_array_equal(keep, [0])
    np.testing.assert_array_equal(segments[0], np.arange(200, 700))
    x[:, 700:] = 1e12
    np.testing.assert_array_equal(core.extract_prego(x, 1000, 700)[0], segments)
    with pytest.raises(ValueError):
        core.extract_prego(x, 1000, 300)


def test_peak_features_recover_frequency_and_preserve_amplitude(core):
    t = np.arange(500) / 1000
    x = np.sin(2 * np.pi * 20 * t) + 0.6 * np.sin(2 * np.pi * 32 * t)
    power, frequency, bands = core.spectral_features(np.stack([x, 2*x]), "M")
    assert power.shape == frequency.shape == (2, 2)
    assert bands.shape == (2, 5)
    np.testing.assert_allclose(frequency, [[20, 32], [20, 32]], atol=0.5)
    np.testing.assert_allclose(power[1] - power[0], np.log10(4), atol=1e-8)
    np.testing.assert_allclose(bands[1] - bands[0], np.log10(4), atol=1e-8)
    assert core.spectral_features(np.stack([x]), "T")[0].shape == (1, 1)


def test_balancing_requires_six_directions_and_disjoint_fixed_folds(core):
    y = np.repeat(np.arange(1, 7), [15, 16, 17, 18, 19, 20])
    idx = core.balanced_indices(y, seed=42)
    np.testing.assert_array_equal(np.bincount(y[idx])[1:], np.full(6, 15))
    np.testing.assert_array_equal(idx, core.balanced_indices(y, seed=42))
    folds = core.fold_assignments(y[idx], seed=42)
    assert set(folds) == set(range(5))
    for fold in range(5):
        np.testing.assert_array_equal(np.bincount(y[idx][folds == fold])[1:], np.full(6, 3))
    with pytest.raises(ValueError):
        core.balanced_indices(y[y != 6], seed=42)
    with pytest.raises(ValueError):
        core.balanced_indices(y, seed=42, count=16)


def test_fusions_keep_expected_columns(core):
    p = np.arange(8).reshape(4, 2)
    f = p + 10
    bands = np.arange(20).reshape(4, 5)
    torus = np.arange(60).reshape(4, 15)
    sets = core.feature_sets(p, f, bands, torus)
    np.testing.assert_array_equal(sets["power_frequency"], np.hstack([p, f]))
    np.testing.assert_array_equal(sets["torus_power_frequency"], np.hstack([torus, p, f]))
    np.testing.assert_array_equal(sets["torus_all_bands"], np.hstack([torus, bands]))
    assert sets["torus_power_frequency"].shape[1] == 19


def test_estimator_imputes_and_scales_only_from_training(core):
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(1, 7), 10)
    train = rng.normal(size=(60, 6))
    train[0, 0] = np.nan
    model = core.make_estimator()
    model.fit(train, y)
    np.testing.assert_allclose(model.named_steps["impute"].statistics_, np.nanmedian(train, axis=0))
    before = model.named_steps["scale"].mean_.copy()
    model.predict(np.full((12, 6), 1e8))
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, before)
    reduced = core.make_estimator(pca_dim=2).fit(train, y)
    assert reduced.named_steps["pca"].components_.shape == (2, 6)


def test_day_parser_does_not_treat_channels_as_independent_days(core):
    assert core.recording_day("o080416002-18") == "2008-04-16"
    assert core.recording_day("o080416009-27") == "2008-04-16"
    assert core.recording_day("y070130005-13") == "2007-01-30"
    with pytest.raises(ValueError):
        core.recording_day("unknown")


def test_orientation_canonicalization_preserves_radii_and_orthonormality(core):
    fit = dict(R1=1., R2=2., minor_radius=.3, mse=.1, mean_error=.2,
               frac_inside=.6, direction=np.array([0.,0.,-1.]),
               u_axis=np.array([-1.,0.,0.]), v_axis=np.array([0.,1.,0.]))
    result = core.pack_torus(fit)
    np.testing.assert_allclose(result[:3], [2., 1., .3])
    axes = result[6:].reshape(3, 3)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(axes) > 0


def test_embedding_is_learned_without_heldout_data(core):
    t = np.arange(500) / 1000
    train = np.stack([np.sin(2*np.pi*20*t + k) + .2*np.sin(2*np.pi*33*t) for k in range(12)])
    model = core.learn_embedding(train, seed=42, bootstraps=10)
    assert model["dimension"] in (3, 5, 7, 9)
    assert 1 <= model["tau"] <= 100
    assert 500 - (model["dimension"]-1)*model["tau"] >= 300
    assert model["n_training_trials"] == 12
    first = core.geometry_coordinates(train[0], model)
    core.geometry_coordinates(np.arange(500, dtype=float)**2, model)
    np.testing.assert_array_equal(core.geometry_coordinates(train[0], model), first)
    assert first.shape[1] == 3


def test_within_fold_permutations_preserve_counts(core):
    y = np.repeat(np.arange(1, 7), 15)
    folds = core.fold_assignments(y, seed=42)
    shuffled = core.permute_within_folds(y, folds, seed=7)
    assert not np.array_equal(y, shuffled)
    for fold in range(5):
        np.testing.assert_array_equal(np.sort(y[folds == fold]), np.sort(shuffled[folds == fold]))


def test_supporting_delay_analysis_matches_counts_without_changing_primary(core):
    from run_prego_single_channel import selected_analyses
    short = np.repeat(np.arange(1, 7), [15, 16, 17, 18, 19, 20])
    long = np.repeat(np.arange(1, 7), [20, 19, 18, 17, 16, 15])
    labels = np.r_[short, long]
    data = dict(direction=labels, delay_label=np.array(["short"]*len(short)+["long"]*len(long)))
    selected, minimum = selected_analyses(data, np.arange(len(labels)), 42)
    assert minimum == {"short": 15, "long": 15}
    np.testing.assert_array_equal(selected["short"], selected["matched_short"])
    for indices in selected.values():
        np.testing.assert_array_equal(np.bincount(labels[indices])[1:], np.full(6, 15))


def test_heldout_predictions_cover_each_trial_for_all_representations(core):
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(1, 7), 15)
    folds = core.fold_assignments(labels)
    power, frequency = rng.normal(size=(90, 2)), rng.normal(size=(90, 2))
    bands, torus = rng.normal(size=(90, 5)), rng.normal(size=(5, 90, 15))
    predictions = core.heldout_predictions(power, frequency, bands, torus, labels, folds)
    assert set(predictions) == set(core.METHODS)
    for pred in predictions.values():
        assert pred.shape == labels.shape
        assert set(pred).issubset(set(range(1, 7)))
    torus[0, folds != 0] = np.nan
    with pytest.raises(ValueError, match="no usable training geometry"):
        core.heldout_predictions(power, frequency, bands, torus, labels, folds)
