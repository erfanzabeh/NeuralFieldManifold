from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic_cloud():
    t = np.linspace(0, 2*np.pi, 120, endpoint=False)
    return np.column_stack([(2+.12*np.cos(7*t))*np.cos(t),
                            (1.5+.12*np.cos(7*t))*np.sin(t), .12*np.sin(7*t)])


@pytest.mark.parametrize("name", ["one", "two"])
def test_optional_diagnostics_preserve_native_fit(name):
    module = importlib.import_module(f"NeuralFieldManifold.fits.{name}_torus")
    fit = getattr(module, f"{name}_torus_fit")
    points = synthetic_cloud()
    original = fit(points)
    detailed = fit(points, return_diagnostics=True)
    assert "diagnostics" not in original
    for key, value in original.items():
        np.testing.assert_array_equal(value, detailed[key])
    diag = detailed["diagnostics"]
    assert isinstance(diag["success"], bool)
    assert diag["nfev"] > 0
    assert np.asarray(diag["params"]).shape == (8,)
    assert np.asarray(diag["distance_3d"]).shape == (len(points),)
    assert np.mean(diag["distance_3d"]**2) == pytest.approx(detailed["mse"])


@pytest.fixture
def core():
    return importlib.import_module("prego_geometric_fits")


def test_fixed_clouds_preserve_trial_boundaries_and_coordinates(core):
    t = np.arange(500)/1000
    segments = np.stack([np.sin(2*np.pi*22*t), np.cos(2*np.pi*17*t)])
    processed, clouds = core.make_clouds(segments)
    assert clouds.shape == (2, 468, 3)
    np.testing.assert_array_equal(clouds[:, :, 0], processed[:, 32:])
    np.testing.assert_array_equal(clouds[:, :, 1], processed[:, 16:484])
    np.testing.assert_array_equal(clouds[:, :, 2], processed[:, :468])
    assert not np.array_equal(clouds[0], clouds[1])
    with pytest.raises(ValueError):
        core.make_clouds(np.zeros((2, 499)))


def test_summary_normalization_uses_full_cloud_variance(core):
    from NeuralFieldManifold.fits.one_torus import one_torus_fit
    p = synthetic_cloud()
    fit = one_torus_fit(p, return_diagnostics=True)
    row = core.summarize_fit(p, fit, "one_torus")
    distance = fit["diagnostics"]["distance_3d"]
    expected = np.sum(distance**2)/np.sum((p-p.mean(axis=0))**2)
    assert row["normalized_3d_set_error"] == pytest.approx(expected)
    assert row["native_r_squared"] == fit["r_squared"]
    assert row["native_mse"] == fit["mse"]
    assert row["n_points"] == 120


def test_known_geometric_distances_include_plane_offset(core):
    fit = dict(center=np.zeros(3), direction=np.array([0., 0., 1.]),
               u_axis=np.array([1., 0., 0.]), v_axis=np.array([0., 1., 0.]),
               R1=2., R2=2., R1_in=1., R2_in=1., minor_radius=.25)
    p = np.array([[1.5, 0, .5], [2.5, 0, 0], [.5, 0, 0]])
    np.testing.assert_allclose(core.recompute_distances(p, fit, "one_torus"), [.5, .5, .5])
    p = np.array([[2, 0, .25], [2, 0, 0], [2.5, 0, 0]])
    np.testing.assert_allclose(core.recompute_distances(p, fit, "two_torus"), [0, 0, .25], atol=1e-12)


def test_degeneracy_and_nonconvergence_remain_flagged(core):
    from NeuralFieldManifold.fits.two_torus import two_torus_fit
    p = synthetic_cloud()
    fit = two_torus_fit(p, return_diagnostics=True)
    fit["minor_radius"] = 2*min(fit["R1"], fit["R2"])
    fit["diagnostics"]["success"] = False
    row = core.summarize_fit(p, fit, "two_torus")
    assert "collapsed_hole" in row["flags"]
    assert "nonconvergence" in row["flags"]
    assert not row["optimizer_success"]
    assert np.isfinite(row["normalized_3d_set_error"])


def test_invalid_fit_is_explicit_and_does_not_mutate_cloud(core):
    p = np.full((468, 3), np.nan)
    result = core.fit_one(p, "one_torus")
    assert result["status"] == "failed"
    assert result["reason"]
    assert np.isnan(p).all()


def test_examples_use_original_id_not_quality(core):
    import pandas as pd
    table = pd.DataFrame(dict(direction=[1, 1, 2, 2], original_trial_number=[90, 12, 11, 80],
                              row_index=[0, 1, 2, 3]))
    assert core.example_rows(table) == [1, 2]
