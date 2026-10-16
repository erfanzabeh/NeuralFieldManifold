import importlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

UNIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIT))


def module():
    return importlib.import_module("prego_tau_sweep")


def test_sweep_reuses_values_with_correct_lag_order_and_counts():
    sweep = module()
    x = np.arange(500, dtype=np.float32)
    before = x.copy()
    clouds = sweep.make_sweep(x)
    assert list(clouds) == list(range(1, 21))
    for tau, cloud in clouds.items():
        np.testing.assert_array_equal(cloud[:, 0], x[2*tau:])
        np.testing.assert_array_equal(cloud[:, 1], x[tau:-tau])
        np.testing.assert_array_equal(cloud[:, 2], x[:-2*tau])
        assert cloud.shape == (500-2*tau, 3)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("x", [np.ones(499), np.full(500, np.nan), np.ones((1, 500))])
def test_sweep_rejects_invalid_epochs(x):
    with pytest.raises(ValueError):
        module().make_sweep(x)


def test_raw_grid_shows_all_delays_on_one_fixed_scale():
    sweep = module()
    x = np.sin(2*np.pi*22*np.arange(500)/1000)
    clouds = sweep.make_sweep(x)
    trial = pd.Series(dict(direction=1, original_trial_number=16, row_index=1))
    results = {tau: {"status": "failed", "reason": "test failure"} for tau in clouds}
    limits = sweep.sweep_limits(clouds, results)
    fig = sweep.grid_figure(clouds, results, trial, limits, overlay=False)
    assert len(fig.axes) == 20
    for tau, ax in enumerate(fig.axes, 1):
        assert f"tau = {tau} ms" in ax.get_title()
        np.testing.assert_allclose(ax.get_xlim(), limits[0])
        np.testing.assert_allclose(ax.get_ylim(), limits[1])
        np.testing.assert_allclose(ax.get_zlim(), limits[2])
        assert (ax.elev, ax.azim) == (22, -58)
    plt.close(fig)


def test_detail_labels_actual_tau_and_preserves_fit_failure():
    sweep = module()
    clouds = sweep.make_sweep(np.sin(np.arange(500)/10))
    trial = pd.Series(dict(direction=2, original_trial_number=10, row_index=39))
    result = {"status": "failed", "reason": "synthetic nonfinite result"}
    limits = sweep.sweep_limits(clouds, {1: result})
    fig = sweep.detail_figure(clouds[7], result, trial, 7, limits)
    text = "\n".join(t.get_text() for t in fig.texts)
    assert "tau = 7 ms" in text
    assert "486 points" in text
    assert "synthetic nonfinite result" in text
    assert fig.axes[0].get_ylabel() == "x(t - 7 ms)"
    assert fig.axes[0].get_zlabel() == "x(t - 14 ms)"
    plt.close(fig)
