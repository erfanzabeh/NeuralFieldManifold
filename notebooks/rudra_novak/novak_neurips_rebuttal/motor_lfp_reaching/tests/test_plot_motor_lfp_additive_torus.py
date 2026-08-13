from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_motor_lfp_additive_torus as additive


def test_additive_feature_matrices_concatenate_expected_columns() -> None:
    band_power = np.arange(10, dtype=float).reshape(2, 5)
    average_psd = np.array([[100.0], [200.0]])
    torus = np.arange(30, dtype=float).reshape(2, 15)

    torus_relevant = additive.feature_matrix("torus_plus_relevant_band", band_power, average_psd, torus)
    torus_average = additive.feature_matrix("torus_plus_average_psd", band_power, average_psd, torus)
    torus_all = additive.feature_matrix("torus_plus_all_band_power", band_power, average_psd, torus)

    assert torus_relevant.shape == (2, 16)
    assert torus_average.shape == (2, 16)
    assert torus_all.shape == (2, 20)
    np.testing.assert_array_equal(torus_relevant[:, :15], torus)
    np.testing.assert_array_equal(torus_relevant[:, 15], band_power[:, 3])
    np.testing.assert_array_equal(torus_average[:, 15], average_psd[:, 0])
    np.testing.assert_array_equal(torus_all[:, 15:], band_power)


def test_additive_plot_labels_use_relevant_band_not_relevant_beta() -> None:
    labels = [additive.FEATURE_LABELS[key] for key in additive.FEATURE_ORDER]

    assert "Relevant band\n(13-30 Hz)" in labels
    assert "Torus +\nRelevant band" in labels
    assert all("Relevant beta" not in label for label in labels)
