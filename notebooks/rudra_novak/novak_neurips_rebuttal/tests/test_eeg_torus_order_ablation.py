from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_eeg_torus_order_ablation as ablation


def test_no_r_geometry_keeps_radii_and_orientation_only() -> None:
    all_torus = np.arange(30, dtype=float).reshape(2, 15)

    reduced = ablation.select_feature_matrix(ablation.NO_R_FEATURE, all_torus)

    expected = np.hstack([all_torus[:, [0, 1]], all_torus[:, 6:15]])
    assert reduced.shape == (2, 11)
    np.testing.assert_array_equal(reduced, expected)
    assert ablation.FEATURE_SPECS[ablation.NO_R_FEATURE].feature_names == [
        "R1",
        "R2",
        "direction_x",
        "direction_y",
        "direction_z",
        "u_axis_x",
        "u_axis_y",
        "u_axis_z",
        "v_axis_x",
        "v_axis_y",
        "v_axis_z",
    ]


def test_full_torus_feature_set_keeps_all_15_columns() -> None:
    all_torus = np.arange(30, dtype=float).reshape(2, 15)

    full = ablation.select_feature_matrix(ablation.FULL_TORUS_FEATURE, all_torus)

    assert full.shape == (2, 15)
    np.testing.assert_array_equal(full, all_torus)
    assert "minor_radius" in ablation.FEATURE_SPECS[ablation.FULL_TORUS_FEATURE].feature_names
    assert "minor_radius" not in ablation.FEATURE_SPECS[ablation.NO_R_FEATURE].feature_names


def test_reporting_table_is_f1_only() -> None:
    summary = np.array(
        [
            (ablation.FULL_TORUS_FEATURE, "Torus features (15D)", 15, 0.67, 0.06, 0.68, 0.07, 21),
            (ablation.NO_R_FEATURE, "No-r geometry (11D)", 11, 0.65, 0.07, 0.66, 0.08, 21),
        ],
        dtype=[
            ("feature_set", "O"),
            ("feature_label", "O"),
            ("n_features", "i8"),
            ("mean_f1", "f8"),
            ("std_f1", "f8"),
            ("mean_accuracy", "f8"),
            ("std_accuracy", "f8"),
            ("n_sessions", "i8"),
        ],
    )
    significance = np.array(
        [("torus_features_15d vs no_r_geometry_11d", 0.082195, "ns")],
        dtype=[("comparison", "O"), ("p_value", "f8"), ("significance", "O")],
    )

    table = ablation.make_reporting_table(
        ablation.pd.DataFrame(summary),
        ablation.pd.DataFrame(significance),
    )

    assert table.columns.tolist() == [
        "scenario",
        "f1_mean_sd",
        "paired_test",
    ]
    assert "accuracy" not in " ".join(table.columns)
    assert "feature" not in " ".join(table.columns)
    assert "session" not in " ".join(table.columns)
    assert table["f1_mean_sd"].tolist() == ["0.670 +/- 0.060 SD", "0.650 +/- 0.070 SD"]
    assert table["paired_test"].tolist() == ["paired Wilcoxon p=0.0822 (ns)", ""]
