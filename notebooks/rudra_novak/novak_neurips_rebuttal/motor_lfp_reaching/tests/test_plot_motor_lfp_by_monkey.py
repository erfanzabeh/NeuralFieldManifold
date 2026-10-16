from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_motor_lfp_by_monkey as plots


def test_summarize_relevant_band_bars_uses_matched_lfp_values() -> None:
    comparison = pd.DataFrame(
        [
            {"analysis_label": "Monkey M", "monkey": "M", "lfp_uid": "m1", "feature_role": "torus_features", "f1": 0.20},
            {"analysis_label": "Monkey M", "monkey": "M", "lfp_uid": "m1", "feature_role": "average_psd", "f1": 0.10},
            {"analysis_label": "Monkey M", "monkey": "M", "lfp_uid": "m1", "feature_role": "relevant_band", "f1": 0.15},
            {"analysis_label": "Monkey M", "monkey": "M", "lfp_uid": "m2", "feature_role": "torus_features", "f1": 0.30},
            {"analysis_label": "Monkey M", "monkey": "M", "lfp_uid": "m2", "feature_role": "average_psd", "f1": 0.20},
            {"analysis_label": "Monkey M", "monkey": "M", "lfp_uid": "m2", "feature_role": "relevant_band", "f1": 0.25},
        ]
    )

    summary = plots.summarize_relevant_band_bars(comparison)

    torus = summary[summary["feature_role"] == "torus_features"].iloc[0]
    assert torus["mean_f1"] == 0.25
    assert round(torus["std_f1"], 6) == 0.070711
    assert torus["n_lfps"] == 2


def test_build_overall_relevant_band_comparison_uses_beta_not_best_band() -> None:
    scores = pd.DataFrame(
        [
            {"lfp_uid": "m1", "monkey": "M", "session_id": "s1", "lfp_id": 1, "feature_set": "torus_nonlinear_15", "f1": 0.21},
            {"lfp_uid": "m1", "monkey": "M", "session_id": "s1", "lfp_id": 1, "feature_set": "average_psd", "f1": 0.11},
            {"lfp_uid": "m1", "monkey": "M", "session_id": "s1", "lfp_id": 1, "feature_set": "beta", "f1": 0.16},
            {"lfp_uid": "m1", "monkey": "M", "session_id": "s1", "lfp_id": 1, "feature_set": "delta", "f1": 0.99},
            {"lfp_uid": "t1", "monkey": "T", "session_id": "s2", "lfp_id": 2, "feature_set": "torus_nonlinear_15", "f1": 0.22},
            {"lfp_uid": "t1", "monkey": "T", "session_id": "s2", "lfp_id": 2, "feature_set": "average_psd", "f1": 0.12},
            {"lfp_uid": "t1", "monkey": "T", "session_id": "s2", "lfp_id": 2, "feature_set": "beta", "f1": 0.17},
            {"lfp_uid": "t1", "monkey": "T", "session_id": "s2", "lfp_id": 2, "feature_set": "delta", "f1": 0.98},
        ]
    )

    comparison, _significance = plots.build_overall_relevant_band_comparison(scores)

    relevant = comparison[comparison["feature_role"] == "relevant_band"]
    assert set(relevant["feature_set"]) == {"beta"}
    assert set(relevant["relevant_band_label"]) == {"Beta (13-30 Hz)"}
    assert set(comparison["analysis_label"]) == {"Full 6-direction LFPs"}
