from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_eeg_reviewer_comparisons as eeg_plots


def test_relevant_band_comparison_uses_delta_not_best_band() -> None:
    scores = pd.DataFrame(
        [
            {"session_id": "s1", "recording_hour": "Hour 1", "feature_set": "all_torus_15", "f1": 0.70},
            {"session_id": "s1", "recording_hour": "Hour 1", "feature_set": "average_psd", "f1": 0.40},
            {"session_id": "s1", "recording_hour": "Hour 1", "feature_set": "delta", "f1": 0.50},
            {"session_id": "s1", "recording_hour": "Hour 1", "feature_set": "low_gamma", "f1": 0.99},
            {"session_id": "s2", "recording_hour": "Hour 2", "feature_set": "all_torus_15", "f1": 0.80},
            {"session_id": "s2", "recording_hour": "Hour 2", "feature_set": "average_psd", "f1": 0.45},
            {"session_id": "s2", "recording_hour": "Hour 2", "feature_set": "delta", "f1": 0.55},
            {"session_id": "s2", "recording_hour": "Hour 2", "feature_set": "low_gamma", "f1": 0.98},
        ]
    )

    comparison, selection, _significance = eeg_plots.relevant_band_comparison(scores)

    relevant = comparison[comparison["feature_role"] == "relevant_band"]
    assert set(relevant["feature_set"]) == {"delta"}
    assert set(relevant["relevant_band_label"]) == {"Delta (0.5-4 Hz)"}
    assert selection.iloc[0]["relevant_band"] == "delta"
