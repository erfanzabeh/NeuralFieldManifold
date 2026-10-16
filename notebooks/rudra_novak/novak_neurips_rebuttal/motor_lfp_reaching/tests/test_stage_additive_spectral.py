"""Guard the corrected-stage spectral comparator against cohort and metric drift."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage_additive_spectral as experiment


def test_spectral_bands_track_oscillation_and_amplitude():
    t = np.arange(300) / 1000
    raw = np.stack((np.sin(2 * np.pi * 20 * t),
                    2 * np.sin(2 * np.pi * 20 * t),
                    np.sin(2 * np.pi * 40 * t)))
    bands, average = experiment.spectral_features(raw)
    assert bands.shape == (3, 5) and average.shape == (3, 1)
    beta = list(experiment.BANDS).index("beta")
    gamma = list(experiment.BANDS).index("low_gamma")
    assert bands[0, beta] > bands[0, gamma]
    assert bands[2, gamma] > bands[2, beta]
    np.testing.assert_allclose(bands[1, beta] - bands[0, beta], np.log10(4), atol=1e-8)
    np.testing.assert_allclose(average[1, 0] - average[0, 0], np.log10(4), atol=1e-8)


def test_saved_inputs_match_corrected_stage_and_frozen_geometry():
    raw, geometry, labels, folds, trial_ids, old_pred, metadata = experiment.load_inputs()
    assert raw.shape == (1188, 300)
    assert geometry.shape == (1188, 15)
    assert metadata.loc[metadata.stage_name.eq("post_TC"), "anchor_code"].eq(203).all()
    assert metadata.loc[metadata.stage_name.eq("post_SC"), "anchor_code"].eq(206).all()
    bands, average = experiment.spectral_features(raw)
    sets = experiment.feature_sets(geometry, bands, average)
    assert {name: arr.shape[1] for name, arr in sets.items()} == {
        "relevant_band": 1, "geometry_relevant_band": 16,
        "average_psd": 1, "geometry_average_psd": 16,
        "geometry": 15, "all_bands": 5, "geometry_all_bands": 20}
    np.testing.assert_array_equal(experiment.decode(geometry, labels, folds), old_pred)
    assert trial_ids.size == 1188
