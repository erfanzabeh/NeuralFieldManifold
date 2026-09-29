"""Check AMI selection, window boundaries, and training-only scope selection."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_ami_analysis import LAGS, average_mutual_information, first_minimum, training_segments


def ami(segments, bins=16):
    return average_mutual_information(segments, LAGS, bins, 1000000, 42)


def test_first_minimum_and_no_fallback():
    minimum, _ = first_minimum((LAGS - 12.) ** 2)
    assert minimum == 12
    assert first_minimum(np.ones(100))[0] is None
    assert first_minimum(-LAGS.astype(float))[0] is None


def test_no_trial_boundary_pairs():
    segments = np.repeat(np.array([[0.] * 300, [10.] * 300]), 30, axis=0)
    np.testing.assert_allclose(ami(segments, bins=2), np.log(2), atol=1e-12)


def test_symmetry_and_affine_invariance():
    segments = np.random.default_rng(42).normal(size=(40, 300))
    reference = ami(segments)
    np.testing.assert_allclose(ami(segments[:, ::-1]), reference, atol=1e-12)
    np.testing.assert_allclose(ami(segments * 7. + 3.), reference, atol=1e-12)


def test_known_oscillation_delay():
    rng = np.random.default_rng(1)
    phase = rng.uniform(0, 2*np.pi, size=(198, 1))
    signal = np.sin(2*np.pi*25*np.arange(300)[None, :]/1000 + phase)
    signal += rng.normal(0, .02, size=signal.shape)
    selected, _ = first_minimum(ami(signal, bins=48))
    assert 8 <= selected <= 12


def test_heldout_changes_cannot_change_training_selection():
    signal = np.random.default_rng(2).normal(size=(24, 300))
    metadata = pd.DataFrame({"heldout_fold_zero_based": np.repeat(np.arange(4), 6)})
    before = training_segments(signal, metadata, 0)
    altered = signal.copy()
    altered[:6] = 1000
    np.testing.assert_array_equal(before, training_segments(altered, metadata, 0))
    np.testing.assert_array_equal(ami(before), ami(training_segments(altered, metadata, 0)))
