import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def stats():
    assert importlib.util.find_spec("prego_statistics") is not None
    return importlib.import_module("prego_statistics")


def test_metrics_are_heldout_macro_f1_and_row_recall(stats):
    y = np.repeat(np.arange(1, 7), 5)
    pred = y.copy()
    pred[:3] = 2
    row, classes, matrix = stats.prediction_metrics(y, pred)
    assert row == pytest.approx(f1_score(y, pred, average="macro"))
    assert row == pytest.approx(classes.mean())
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    assert matrix[0, 0] == pytest.approx(.4)


def test_holm_is_across_the_full_family(stats):
    p = np.array([.001, .02, .3, .8, .003, .06, 1, .4])
    expected = [.008, .12, 1, 1, .021, .3, 1, 1]
    np.testing.assert_allclose(stats.holm(p), expected)


def test_cluster_interval_resamples_days_not_individual_channels(stats):
    differences = pd.DataFrame(dict(day=["a"]*10 + ["b"], difference=[.1]*10 + [-.2]))
    mean, low, high = stats.cluster_interval(differences, seed=42, draws=2000)
    assert mean == pytest.approx(.8/11)
    assert low == pytest.approx(-.2)
    assert high == pytest.approx(.1)


def test_example_selection_does_not_read_decoder_scores(stats):
    frame = pd.DataFrame(dict(recording=["b", "a", "c", "d"], monkey=["M"]*4,
                              fit_error=[2., 2., 1., 3.], macro_f1=[0., 1., .5, .4]))
    assert stats.select_example(frame) == "a"
    frame["macro_f1"] = [1., 0., 0., 1.]
    assert stats.select_example(frame) == "a"
