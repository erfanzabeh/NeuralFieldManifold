import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score, confusion_matrix
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage_geometry_decoding as core


def result():
    return dict(status="returned", summary=dict(flags="", optimizer_success=True), fit=dict(
        R1=3., R2=2., R1_in=1.5, R2_in=1., minor_radius=.625,
        mse=.1, mean_error=.2, frac_inside=.8, direction=np.array([0., 0., -1.]),
        u_axis=np.array([-1., 0., 0.]), v_axis=np.array([0., 1., 0.])))


def dataset():
    rng = np.random.default_rng(97)
    ids = np.repeat(np.arange(30), 6)
    labels = np.tile(np.arange(6), 30)
    folds = np.repeat(np.arange(30) % 5, 6)
    reach = np.repeat(np.arange(30) % 6 + 1, 6)
    x = rng.normal(size=(180, 15)) + labels[:, None]/5
    x[0] = np.nan
    return x, labels, folds, ids, reach


def test_exact_immediate_onset_slices_and_no_overlap():
    intervals = core.stage_intervals([1000, 1900, 2700])
    np.testing.assert_array_equal(intervals, [[700, 1000], [1000, 1300], [1600, 1900],
                                             [1900, 2200], [2400, 2700], [2700, 3000]])
    assert (np.diff(intervals, axis=1) == 300).all()
    with pytest.raises(ValueError, match="overlap"):
        core.stage_intervals([1000, 1500, 2700])
    with pytest.raises(ValueError):
        core.stage_intervals([1000.1, 1900, 2700])


def test_cloud_uses_only_observed_window_and_all_coordinates():
    x = np.arange(300, dtype=np.float32)
    cloud = core.embed_window(x)
    assert cloud.shape == (294, 3)
    np.testing.assert_array_equal(cloud[0], [6, 3, 0])
    np.testing.assert_array_equal(cloud[-1], [299, 296, 293])
    np.testing.assert_array_equal(cloud[:, 1], x[3:297])


def test_preprocessing_local_normalization_and_failure_not_swallowed(monkeypatch):
    x = np.sin(np.arange(300)*.12) + .001*np.arange(300)
    np.testing.assert_allclose(core.preprocess_window(x), core.preprocess_window(7*x+12), atol=1e-6)
    with pytest.raises(ValueError):
        core.preprocess_window(np.ones(300))
    def failing_filter(*args, **kwargs):
        raise ValueError("test filter failure")
    monkeypatch.setattr(core, "bandpass_filter", failing_filter)
    with pytest.raises(ValueError, match="test filter failure"):
        core.preprocess_window(x)


def test_fifteen_features_include_mean_axial_half_width():
    x, reason = core.geometry_features(result())
    assert reason == "" and len(core.COLUMNS) == 15 and x.shape == (15,)
    np.testing.assert_array_equal(x[:6], [3, 2, .625, .1, .2, .8])
    np.testing.assert_array_equal(x[6:], [0, 0, 1, 1, 0, 0, 0, 1, 0])
    assert x[2] != x[0]-x[1]


def test_equivalent_signs_and_axes_preserve_all_features():
    original = result()
    changed = copy.deepcopy(original)
    changed["fit"]["direction"] *= -1
    changed["fit"]["u_axis"] *= -1
    f = changed["fit"]
    f["R1"], f["R2"] = f["R2"], f["R1"]
    f["R1_in"], f["R2_in"] = f["R2_in"], f["R1_in"]
    f["u_axis"], f["v_axis"] = f["v_axis"], f["u_axis"]
    np.testing.assert_array_equal(core.geometry_features(original)[0], core.geometry_features(changed)[0])


@pytest.mark.parametrize("invalid", ["failure", "nonconvergence", "invalid_width", "nonfinite_width", "invalid_radii"])
def test_unusable_fits_return_all_missing(invalid):
    r = result()
    if invalid == "failure":
        r = dict(status="failed", reason="failure")
    elif invalid == "nonconvergence":
        r["summary"]["optimizer_success"] = False
    elif invalid == "invalid_width":
        r["fit"]["minor_radius"] = 9
    elif invalid == "nonfinite_width":
        r["fit"]["minor_radius"] = np.nan
    else:
        r["fit"]["R1_in"] = 4
    x, reason = core.geometry_features(r)
    assert x.shape == (15,) and np.isnan(x).all() and reason


def test_bounds_and_circular_orientation_are_not_excluded():
    r = result()
    r["summary"]["flags"] = "active_optimizer_bound;near_optimizer_bound;nearly_circular_orientation"
    assert np.isfinite(core.geometry_features(r)[0]).all()


def test_trial_grouping_rejects_window_level_splits():
    _, y, folds, ids, _ = dataset()
    core.validate_groups(y, folds, ids)
    folds[0] = 1
    with pytest.raises(ValueError, match="cross folds"):
        core.validate_groups(y, folds, ids)


def test_decoder_scaling_and_model_independent_of_test_data():
    x, y, folds, ids, _ = dataset()
    with threadpool_limits(limits=1):
        pred, models = core.decode(x, y, folds, ids, True)
        changed = x.copy()
        changed[folds == 0] = 1e6
        _, modified = core.decode(changed, y, folds, ids, True)
    assert pred.shape == (180,)
    a, b = models[0], modified[0]
    assert list(a.named_steps) == ["impute", "scale", "lda"] and a.n_features_in_ == 15
    np.testing.assert_allclose(a["impute"].statistics_, np.nanmedian(x[folds != 0], axis=0))
    np.testing.assert_array_equal(a["impute"].statistics_, b["impute"].statistics_)
    np.testing.assert_array_equal(a["scale"].mean_, b["scale"].mean_)
    np.testing.assert_array_equal(a["lda"].coef_, b["lda"].coef_)


def test_all_missing_training_column_is_explicit_failure():
    x, y, folds, ids, _ = dataset()
    x[folds != 0, 0] = np.nan
    with pytest.raises(ValueError, match="training"):
        core.decode(x, y, folds, ids)


def test_metrics_match_sklearn_and_allow_zero_recall():
    _, y, _, _, _ = dataset()
    p = np.random.default_rng(8).integers(0, 5, len(y))
    s = core.scores(y, p)
    assert s["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", labels=np.arange(6)))
    np.testing.assert_allclose(s["confusion_fraction"], confusion_matrix(y, p, normalize="true"))


def test_permutations_preserve_one_of_each_stage_within_trial_and_refit():
    x, y, folds, ids, _ = dataset()
    shuffled = core.shuffle_stages(y, ids, 7)
    assert not np.array_equal(shuffled, y)
    for t in np.unique(ids):
        np.testing.assert_array_equal(np.sort(shuffled[ids == t]), np.arange(6))
    with threadpool_limits(limits=1):
        r = core.permutation(7, x, y, folds, ids)
        expected, _ = core.decode(x, shuffled, folds, ids)
    np.testing.assert_array_equal(r["predictions"], expected)
    assert r["macro_f1"] == core.scores(shuffled, expected)["macro_f1"]


def test_bootstrap_resamples_whole_trials_within_reach_strata():
    _, y, _, ids, reach = dataset()
    p = np.random.default_rng(8).integers(0, 6, len(y))
    boot = core.bootstrap(y, p, ids, reach, count=10)
    assert boot["trial_indices"].shape == (10, 30)
    for b, trial_sample in enumerate(boot["trial_indices"]):
        ix = np.concatenate([np.flatnonzero(ids == t) for t in trial_sample])
        np.testing.assert_array_equal(np.bincount(reach[ix])[1:], np.full(6, 30))
        np.testing.assert_array_equal(np.bincount(y[ix]), np.full(6, 30))
        assert boot["macro_f1"][b] == core.scores(y[ix], p[ix])["macro_f1"]


def test_fit_resume_reuses_only_identity_and_cloud_matched_checkpoint(tmp_path, monkeypatch):
    import run_stage_geometry_decoding as runner
    (tmp_path/"checkpoints").mkdir()
    cloud = core.embed_window(np.arange(300, dtype=np.float32))
    row = dict(window_index=0, original_trial_number=6, stage_name="pre_TC", preprocessing_error="")
    calls = []
    def fake_fit(points, model):
        calls.append(model)
        return dict(status="returned", elapsed_seconds=.1, cloud_sha256=runner.array_hash(points))
    monkeypatch.setattr(runner, "fit_one", fake_fit)
    runner.fit_window(tmp_path, row, cloud)
    _, _, status = runner.fit_window(tmp_path, row, cloud)
    assert status == "reused" and calls == ["one_torus"]
    with pytest.raises(AssertionError):
        runner.fit_window(tmp_path, row, cloud+1)


def test_plot_geometry_does_not_invoke_fit_or_decode(monkeypatch):
    import matplotlib.pyplot as plt
    import plot_stage_geometry_decoding as plotting
    import prego_geometric_fits
    import NeuralFieldManifold.fits.one_torus as native
    def prohibited(*args, **kwargs):
        raise AssertionError("Plotting must not fit or decode")
    monkeypatch.setattr(prego_geometric_fits, "fit_one", prohibited)
    monkeypatch.setattr(native, "one_torus_fit", prohibited)
    monkeypatch.setattr(core, "decode", prohibited)
    r = result()
    r["fit"]["center"] = [0, 0, 0]
    r["usable"] = True
    cloud = np.random.default_rng(73).normal(size=(294, 3))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plotting.cloud_axis(ax, cloud, r, 0, 5, True)
    fig.canvas.draw()
    plt.close(fig)
