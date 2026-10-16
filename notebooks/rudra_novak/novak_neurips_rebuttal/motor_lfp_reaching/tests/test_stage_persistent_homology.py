"""Synthetic PH and frozen cue-offset input checks, independent of torus fitting."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage_persistent_homology as ph


def square():
    return np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])


def test_square_loop():
    d = ph.compute_diagrams(square())
    ph.validate_diagrams(d, 4)
    np.testing.assert_allclose(d[1], [[1, np.sqrt(2)]], atol=1e-6)
    assert len(d[2]) == 0
    np.testing.assert_array_equal(ph.betti_curve(d[0], np.array([0., .5, 1., 10.])), [4, 4, 1, 1])


def test_octahedral_surface():
    cloud = np.concatenate([np.eye(3), -np.eye(3)])
    d = ph.compute_diagrams(cloud)
    ph.validate_diagrams(d, 6)
    np.testing.assert_allclose(d[2], [[np.sqrt(2), 2]], atol=1e-6)


def test_betti_boundaries_and_infinity():
    d = np.array([[0., np.inf], [.1, .3], [.2, .3]])
    np.testing.assert_array_equal(ph.betti_curve(d, np.array([0, .1, .2, .3, 1.])), [1, 2, 3, 1, 1])
    np.testing.assert_array_equal(ph.betti_curve(np.empty((0, 2)), np.linspace(0, 1, 5)), np.zeros(5))
    assert ph.longest_lifetime(np.empty((0, 2))) == 0
    assert np.isnan(ph.longest_lifetime(np.array([[0., np.inf]])))
    assert ph.longest_lifetime(np.array([[0., .5], [.2, 1.]])) == .8


def sorted_diagram(d):
    return d[np.lexsort((d[:, 1], d[:, 0]))]


def test_translation_rotation_scaling():
    rng = np.random.default_rng(204)
    cloud = rng.normal(size=(24, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    ref = ph.compute_diagrams(cloud)
    moved = ph.compute_diagrams(cloud @ q + [5., -3., 2.])
    scaled = ph.compute_diagrams(cloud * 3.5)
    for a, b, c in zip(ref, moved, scaled):
        np.testing.assert_allclose(sorted_diagram(a), sorted_diagram(b), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sorted_diagram(a)*3.5, sorted_diagram(c), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ph.centered_rms(cloud @ q + 10), ph.centered_rms(cloud), atol=1e-12)
    np.testing.assert_allclose(ph.longest_lifetime(ref[1])/ph.centered_rms(cloud),
                               ph.longest_lifetime(scaled[1])/ph.centered_rms(cloud*3.5), rtol=1e-6)


def test_invalid_input():
    with pytest.raises(ValueError):
        ph.compute_diagrams(np.array([[0., np.nan, 1.]]))
    with pytest.raises(ValueError):
        ph.compute_diagrams(np.zeros((4, 2)))
    assert ph.centered_rms(np.ones((4, 3))) == 0


def test_frozen_inputs_and_pilot():
    clouds, metadata = ph.load_inputs()
    assert np.isfinite(clouds).all()
    rows = metadata.iloc[ph.pilot_rows(metadata)]
    assert rows.original_trial_number.tolist() == [6]*6 + [9]*6 + [10]*6
    assert rows.stage.tolist() == list(range(6))*3
    assert clouds.shape == (1188, 294, 3)
    assert set(metadata.loc[metadata.stage_name == "post_TC", "anchor_code"]) == {203}
    assert set(metadata.loc[metadata.stage_name == "post_SC", "anchor_code"]) == {206}


def test_checkpoint_reuse_and_hash_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "OUTPUT", tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "diagrams").mkdir()
    ph.write_json(tmp_path / "config.json", {"test": True})
    cloud = square()
    d = ph.compute_diagrams(cloud)
    meta_path, data_path = ph.checkpoint_paths(0)
    np.savez_compressed(data_path, **{f"H{h}": a for h, a in enumerate(d)})
    row = dict(window_index=0, original_trial_number=6, trial_index=0, stage_name="pre_TC")
    ph.write_json(meta_path, dict(row, status="success", reason="", elapsed_seconds=0.01,
        cloud_sha256=ph.array_digest(cloud), config_sha256=ph.digest(tmp_path / "config.json"),
        diagram_sha256=ph.digest(data_path)))
    import pandas as pd
    before = ph.digest(meta_path)
    _, reused = ph.run_window(0, np.array([cloud]), pd.DataFrame([row]))
    assert reused and ph.digest(meta_path) == before
    with pytest.raises(AssertionError):
        ph.read_checkpoint(0, cloud + 1, row)


@pytest.mark.skipif(not (ph.OUTPUT / "validation/numerical_checks.json").exists(), reason="Full run not yet complete")
def test_saved_results_independently():
    import pandas as pd
    measures = pd.read_csv(ph.OUTPUT / "tables/window_measurements.csv", float_precision="round_trip")
    summaries = pd.read_csv(ph.OUTPUT / "tables/betti_summary.csv", float_precision="round_trip")
    np.testing.assert_array_equal(measures.window_index, np.arange(1188))
    with np.load(ph.OUTPUT / "tables/betti_curves.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
        for row in measures.itertuples():
            if row.status != "success":
                for mode in ("raw", "rms_normalized"):
                    for h in range(3):
                        assert np.isnan(saved[f"{mode}_H{h}"][row.window_index]).all()
                continue
            with np.load(ph.OUTPUT / "diagrams" / f"window_{row.window_index:04d}.npz", allow_pickle=False) as data:
                for mode in ("raw", "rms_normalized"):
                    if mode == "rms_normalized" and not row.normalization_valid:
                        continue
                    divisor = 1. if mode == "raw" else row.centered_rms
                    for h in range(3):
                        d = data[f"H{h}"]/divisor
                        grid = saved[f"{mode}_H{h}_grid"]
                        # Independent event-count algorithm, not the analysis broadcast helper.
                        births = np.searchsorted(np.sort(d[:, 0]), grid, side="right")
                        deaths = np.searchsorted(np.sort(d[:, 1]), grid, side="right")
                        np.testing.assert_array_equal(births-deaths, saved[f"{mode}_H{h}"][row.window_index])
                lifetimes = np.diff(data["H1"], axis=1).ravel()
                longest = lifetimes.max() if len(lifetimes) else 0.
                np.testing.assert_allclose(longest, row.longest_H1_lifetime, rtol=0, atol=1e-12)
                if row.normalization_valid:
                    np.testing.assert_allclose(longest/row.centered_rms, row.normalized_H1_lifetime, rtol=1e-12)
        for (mode, h, stage), group in summaries[summaries.cohort == "available"].groupby(
                ["distance_convention", "homology_dimension", "stage"]):
            group = group.sort_values("grid_index")
            values = saved[f"{mode}_H{h}"][measures.stage == stage]
            values = values[np.isfinite(values).all(axis=1)]
            assert group.n_windows.eq(len(values)).all()
            if len(values):
                np.testing.assert_allclose(group["mean"], values.mean(axis=0), rtol=1e-12, atol=1e-12)
                for q, column in ((.25, "q25"), (.5, "median"), (.75, "q75")):
                    np.testing.assert_allclose(group[column], np.quantile(values, q, axis=0), rtol=1e-12, atol=1e-12)
