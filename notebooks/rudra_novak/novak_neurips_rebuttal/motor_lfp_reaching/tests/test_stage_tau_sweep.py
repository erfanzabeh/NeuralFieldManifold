from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage_tau_sweep as sweep


def test_native_and_matched_clouds_share_the_correct_time_anchors():
    signal = np.arange(300, dtype=np.float32)
    for tau in sweep.TAUS:
        native = sweep.cloud(signal, tau, "native")
        matched = sweep.cloud(signal, tau, "matched260")
        assert native.shape == (300 - 2*tau, 3)
        assert matched.shape == (260, 3)
        np.testing.assert_array_equal(native[0], [2*tau, tau, 0])
        np.testing.assert_array_equal(native[-1], [299, 299-tau, 299-2*tau])
        np.testing.assert_array_equal(matched[0], [40, 40-tau, 40-2*tau])
        np.testing.assert_array_equal(matched[-1], native[-1])
        np.testing.assert_array_equal(matched, native[-260:])
    np.testing.assert_array_equal(sweep.cloud(signal, 20, "native"),
                                  sweep.cloud(signal, 20, "matched260"))
    assert sweep.canonical_variant("matched260", 20) == "native"
    assert sweep.canonical_variant("matched260", 3) == "matched260"


def test_checkpoint_checks_trial_and_cloud_identity(tmp_path):
    signal = np.arange(300, dtype=np.float32)
    points = sweep.cloud(signal, 3, "native")
    row = dict(window_index=12, original_trial_number=6, stage_name="pre_TC")
    path = tmp_path / "fit.json"
    sweep.write_json(path, dict(**row, tau_ms=3, cloud_sha256=sweep.array_hash(points)))
    sweep.checked_fit(path, row, points, 3)
    with pytest.raises(ValueError, match="identity/hash"):
        sweep.checked_fit(path, row, points, 4)
    with pytest.raises(ValueError, match="identity/hash"):
        sweep.checked_fit(path, row, points + 1, 3)


def test_frozen_corrected_windows_and_original_tau3_clouds():
    metadata, processed = sweep.load_inputs()
    assert len(metadata) == 1188
    assert processed.shape == (1188, 300)
    assert (metadata[metadata.stage_name == "post_TC"].anchor_code == 203).all()
    assert (metadata[metadata.stage_name == "post_SC"].anchor_code == 206).all()
    assert set(metadata.groupby("trial_index").size()) == {6}
