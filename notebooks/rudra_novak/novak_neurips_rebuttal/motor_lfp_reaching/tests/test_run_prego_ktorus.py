from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def runner():
    return importlib.import_module("run_prego_ktorus")


def test_existing_run_is_never_overwritten(runner, tmp_path):
    root = tmp_path / "run"
    config = dict(version=1, seed=42)
    runner.initialize_run(root, config, {"file": "hash"})
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.initialize_run(root, config, {"file": "hash"})
    assert before == (root / "manifest.json").read_bytes()
    runner.initialize_run(root, config, {"file": "hash"}, resume=True)


@pytest.mark.parametrize("change", ["config", "source"])
def test_resume_rejects_stale_provenance(runner, tmp_path, change):
    root = tmp_path / "run"
    config = dict(seed=42)
    runner.initialize_run(root, config, {"file": "hash"})
    with pytest.raises(ValueError):
        runner.initialize_run(root, dict(seed=43) if change == "config" else config,
                              {"file": "changed"} if change == "source" else {"file": "hash"}, resume=True)


def test_output_cannot_be_inside_or_above_previous_run(runner):
    for path in [runner.PREVIOUS, runner.PREVIOUS / "oops", runner.PREVIOUS.parent]:
        with pytest.raises(ValueError):
            runner.validate_output(path)
    runner.validate_output(runner.DEFAULT_OUTPUT)


def test_safe_json_converts_nonfinite_diagnostics_to_null(runner, tmp_path):
    path = tmp_path / "test.json"
    runner.write_json(path, dict(a=np.array([1., np.inf]), b=np.float64(np.nan)))
    assert json.loads(path.read_text()) == dict(a=[1., None], b=None)


def test_checkpoint_digest_detects_modifications(runner, tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"one")
    original = runner.sha256(path)
    path.write_bytes(b"two")
    assert runner.sha256(path) != original


def test_trial_and_fold_hash_check(runner):
    current = dict(raw_trial_indices=np.arange(10), folds=np.arange(10)%5,
                   labels=np.arange(10)%6+1, original_trial_number=np.arange(10)+20)
    runner.verify_selection(current, current)
    changed = {k: v.copy() for k, v in current.items()}
    changed["raw_trial_indices"][0] = 12
    with pytest.raises(ValueError):
        runner.verify_selection(current, changed)


def test_original_trial_ids_are_checked(runner):
    current = dict(raw_trial_indices=np.arange(10), folds=np.arange(10)%5,
                   labels=np.arange(10)%6+1, original_trial_number=np.arange(10)+20)
    changed = {k: v.copy() for k, v in current.items()}
    changed["original_trial_number"][0] = 222
    with pytest.raises(ValueError):
        runner.verify_selection(current, changed)


def test_environment_is_part_of_resume_validation(runner, tmp_path, monkeypatch):
    root = tmp_path / "run"
    monkeypatch.setattr(runner, "environment_info", lambda: {"python": "A"})
    runner.initialize_run(root, {}, {})
    monkeypatch.setattr(runner, "environment_info", lambda: {"python": "B"})
    with pytest.raises(ValueError, match="Environment"):
        runner.initialize_run(root, {}, {}, resume=True)


def test_fold_binding_rejects_swapped_checkpoint(runner):
    embedding = dict(K=2, tau=14, dimension=5, frequencies=[12., 28.], training_trial_indices=[0, 1, 3],
                     feature_names=["a", "b"], selection_hash="selection")
    checkpoint = dict(binding=runner.fold_binding(embedding, 0),
                      raw_trial_indices=np.arange(4), feature_names=np.array(["a", "b"]),
                      features=np.ones((4, 2)))
    runner.validate_fold_checkpoint(checkpoint, embedding, 0, np.arange(4))
    with pytest.raises(ValueError):
        runner.validate_fold_checkpoint(checkpoint, embedding, 1, np.arange(4))
    modified = dict(embedding, tau=19)
    with pytest.raises(ValueError):
        runner.validate_fold_checkpoint(checkpoint, modified, 0, np.arange(4))


def test_completed_artifacts_reject_missing_diagnostics(runner, tmp_path):
    (tmp_path / "features.npz").write_bytes(b"features")
    (tmp_path / "diagnostics.csv").write_text("errors\n0.1\n")
    artifacts = runner.artifact_hashes(tmp_path)
    runner.verify_artifacts(tmp_path, artifacts)
    (tmp_path / "diagnostics.csv").unlink()
    with pytest.raises(ValueError):
        runner.verify_artifacts(tmp_path, artifacts)


def test_all_seven_methods_have_null_coverage(runner):
    from prego_ktorus import METHODS
    assert runner.NULL_METHODS == METHODS
