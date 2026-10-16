import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_checkpoints_detect_changed_artifacts(tmp_path):
    run = importlib.import_module("run_prego_ols_audit")
    p = tmp_path / "scores.csv"
    p.write_text("a\n1\n")
    expected = run.artifact_hashes(tmp_path)
    run.verify_artifacts(tmp_path, expected)
    p.write_text("a\n2\n")
    with pytest.raises(ValueError, match="Changed"):
        run.verify_artifacts(tmp_path, expected)


def test_initialization_refuses_overwrite_and_changed_configuration(tmp_path):
    run = importlib.import_module("run_prego_ols_audit")
    root = tmp_path / "prego_ols_audit_v1"
    run.initialize(root, {"p": 20}, {"source": "abc"}, False)
    with pytest.raises(FileExistsError):
        run.initialize(root, {"p": 20}, {"source": "abc"}, False)
    run.initialize(root, {"p": 20}, {"source": "abc"}, True)
    with pytest.raises(ValueError):
        run.initialize(root, {"p": 21}, {"source": "abc"}, True)


def test_trial_selection_rejects_wrong_delay_and_changed_labels():
    run = importlib.import_module("run_prego_ols_audit")
    indices = np.arange(30)
    saved = dict(raw_trial_indices=indices, labels=np.tile(np.arange(1, 7), 5),
                 folds=np.repeat(np.arange(5), 6), original_trial_number=indices+1)
    data = dict(direction=saved["labels"].copy(), original_trial_number=indices+1,
                delay_label=np.array(["short"]*30))
    run.validate_trials(data, indices, saved)
    data["delay_label"][0] = "long"
    with pytest.raises(ValueError):
        run.validate_trials(data, indices, saved)


def test_serialization_marks_undefined_numbers(tmp_path):
    run = importlib.import_module("run_prego_ols_audit")
    p = tmp_path / "value.json"
    run.write_json(p, dict(value=np.nan, array=np.array([np.inf, 1])))
    assert json.loads(p.read_text()) == dict(value=None, array=[None, 1])
