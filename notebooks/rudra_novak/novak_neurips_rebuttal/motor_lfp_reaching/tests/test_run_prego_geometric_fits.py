import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_existing_output_is_protected_and_resume_checks_inputs(tmp_path):
    run = importlib.import_module("run_prego_geometric_fits")
    root = tmp_path/"experiment"
    run.initialize(root, {"tau": 16}, {"input": "abc"}, False)
    with pytest.raises(FileExistsError):
        run.initialize(root, {"tau": 16}, {"input": "abc"}, False)
    run.initialize(root, {"tau": 16}, {"input": "abc"}, True)
    with pytest.raises(ValueError):
        run.initialize(root, {"tau": 15}, {"input": "abc"}, True)
    with pytest.raises(ValueError):
        run.initialize(root, {"tau": 16}, {"input": "changed"}, True)


def test_source_trials_and_raw_epochs_match_exactly():
    run = importlib.import_module("run_prego_geometric_fits")
    data, trials, alignment = run.load_inputs(run.INPUT)
    assert data["segments"].shape == (198, 500)
    assert trials.groupby("direction").size().tolist() == [33]*6
    assert alignment["go_sample_zero_based"] == 4500
    assert (trials.delay == "short").all()
    np.testing.assert_array_equal(trials.original_trial_number, data["original_trial_number"])


def test_checkpoint_rejects_wrong_cloud(tmp_path):
    run = importlib.import_module("run_prego_geometric_fits")
    p = tmp_path/"fit.json"
    p.write_text(json.dumps(dict(model="one_torus", row_index=0, cloud_sha256="wrong")))
    with pytest.raises(ValueError, match="checkpoint"):
        run.validate_checkpoint(p, 0, "one_torus", np.zeros((468, 3)))
