import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_frozen_embedding_keeps_original_coordinates_and_trial_boundaries():
    from run_prego_fixed_geometry_decoding import frozen_clouds
    x = np.arange(198*500).reshape(198, 500).astype(np.float32)
    clouds = frozen_clouds(x)
    assert clouds.shape == (198, 494, 3)
    for coordinate, offset in enumerate((6, 3, 0)):
        np.testing.assert_array_equal(clouds[:, :, coordinate], x[:, offset:offset+494])
    with pytest.raises(ValueError):
        frozen_clouds(x[:, :-1])


def test_resume_rejects_changed_inputs_or_configuration(tmp_path):
    from run_prego_fixed_geometry_decoding import initialize
    root = tmp_path/'run'
    initialize(root, {'tau': 3}, {'input': 'hash'}, False)
    initialize(root, {'tau': 3}, {'input': 'hash'}, True)
    with pytest.raises(FileExistsError):
        initialize(root, {'tau': 3}, {'input': 'hash'}, False)
    with pytest.raises(ValueError):
        initialize(root, {'tau': 4}, {'input': 'hash'}, True)
    with pytest.raises(ValueError):
        initialize(root, {'tau': 3}, {'input': 'new'}, True)


def test_permutation_checkpoint_rejects_wrong_labels_and_scores():
    from run_prego_fixed_geometry_decoding import validate_null
    from prego_fixed_geometry_decoding import score_predictions
    from prego_decoding import permute_within_folds
    labels = np.tile(np.arange(1, 7), 10)
    folds = np.repeat(np.arange(5), 12)
    shuffled = permute_within_folds(labels, folds, 52000)
    predictions = np.tile(shuffled[:, None], (1, 5))
    scores = score_predictions(shuffled, predictions)
    saved = dict(index=np.array([0]), labels=shuffled[None], predictions=predictions[None],
                 macro_f1=scores['macro_f1'][None], accuracy=scores['accuracy'][None])
    validate_null(saved, labels, folds, [0])
    saved['macro_f1'][0, 0] = .2
    with pytest.raises(AssertionError):
        validate_null(saved, labels, folds, [0])
    saved['macro_f1'][0, 0] = 1.
    saved['labels'][0, 0] = 7
    with pytest.raises(AssertionError):
        validate_null(saved, labels, folds, [0])


def test_resume_rejects_environment_change_without_overwriting_record(tmp_path, monkeypatch):
    import run_prego_fixed_geometry_decoding as runner
    root = tmp_path/'run'
    runner.initialize(root, {}, {}, False)
    before = (root/'numerical_environment.json').read_bytes()
    monkeypatch.setattr(runner, 'numerical_environment', lambda: {'numpy': 'changed'})
    with pytest.raises(ValueError, match='environment'):
        runner.initialize(root, {}, {}, True)
    assert (root/'numerical_environment.json').read_bytes() == before
