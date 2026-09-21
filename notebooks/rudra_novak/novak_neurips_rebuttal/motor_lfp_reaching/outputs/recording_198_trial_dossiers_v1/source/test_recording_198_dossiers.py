import importlib.util
import unittest

import numpy as np
import pandas as pd


class DossierTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('build_recording_198_dossiers')
        self.assertIsNotNone(spec, 'Dossier builder has not been implemented')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_selection_is_only_trial_count_and_identifier(self):
        module = self.module()
        cohort = pd.DataFrame({'recording': ['z', 'a', 'b'], 'n_trials': [198, 198, 192]})
        self.assertEqual(module.selected_recordings(cohort), ['a', 'z'])

    def test_macro_f1_from_saved_predictions(self):
        module = self.module()
        frame = pd.DataFrame({'method': ['x'] * 6, 'raw_trial_index': range(6),
                              'true_direction': range(1, 7), 'predicted_direction': range(1, 7)})
        result = module.prediction_metrics(frame)
        self.assertEqual(result.iloc[0].macro_f1, 1.)
        frame['predicted_direction'] = [2, 3, 4, 5, 6, 1]
        self.assertEqual(module.prediction_metrics(frame).iloc[0].macro_f1, 0.)

    def test_repeated_prediction_is_rejected(self):
        module = self.module()
        frame = pd.DataFrame({'method': ['x'] * 2, 'raw_trial_index': [0, 0],
                              'true_direction': [1, 1], 'predicted_direction': [1, 1]})
        with self.assertRaises(ValueError):
            module.prediction_metrics(frame)

    def test_fold_identity_mismatch_is_rejected(self):
        module = self.module()
        arrays = dict(raw_trial_indices=np.arange(198), labels=np.repeat(np.arange(1, 7), 33),
                      original_trial_number=np.arange(198), folds=np.arange(198) % 5)
        module.validate_trials(arrays, arrays)
        changed = {**arrays, 'folds': np.zeros(198, dtype=int)}
        with self.assertRaises(ValueError):
            module.validate_trials(arrays, changed)


if __name__ == '__main__':
    unittest.main()
