"""Independent analytic checks for the frozen PH experiment."""
import importlib.util
from pathlib import Path
import sys
import unittest
import io
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

UNIT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(UNIT))


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if importlib.util.find_spec('run_prego_persistent_homology'):
            import run_prego_persistent_homology
            cls.module = run_prego_persistent_homology

    def api(self):
        self.assertIsNotNone(self.module, 'Independent persistence implementation is missing')
        return self.module

    def test_square_loop(self):
        api = self.api()
        x = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.]])
        d = api.compute_diagrams(x)
        np.testing.assert_allclose(d[1], [[1, np.sqrt(2)]], rtol=1e-6)
        self.assertEqual(np.isinf(d[0][:, 1]).sum(), 1)
        self.assertEqual(len(d[2]), 0)

    def test_octahedron_surface(self):
        d = self.api().compute_diagrams(np.r_[np.eye(3), -np.eye(3)])
        np.testing.assert_allclose(d[2], [[np.sqrt(2), 2]], rtol=1e-6)

    def test_betti_half_open_and_infinite_intervals(self):
        f = self.api().betti_curve
        diagram = np.array([[0, 1], [.5, 2], [0, np.inf]])
        np.testing.assert_array_equal(f(diagram, np.array([0, .5, 1, 2, 3])), [2, 3, 2, 1, 1])
        np.testing.assert_array_equal(f(np.empty((0, 2)), np.arange(4)), [0]*4)

    def test_lifetime_empty_infinite_and_rms(self):
        api = self.api()
        self.assertEqual(api.longest_finite_lifetime(np.empty((0, 2))), 0)
        self.assertTrue(np.isnan(api.longest_finite_lifetime(np.array([[1, np.inf]]))))
        self.assertEqual(api.longest_finite_lifetime(np.array([[1, 4], [0, 2]])), 3)
        self.assertEqual(api.centered_rms(np.array([[-1., 0, 0], [1, 0, 0]])), 1)

    def test_rigid_motion_and_scale(self):
        api = self.api()
        x = np.r_[np.eye(3), -np.eye(3)]
        angle = .713
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        base = api.compute_diagrams(x)
        moved = api.compute_diagrams(x @ rotation.T + [5, -2, 3])
        scaled = api.compute_diagrams(3*x)
        for d, e, f in zip(base, moved, scaled):
            np.testing.assert_allclose(d, e, atol=5e-6)
            np.testing.assert_allclose(d*3, f, atol=5e-6)
        self.assertAlmostEqual(api.longest_finite_lifetime(base[2])/api.centered_rms(x),
                               api.longest_finite_lifetime(scaled[2])/api.centered_rms(3*x), places=6)

    def test_frozen_clouds_and_selection(self):
        api = self.api()
        clouds, trials = api.load_inputs()
        self.assertEqual(clouds.shape, (198, 494, 3))
        self.assertEqual(trials.direction.value_counts().tolist(), [33]*6)
        self.assertEqual(api.pilot_rows(trials), [156, 194, 39, 67, 1, 110])
        selected = api.example_rows(trials)
        self.assertEqual(selected, [1, 39, 67, 110, 156, 194])

    def test_csv_validation_preserves_missing_numbers(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'verify_table'), 'Missing-value-safe CSV verification missing')
        table = pd.DataFrame({'R1': [1.5, np.nan], 'failure_reason': ['', 'timeout'],
                              'usable': [True, False]})
        api.verify_table(io.StringIO(table.to_csv(index=False)), table)
        wrong = table.copy()
        wrong.loc[0, 'R1'] = 99
        with self.assertRaises(AssertionError):
            api.verify_table(io.StringIO(wrong.to_csv(index=False)), table)

    def test_source_amendment_rejects_calculation_changes(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'check_calculation_unchanged'), 'Calculation freeze check missing')
        original = 'PARAMETERS = {"coeff":2}\ndef compute_diagrams(x):\n return x\ndef verify():\n return 0\n'
        api.check_calculation_unchanged(original, original.replace('return 0', 'return 1'))
        with self.assertRaises(ValueError):
            api.check_calculation_unchanged(original, original.replace('return x', 'return x*2'))
        with self.assertRaises(ValueError):
            api.check_calculation_unchanged(original, original.replace('"coeff":2', '"coeff":41'))

    def test_resume_reuses_success_and_failure_without_computation(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root/'checkpoints').mkdir()
            (root/'diagrams').mkdir()
            api.write_json(root/'config.json', {'test': True})
            clouds = np.array([[[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
            trials = pd.DataFrame({'original_trial_number': [23]})
            diagram_path = root/'diagrams/trial_000.npz'
            np.savez_compressed(diagram_path, H0=np.array([[0, 1], [0, 1], [0, 1], [0, np.inf]]),
                                H1=np.array([[1, np.sqrt(2)]]), H2=np.empty((0, 2)))
            with patch.object(api, 'OUTPUT', root), patch.object(api.subprocess, 'run',
                    side_effect=AssertionError('Resume unexpectedly started a worker')):
                for status in ('success', 'failed', 'timeout'):
                    api.write_json(root/'checkpoints/trial_000.json', dict(
                        row_index=0, original_trial_number=23, cloud_sha256=api.array_digest(clouds[0]),
                        config_sha256=api.digest(root/'config.json'), status=status,
                        reason='' if status == 'success' else 'known failure',
                        diagram_sha256=api.digest(diagram_path)))
                    before = api.digest(root/'checkpoints/trial_000.json')
                    result, reused = api.run_trial(0, clouds, trials)
                    self.assertTrue(reused)
                    self.assertEqual(result['status'], status)
                    self.assertEqual(api.digest(root/'checkpoints/trial_000.json'), before)
                with self.assertRaises(AssertionError):
                    api.run_trial(0, clouds+1, trials)


if __name__ == '__main__':
    unittest.main(verbosity=2)
