"""Plot-only contracts, tested without loading Ripser or any decoder."""
import importlib.util
from pathlib import Path
import sys
import subprocess
import unittest

import numpy as np
import pandas as pd

UNIT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(UNIT))


class PlotTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('plot_prego_persistent_homology'),
                             'PNG-only plot module is missing')
        import plot_prego_persistent_homology
        return plot_prego_persistent_homology

    def test_distribution_uses_every_trial_and_summary(self):
        api = self.api()
        table = pd.DataFrame({'direction': np.repeat(np.arange(1, 7), 3),
                              'value': np.arange(18)/10, 'row_index': np.arange(18)})
        fig, ax, plotted = api.distribution_panel(table, 'value', 'Measurement')
        self.assertEqual(len(plotted), 18)
        np.testing.assert_array_equal(plotted.value, table.value)
        self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines()))
        self.assertEqual(ax.get_title(), '')
        self.assertEqual(ax.get_ylabel(), 'Measurement')
        api.plt.close(fig)

    def test_outline_is_saved_ellipse_not_refitting(self):
        api = self.api()
        fit = dict(center=[1, 2, 3], u_axis=[1, 0, 0], v_axis=[0, 1, 0],
                   R1=2, R2=1, R1_in=.5, R2_in=.25)
        outer, inner = api.outlines(fit)
        for cloud, a, b in [(outer, 2, 1), (inner, .5, .25)]:
            np.testing.assert_allclose(((cloud[:, 0]-1)/a)**2 + ((cloud[:, 1]-2)/b)**2, 1)
            np.testing.assert_allclose(cloud[:, 2], 3)

    def test_no_computation_imported(self):
        self.api()
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import plot_prego_persistent_homology; "
            "assert not {'run_prego_persistent_homology','ripser','prego_decoding'} & set(sys.modules)"],
            cwd=UNIT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_examples_reject_wrong_direction_or_trial(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'validate_examples'), 'Strict example selection check missing')
        trials = pd.DataFrame({'row_index': range(12), 'direction': np.repeat(range(1, 7), 2),
                               'original_trial_number': np.tile([10, 5], 6) + np.repeat(np.arange(6)*20, 2)})
        chosen = trials.iloc[[1, 3, 5, 7, 9, 11]].reset_index(drop=True)
        api.validate_examples(chosen, trials)
        wrong = chosen.copy()
        wrong.loc[0, 'original_trial_number'] = 10
        with self.assertRaises(AssertionError):
            api.validate_examples(wrong, trials)


if __name__ == '__main__':
    unittest.main(verbosity=2)
