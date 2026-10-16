from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_only_valid_raw_pairs_are_plotted_without_transformation():
    from plot_prego_radius_pair import valid_pairs
    table = pd.DataFrame(dict(row_index=[0, 1, 2, 3], direction=[1, 2, 3, 4],
                              R1=[4., np.nan, 6., 5.], R2=[1., np.nan, 2., np.inf]))
    result = valid_pairs(table)
    assert result.row_index.tolist() == [0, 2]
    np.testing.assert_array_equal(result[['R1', 'R2']], [[4., 1.], [6., 2.]])


def test_joint_plot_contains_all_pairs_on_radius_axes_and_six_colors():
    from plot_prego_radius_pair import radius_plot
    rng = np.random.default_rng(8)
    table = pd.DataFrame(dict(R1=rng.normal(4, .5, 60), R2=rng.normal(1.2, .15, 60),
                              direction=np.repeat(np.arange(1, 7), 10)))
    fig, ax, density = radius_plot(table)
    assert ax.get_xlabel() == 'R1'
    assert ax.get_ylabel() == 'R2'
    scatter = ax.collections[-1]
    offsets = np.asarray(scatter.get_offsets())
    expected = table.sample(frac=1, random_state=42)[['R1', 'R2']].to_numpy()
    np.testing.assert_array_equal(offsets, expected)
    assert len(np.unique(scatter.get_facecolors(), axis=0)) == 6
    assert density['density'].shape == (6, 180, 180)
    assert not any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())
    plt.close(fig)
