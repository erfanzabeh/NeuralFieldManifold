from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_groups_follow_clockwise_targets_from_upper_right():
    from plot_prego_radius_left_right import group_directions
    table = pd.DataFrame({'direction': [6, 3, 1, 5, 2, 4]})
    result = group_directions(table)
    assert result.reach_side.tolist() == ['Leftward', 'Rightward', 'Rightward',
                                          'Leftward', 'Rightward', 'Leftward']
    assert result.target_position.tolist() == ['upper left', 'lower right', 'upper right',
                                              'left', 'right', 'lower left']
    assert result.target_event_code.tolist() == [106, 103, 101, 105, 102, 104]
    assert table.columns.tolist() == ['direction']


@pytest.mark.parametrize('invalid', [0, 7, 1.5, np.nan])
def test_unknown_direction_is_rejected(invalid):
    from plot_prego_radius_left_right import group_directions
    with pytest.raises(ValueError, match='direction'):
        group_directions(pd.DataFrame({'direction': [1, invalid]}))


def test_plot_preserves_coordinates_and_assigns_two_correct_colors():
    from plot_prego_radius_left_right import group_directions, radius_plot, COLORS
    rng = np.random.default_rng(8)
    table = pd.DataFrame(dict(R1=rng.normal(4, .5, 60), R2=rng.normal(1.2, .15, 60),
                              direction=np.repeat(np.arange(1, 7), 10)))
    grouped = group_directions(table)
    fig, ax, density = radius_plot(grouped)
    points = grouped.sample(frac=1, random_state=42)
    scatter = ax.collections[-1]
    np.testing.assert_array_equal(scatter.get_offsets(), points[['R1', 'R2']])
    np.testing.assert_allclose(scatter.get_facecolors(),
                               [to_rgba(COLORS[g], .66) for g in points.reach_side])
    assert ax.get_xlabel() == 'R1' and ax.get_ylabel() == 'R2'
    assert density['density'].shape == (2, 180, 180)
    assert density['groups'].tolist() == ['Leftward', 'Rightward']
    assert not any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())
    assert all('Directions' in text.get_text() for text in fig.legends[0].get_texts())
    plt.close(fig)


def test_frozen_data_accounting_and_original_trial_labels():
    from plot_prego_radius_left_right import load_grouped_trials
    table, pairs = load_grouped_trials()
    assert len(table) == 198 and len(pairs) == 197
    assert table.reach_side.value_counts().to_dict() == {'Leftward': 99, 'Rightward': 99}
    assert pairs.reach_side.value_counts().to_dict() == {'Leftward': 99, 'Rightward': 98}
    assert table.loc[~table.row_index.isin(pairs.row_index), 'original_trial_number'].tolist() == [128]
