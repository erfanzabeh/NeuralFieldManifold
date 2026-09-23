from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_selection_is_unique_label_blind_and_order_independent():
    from plot_prego_geometry_examples import ROOT, select_examples
    measures = pd.read_csv(ROOT / 'tables/trial_measurements.csv')
    chosen = select_examples(measures)
    assert len(chosen) == chosen.row_index.nunique() == 6
    assert chosen.geometry_usable.all()
    shuffled = measures.sample(frac=1, random_state=12).copy()
    shuffled['direction'] = 7 - shuffled.direction
    repeated = select_examples(shuffled)
    assert repeated.row_index.tolist() == chosen.row_index.tolist()
    assert chosen.original_trial_number.tolist() == [90, 592, 527, 205, 389, 581]


def test_selection_ties_use_original_trial_number_and_exclude_unusable():
    from plot_prego_geometry_examples import select_examples
    measures = pd.DataFrame({'row_index': range(8), 'original_trial_number': range(18, 10, -1),
                             'geometry_usable': [False] + [True] * 7, 'status': 'success',
                             'normalized_H1_lifetime': 1., 'R1': 2., 'R2': 1.})
    result = select_examples(measures)
    assert result.original_trial_number.tolist() == [11, 12, 13, 14, 15, 16]
    assert 0 not in result.row_index.values


def test_panel_preserves_cloud_coordinates_outlines_and_shared_limits():
    from plot_prego_geometry_examples import common_limits, make_panel, plt
    cloud = np.random.default_rng(15).normal(size=(494, 3))
    boundary = (cloud[:20] * 2, cloud[:20] * .5)
    original = cloud.copy()
    limits = common_limits([cloud], [boundary])
    fig = make_panel(cloud, boundary, 2, 622, limits)
    ax = fig.axes[0]
    np.testing.assert_array_equal(np.column_stack(ax.collections[0]._offsets3d), original)
    np.testing.assert_array_equal(cloud, original)
    for line, expected in zip(ax.lines, boundary):
        np.testing.assert_array_equal(np.column_stack(line.get_data_3d()), expected)
    assert ax.get_xlim() == ax.get_ylim() == ax.get_zlim() == limits
    assert limits[0] < min(cloud.min(), boundary[0].min())
    assert limits[1] > max(cloud.max(), boundary[0].max())
    assert (ax.elev, ax.azim) == (25, -60)
    assert fig.texts[0].get_text() == 'Trial 622 | Direction 2'
    plt.close(fig)
