from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_deviations_use_trial_weighted_pool_not_mean_of_direction_means():
    from plot_prego_betti_profiles import deviation_table
    curves = pd.DataFrame({'row_index': [0, 1, 2], 'direction': [1, 2, 2],
                           'homology_dimension': 1, 'grid_index': 0,
                           'distance': .2, 'betti': [6, 0, 3]})
    result = deviation_table(curves)
    np.testing.assert_allclose(result.pooled_mean_betti, [3, 3])
    np.testing.assert_allclose(result.delta_betti, [3, -1.5])


def test_lifetime_ranks_use_differences_not_deaths_and_account_for_missing_second():
    from plot_prego_betti_profiles import ranked_lifetimes
    trials = pd.DataFrame({'row_index': [0, 1, 2], 'original_trial_number': [2, 4, 8],
                           'direction': [1, 2, 3], 'status': 'success'})
    intervals = pd.DataFrame({'row_index': [0, 0, 1, 2], 'homology_dimension': [1, 1, 1, 0],
                              'birth': [.9, .1, .2, 0], 'death': [1, .6, .4, np.inf]})
    result = ranked_lifetimes(intervals, trials)
    np.testing.assert_allclose(result.longest_h1, [.5, .2, 0])
    np.testing.assert_allclose(result.second_longest_h1, [.1, 0, 0])
    np.testing.assert_allclose(result.lifetime_gap, [.4, .2, 0])
    assert result.h1_interval_count.tolist() == [2, 1, 0]
    intervals.loc[0, 'death'] = np.inf
    with pytest.raises(ValueError, match='Infinite H1'):
        ranked_lifetimes(intervals, trials)


def test_saved_data_all_trials_and_shared_heatmap_scale():
    from plot_prego_betti_profiles import ROOT, deviation_table, heatmap_panel, plt
    curves = pd.read_csv(ROOT / 'tables/betti_curves.csv')
    result = deviation_table(curves)
    assert len(result) == 3072 and result.n_trials.eq(33).all()
    np.testing.assert_allclose(result.groupby('grid_index').delta_betti.mean(), 0, atol=1e-12)
    full, ax, mesh = heatmap_panel(result)
    zoom, _, other = heatmap_panel(result, zoom=True)
    assert mesh.get_clim() == other.get_clim()
    assert mesh.get_clim()[0] == -mesh.get_clim()[1]
    assert ax.get_ylim() == (6.5, .5)
    np.testing.assert_allclose(np.asarray(mesh.get_array()).reshape(6, 512),
                               result.pivot(index='direction', columns='distance', values='delta_betti'))
    plt.close(full)
    plt.close(zoom)


def test_rank_scatter_preserves_all_saved_values_and_equal_axes():
    from plot_prego_betti_profiles import ROOT, ranked_lifetimes, lifetime_panel, plt
    intervals = pd.read_csv(ROOT / 'tables/persistence_intervals.csv', float_precision='round_trip')
    trials = pd.read_csv(ROOT / 'tables/trial_measurements.csv', float_precision='round_trip')
    ranks = ranked_lifetimes(intervals, trials)
    assert len(ranks) == 198 and ranks.row_index.is_unique
    np.testing.assert_allclose(ranks.longest_h1, trials.longest_H1_lifetime)
    for row in ranks.itertuples():
        with np.load(ROOT / 'diagrams' / f'trial_{row.row_index:03d}.npz') as saved:
            lifetimes = np.sort(np.diff(saved['H1'], axis=1).ravel())
        np.testing.assert_allclose([row.longest_h1, row.second_longest_h1], lifetimes[-2:][::-1])
    fig, ax = lifetime_panel(ranks)
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_aspect() == 1
    assert len(ax.collections) == 6
    for direction, scatter in enumerate(ax.collections, 1):
        points = scatter.get_offsets()
        assert len(points) == 33
        np.testing.assert_allclose(points, ranks.loc[ranks.direction.eq(direction),
                                                    ['second_longest_h1', 'longest_h1']])
    plt.close(fig)
