from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_exact_scan_boundaries_and_opposite_pair_objective():
    from plot_prego_betti_threshold_scan import scan_thresholds
    trials = pd.DataFrame({'row_index': range(6), 'direction': range(1, 7), 'status': 'success'})
    intervals = pd.DataFrame({'row_index': [0, 2, 4, 5], 'homology_dimension': [1, 1, 1, 0],
                              'birth': [.1, .2, .2, 0], 'death': [.4, .4, .3, np.inf]})
    sweep, curves, selection = scan_thresholds(intervals, trials)
    np.testing.assert_allclose(sweep.distance, [0, .1, .2, .3, .4])
    np.testing.assert_array_equal(sweep.sum_absolute_mean_difference, [0, 1, 3, 2, 0])
    assert selection['distance'] == .2
    assert selection['maximizing_intervals'] == [{'start_inclusive': .2, 'end_exclusive': .3}]
    assert curves[(curves.direction == 5) & (curves.distance == .3)]['mean'].iloc[0] == 0


def test_ties_choose_first_and_simultaneous_birth_death_do_not_overlap():
    from plot_prego_betti_threshold_scan import scan_thresholds
    trials = pd.DataFrame({'row_index': range(6), 'direction': range(1, 7), 'status': 'success'})
    intervals = pd.DataFrame({'row_index': [0, 0], 'homology_dimension': [1, 1],
                              'birth': [.1, .2], 'death': [.2, .3]})
    sweep, _, selection = scan_thresholds(intervals, trials)
    assert selection['distance'] == .1
    assert sweep.sum_absolute_mean_difference.max() == 1


def test_saved_scan_matches_independent_searchsorted_counts_at_all_events():
    from plot_prego_betti_threshold_scan import ROOT, scan_thresholds
    intervals = pd.read_csv(ROOT / 'tables/persistence_intervals.csv', float_precision='round_trip')
    trials = pd.read_csv(ROOT / 'tables/trial_measurements.csv')
    sweep, _, selection = scan_thresholds(intervals, trials)
    h1 = intervals[intervals.homology_dimension == 1]
    independently_counted = {}
    for direction in range(1, 7):
        group = h1[h1.direction == direction]
        births = np.searchsorted(np.sort(group.birth), sweep.distance, side='right')
        deaths = np.searchsorted(np.sort(group.death), sweep.distance, side='right')
        means = (births - deaths) / 33
        independently_counted[direction] = means
        np.testing.assert_allclose(sweep[f'mean_direction_{direction}'], means, atol=0)
    objective = sum(np.abs(independently_counted[a]-independently_counted[b])
                    for a, b in [(1, 4), (2, 5), (3, 6)])
    np.testing.assert_allclose(sweep.sum_absolute_mean_difference, objective)
    assert np.isclose(selection['score'], objective.max())
    assert selection['distance'] == 0.22000855207443237


def test_reference_marks_both_distances_without_changing_curve_values():
    from plot_prego_betti_threshold_scan import marked_curves, plt
    curves = pd.DataFrame({'distance': [0., .2, .3] * 6,
                          'direction': np.repeat(range(1, 7), 3),
                          'mean': [0., 10., 0.] * 6, 'q25': [0., 8., 0.] * 6,
                          'q75': [0., 12., 0.] * 6})
    fig = marked_curves(curves, .22, zoom=True)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[-2].get_xdata(), [.23, .23])
    np.testing.assert_allclose(ax.lines[-1].get_xdata(), [.22, .22])
    for line in ax.lines[:6]:
        np.testing.assert_allclose(line.get_ydata(), [0, 10, 0])
    assert ax.get_xlim() == (.16, .3)
    fig.canvas.draw()
    for value, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if .16 <= value <= .30:
            bounds = label.get_window_extent(fig.canvas.get_renderer())
            assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width
    plt.close(fig)
