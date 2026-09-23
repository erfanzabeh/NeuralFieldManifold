from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_exact_counts_include_birth_exclude_death_and_preserve_empty_trials():
    from plot_prego_betti_opposites import counts_at_distance
    trials = pd.DataFrame({'row_index': [0, 1, 2], 'direction': [1, 4, 2],
                           'status': ['success'] * 3})
    intervals = pd.DataFrame({
        'row_index': [0, 0, 0, 0, 1], 'homology_dimension': [1, 1, 1, 0, 1],
        'birth': [.25, 0, .1, 0, .251], 'death': [.5, .25, np.inf, np.inf, .8]})
    result = counts_at_distance(intervals, trials, .25)
    assert result.betti_h1.tolist() == [2, 0, 0]
    assert result.direction.tolist() == [1, 4, 2]
    assert result.distance.tolist() == [.25, .25, .25]


def test_failed_topology_is_not_silently_made_zero():
    from plot_prego_betti_opposites import counts_at_distance
    trials = pd.DataFrame({'row_index': [0], 'direction': [1], 'status': ['failed']})
    with pytest.raises(ValueError, match='success'):
        counts_at_distance(pd.DataFrame(), trials, .25)


def test_saved_counts_agree_with_each_complete_diagram():
    from plot_prego_betti_opposites import ROOT, load_counts
    counts = load_counts(ROOT)
    assert len(counts) == 198
    assert counts.direction.value_counts().to_dict() == dict.fromkeys(range(1, 7), 33)
    for trial in counts.itertuples():
        with np.load(ROOT / 'diagrams' / f'trial_{trial.row_index:03d}.npz') as saved:
            expected = sum(b <= .25 < d for b, d in saved['H1'])
        assert trial.betti_h1 == expected
    assert counts.groupby('direction').betti_h1.sum().tolist() == [883, 792, 856, 864, 852, 829]


def test_requested_distance_updates_counts_labels_and_reference():
    from plot_prego_betti_opposites import ROOT, load_counts, opposite_panel, reference_panel
    counts = load_counts(ROOT, distance=.23)
    assert counts.distance.eq(.23).all() and len(counts) == 198
    for trial in counts.itertuples():
        with np.load(ROOT / 'diagrams' / f'trial_{trial.row_index:03d}.npz') as saved:
            expected = sum(b <= .23 < d for b, d in saved['H1'])
        assert trial.betti_h1 == expected
    fig, axes, _ = opposite_panel(counts)
    assert axes[0].get_ylabel().endswith('distance 0.23')
    plt.close(fig)
    summary = pd.read_csv(ROOT / 'tables/betti_summary.csv')
    fig = reference_panel(summary, distance=.23)
    np.testing.assert_allclose(fig.axes[0].lines[-1].get_xdata(), [.23, .23])
    assert fig.axes[0].texts[-1].get_text() == '0.23'
    plt.close(fig)


def test_plot_uses_true_opposites_all_trials_common_scale_and_horizontal_jitter_only():
    from plot_prego_betti_opposites import ROOT, load_counts, opposite_panel
    counts = load_counts(ROOT)
    fig, axes, plotted = opposite_panel(counts)
    assert [tuple(int(t.get_text()) for t in ax.get_xticklabels()) for ax in axes] == [
        (1, 4), (2, 5), (3, 6)]
    assert len({ax.get_ylim() for ax in axes}) == 1
    assert axes[0].get_ylim()[0] == 0
    assert len(plotted) == 198 and plotted.row_index.is_unique
    pd.testing.assert_series_equal(plotted.set_index('row_index').betti_h1.sort_index(),
                                   counts.set_index('row_index').betti_h1.sort_index())
    for ax, pair in zip(axes, [(1, 4), (2, 5), (3, 6)]):
        for scatter_index, direction in zip([0, 2], pair):
            offsets = ax.collections[scatter_index].get_offsets()
            expected = plotted[plotted.direction == direction]
            np.testing.assert_allclose(offsets, expected[['plot_x', 'betti_h1']])
            assert len(offsets) == 33
        assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    plt.close(fig)
