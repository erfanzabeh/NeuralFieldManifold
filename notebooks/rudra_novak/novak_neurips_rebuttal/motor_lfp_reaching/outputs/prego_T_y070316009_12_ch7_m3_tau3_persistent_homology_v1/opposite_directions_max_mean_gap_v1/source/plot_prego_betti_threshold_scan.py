"""Exploratory exact-threshold comparison from frozen persistence intervals only."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from plot_prego_betti_opposites import PAIRS, counts_at_distance, opposite_panel
from plot_prego_persistent_homology import COLORS, ROOT, SOURCE, digest, plt, style


def scan_thresholds(intervals, trials):
    counts_at_distance(intervals, trials, 0.)
    group_sizes = trials.direction.value_counts().sort_index()
    if list(group_sizes.index) != list(range(1, 7)) or group_sizes.nunique() != 1:
        raise ValueError('Expected equal nonzero trial counts in all six directions.')
    h1 = intervals.loc[intervals.homology_dimension.eq(1)]
    birth, death = h1.birth.to_numpy(), h1.death.to_numpy()
    if (not np.isfinite(birth).all() or np.isnan(death).any()
            or np.any(birth < 0) or np.any(death < birth)):
        raise ValueError('Invalid H1 birth/death intervals.')
    finite = np.isfinite(death)
    grid = np.unique(np.r_[0., birth, death[finite]])
    rows = pd.Series(np.arange(len(trials)), index=trials.row_index)
    trial_index = h1.row_index.map(rows).to_numpy(dtype=int)
    # Evaluate every constant interval of the filtration, with deaths excluded at their boundary.
    values = np.zeros((len(grid), len(trials)), dtype=np.int32)
    np.add.at(values, (np.searchsorted(grid, birth), trial_index), 1)
    np.add.at(values, (np.searchsorted(grid, death[finite]), trial_index[finite]), -1)
    np.cumsum(values, axis=0, dtype=np.int32, out=values)
    sweep = pd.DataFrame({'distance': grid})
    curves = []
    totals = []
    for direction in range(1, 7):
        group = values[:, trials.direction.to_numpy() == direction]
        total = group.sum(axis=1)
        totals.append(total)
        sweep[f'mean_direction_{direction}'] = total / len(group[0])
        q25, median, q75 = np.quantile(group, [.25, .5, .75], axis=1)
        curves.append(pd.DataFrame({'distance': grid, 'direction': direction,
            'mean': total / len(group[0]), 'q25': q25, 'median': median, 'q75': q75}))
    score_numerator = np.zeros(len(grid), dtype=np.int64)
    for a, b in PAIRS:
        delta = np.abs(totals[a-1] - totals[b-1])
        sweep[f'absolute_gap_{a}_{b}'] = delta / group_sizes.iloc[0]
        score_numerator += delta
    sweep['sum_absolute_mean_difference'] = score_numerator / group_sizes.iloc[0]
    # Equal cohort sizes permit exact integer tie comparisons before dividing by n.
    winners = np.flatnonzero(score_numerator == score_numerator.max())
    chosen = int(winners[0])
    ends = np.r_[grid[1:], np.inf]
    selection = {
        'distance': float(grid[chosen]),
        'objective': 'abs(mean_1-mean_4)+abs(mean_2-mean_5)+abs(mean_3-mean_6)',
        'score': float(sweep.sum_absolute_mean_difference.iloc[chosen]),
        'n_thresholds': len(grid), 'selection': 'post_hoc_exploratory',
        'search_domain': 'All nonnegative distances; changes at every saved H1 birth/death event.',
        'tie_rule': 'Smallest distance attaining the exact maximum.',
        'maximizing_intervals': [
            {'start_inclusive': float(grid[i]),
             'end_exclusive': float(ends[i]) if np.isfinite(ends[i]) else None}
            for i in winners],
    }
    return sweep, pd.concat(curves, ignore_index=True), selection


def marked_curves(curves, selected, zoom=False):
    style()
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    fig.subplots_adjust(left=.13, right=.955, bottom=.18, top=.80)
    for direction, color in enumerate(COLORS, 1):
        group = curves.loc[curves.direction.eq(direction)]
        ax.fill_between(group.distance, group.q25, group.q75, step='post',
                        color=color, alpha=.08, lw=0)
        ax.step(group.distance, group['mean'], where='post', color=color, lw=1.)
    ax.axvline(.23, color='#777777', ls='--', lw=1.)
    ax.axvline(selected, color='#202020', ls=':', lw=1.25)
    ax.set(xlabel='Distance threshold', ylabel=r'$\beta_1$',
           xlim=(.16, .30) if zoom else (0, curves.distance.max()),
           ylim=(0, np.ceil(curves.q75.max()/10)*10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    if zoom:
        ax.xaxis.set_major_locator(MultipleLocator(.02))
    fig.legend(handles=[Line2D([], [], color=c, lw=1.5, label=str(i))
                        for i, c in enumerate(COLORS, 1)],
               loc='upper center', bbox_to_anchor=(.55, .99), ncol=6, frameon=False,
               handlelength=1.5, title='Reach direction', title_fontsize=8.5)
    ax.legend(handles=[Line2D([], [], color='#777777', ls='--', lw=1., label='Reference: 0.23'),
                       Line2D([], [], color='#202020', ls=':', lw=1.25,
                              label=f'Maximum: {selected:.6f}')],
              loc='upper right', frameon=False, fontsize=8)
    return fig


def generate():
    output = ROOT / 'opposite_directions_max_mean_gap_v1'
    protected = {str(p): digest(p) for base in (ROOT, SOURCE) for p in base.rglob('*')
                 if p.is_file() and output not in p.parents}
    intervals = pd.read_csv(ROOT / 'tables/persistence_intervals.csv', float_precision='round_trip')
    trials = pd.read_csv(ROOT / 'tables/trial_measurements.csv', float_precision='round_trip')
    if len(trials) != 198 or trials.direction.value_counts().to_dict() != dict.fromkeys(range(1, 7), 33):
        raise ValueError('Frozen 198-trial cohort does not match.')
    sweep, curves, selection = scan_thresholds(intervals, trials)
    selected = selection['distance']
    counts = counts_at_distance(intervals, trials, selected)
    for trial in counts.itertuples():
        with np.load(ROOT / 'diagrams' / f'trial_{trial.row_index:03d}.npz') as saved:
            if sum(b <= selected < d for b, d in saved['H1']) != trial.betti_h1:
                raise ValueError(f'Count disagrees with saved diagram: {trial.row_index}')
    for directory in ('figures', 'tables', 'captions', 'source'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output / 'tables/threshold_scan.csv', index=False)
    # The stored full event-wise curves permit regeneration without threshold selection.
    curves.to_csv(output / 'tables/exact_betti_curves.csv.gz', index=False, compression='gzip')
    stats = counts.groupby('direction').betti_h1.agg(
        n='count', mean='mean', sd='std', median='median',
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    stats.to_csv(output / 'tables/direction_summary.csv')
    comparison, axes, plotted = opposite_panel(counts)
    for ax in axes:
        ax.set_ylim(0, max(60, ax.get_ylim()[1]))
    plotted[['row_index', 'original_trial_number', 'direction', 'opposite_pair',
             'distance', 'betti_h1', 'plot_x']].to_csv(output / 'tables/trial_counts.csv', index=False)
    exports = {}
    for name, fig in [('betti_h1_max_gap_comparison', comparison),
                      ('betti_h1_thresholds_overview', marked_curves(curves, selected)),
                      ('betti_h1_thresholds_zoom', marked_curves(curves, selected, zoom=True))]:
        target = output / 'figures' / f'{name}.png'
        fig.savefig(target, dpi=600, metadata={'Software': 'NeuralFieldManifold saved-interval renderer'})
        exports[name] = {'sha256': digest(target), 'dpi': 600, 'inches': fig.get_size_inches().tolist()}
        plt.close(fig)
    at_reference = counts_at_distance(intervals, trials, .23).groupby('direction').betti_h1.mean()
    selection['reference_distance'] = .23
    selection['reference_score'] = float(sum(abs(at_reference[a]-at_reference[b]) for a, b in PAIRS))
    selection['n_trials'] = len(trials)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    common = ('Monkey T, y070316009-12, channel 7; 198 trials, 33 per direction. '
              'Short-delay, final 500 ms before GO; fixed m=3, tau=3 ms, 494 observed points. '
              'Saved H1 intervals only, with birth <= distance < death. '
              'Threshold chosen post hoc to maximize the sum of absolute differences in mean '
              'H1 counts for direction pairs 1/4, 2/5, 3/6. '
              f'Exact selected distance: {selected!r}; score: {selection["score"]:.9f}. '
              'All H1 birth/death thresholds were searched; ties favor the smallest distance. '
              'This is a selected descriptive contrast, not evidence of validated separation. '
              'No significance tests, Ripser calculations, fitting, or decoding were performed. ')
    (output / 'captions/betti_h1_max_gap_comparison.txt').write_text(
        common + 'Dots: all 198 trials; white circles: medians; black lines: interquartile ranges. '
        'The selection criterion uses means, whereas the plot summaries show medians/IQR, '
        'unchanged from the earlier panels. Horizontal jitter only; shared zero-based axes.\n')
    for suffix, extent in [('overview', 'Full distance range.'), ('zoom', 'Zoomed x-axis: 0.16-0.30.')]:
        (output / 'captions' / f'betti_h1_thresholds_{suffix}.txt').write_text(
            common + 'Colored steps: exact event-wise direction means; shaded bands: interquartile ranges, '
            'not confidence intervals. Dashed gray line: 0.23; dotted black line: selected maximum. '
            'The interval summaries are evaluated at all events, not interpolated from the old '
            '512-point plotting grid. ' + extent + '\n')
    (output / 'README.md').write_text(
        '# Exploratory opposite-direction threshold scan\n\n'
        f'Maximum at distance **{selected:.9f}**. Sum of absolute mean differences: '
        f'**{selection["score"]:.3f}**, versus **{selection["reference_score"]:.3f}** at 0.23.\n\n'
        'The exact maximum occupies very narrow distance intervals (see selection.json). '
        'It is a data-selected contrast, not a robust optimal scale or a significance result. '
        'All trials are displayed; neither threshold nor results were validated independently. '
        'Mean differences determine the maximum, while white plot markers retain median/IQR summaries.\n\n'
        'Three 600-dpi PNGs: full Betti curves, zoomed curves, and opposite-direction trial distributions. '
        'Both curve views mark 0.23 and the selected distance. Existing results remain unchanged.\n')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError(f'Protected file changed: {path}')
    if 'ripser' in sys.modules:
        raise RuntimeError('Unexpected Ripser import.')
    shutil.copy2(__file__, output / 'source' / Path(__file__).name)
    (output / 'provenance.json').write_text(json.dumps({
        'renderer_sha256': digest(__file__), 'saved_interval_postprocessing_only': True,
        'protected_files_verified_unchanged': len(protected), 'protected_sha256': protected,
        'outputs': exports}, indent=2) + '\n')
    print(json.dumps(selection, indent=2))
    print(stats.round(3).to_string())
    print(f'Saved three PNGs in {output / "figures"}; {len(protected)} protected files unchanged.')


if __name__ == '__main__':
    generate()
