"""Plot saved H1 interval counts at a specified distance, without computing persistence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from plot_prego_persistent_homology import COLORS, ROOT, SOURCE, betti_panel, digest, style

DISTANCE = 0.25
PAIRS = ((1, 4), (2, 5), (3, 6))


def counts_at_distance(intervals, trials, distance):
    if not trials.status.eq('success').all():
        raise ValueError('All supplied topology trials must have success status.')
    if not trials.row_index.is_unique:
        raise ValueError('Trial row_index must be unique.')
    if not trials.direction.isin(range(1, 7)).all():
        raise ValueError('Unknown reach direction.')
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('Distance must be finite and nonnegative.')
    if not intervals.row_index.isin(trials.row_index).all():
        raise ValueError('Persistence intervals include unknown trials.')
    # Count exact half-open intervals; the plotting grid need not contain this distance.
    alive = intervals.loc[intervals.homology_dimension.eq(1)
                          & intervals.birth.le(distance) & intervals.death.gt(distance)]
    counts = alive.groupby('row_index').size()
    result = trials.copy().sort_values('row_index')
    result['betti_h1'] = result.row_index.map(counts).fillna(0).astype(int)
    result['distance'] = distance
    return result


def load_counts(root, distance=DISTANCE):
    intervals = pd.read_csv(root / 'tables/persistence_intervals.csv', float_precision='round_trip')
    trials = pd.read_csv(root / 'tables/trial_measurements.csv', float_precision='round_trip')
    counts = counts_at_distance(intervals, trials, distance)
    if len(counts) != 198 or counts.direction.value_counts().to_dict() != dict.fromkeys(range(1, 7), 33):
        raise ValueError('Expected the frozen 198-trial cohort, 33 per direction.')
    return counts


def opposite_panel(counts):
    style()
    if counts.distance.nunique() != 1:
        raise ValueError('Each comparison panel must use exactly one distance.')
    distance = float(counts.distance.iloc[0])
    plotted = counts.copy().sort_values('row_index')
    positions = {direction: x for pair in PAIRS for x, direction in enumerate(pair, 1)}
    pair_ids = {direction: f'{a}-{b}' for a, b in PAIRS for direction in (a, b)}
    plotted['opposite_pair'] = plotted.direction.map(pair_ids)
    plotted['plot_x'] = (plotted.direction.map(positions)
                         + np.random.default_rng(220922).uniform(-.20, .20, len(plotted)))
    fig, axes = plt.subplots(1, 3, figsize=(5.3, 2.8), sharey=True)
    fig.subplots_adjust(left=.13, right=.985, bottom=.23, top=.95, wspace=.24)
    upper = max(10, np.ceil((plotted.betti_h1.max() + 1) / 10) * 10)
    for ax, pair in zip(axes, PAIRS):
        for x, direction in enumerate(pair, 1):
            group = plotted[plotted.direction == direction]
            ax.scatter(group.plot_x, group.betti_h1, color=COLORS[direction-1],
                       s=15, alpha=.70, edgecolors='white', linewidths=.3, zorder=2)
            q1, median, q3 = np.quantile(group.betti_h1, [.25, .5, .75])
            ax.plot([x, x], [q1, q3], color='#252525', lw=2.5,
                    solid_capstyle='butt', zorder=3)
            ax.scatter([x], [median], s=24, facecolors='white', edgecolors='#252525',
                       linewidths=1., zorder=4)
        ax.set(xlim=(.5, 2.5), ylim=(0, upper), xticks=[1, 2], xticklabels=[str(d) for d in pair])
        ax.yaxis.set_major_locator(MultipleLocator(10))
        if ax is not axes[0]:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)
    axes[0].set_ylabel(rf'$\beta_1$ at distance {distance:g}')
    fig.text(.56, .065, 'Reach direction', ha='center', fontsize=9)
    return fig, axes, plotted


def reference_panel(summary, distance=DISTANCE):
    fig = betti_panel(summary, 1)
    ax = fig.axes[0]
    ax.axvline(distance, color='#555555', ls='--', lw=.9, zorder=5)
    ax.text(distance + .025, .97, f'{distance:g}', transform=ax.get_xaxis_transform(),
            va='top', ha='left', fontsize=8, color='#444444')
    return fig


def generate(root=ROOT, distance=DISTANCE):
    root = Path(root).resolve()
    distance = float(distance)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('Distance must be finite and nonnegative.')
    tag = str(distance).replace('.', '')
    output = root / f'opposite_directions_distance{tag}'
    protected_paths = [p for base in (root, SOURCE) for p in base.rglob('*')
                       if p.is_file() and output not in p.parents]
    protected = {str(p): digest(p) for p in protected_paths}
    counts = load_counts(root, distance)
    summary = pd.read_csv(root / 'tables/betti_summary.csv', float_precision='round_trip')
    for folder in ('figures', 'tables', 'captions', 'source'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    fig, _, plotted = opposite_panel(counts)
    reference = reference_panel(summary, distance)
    exports = {}
    for name, figure in [(f'betti_h1_opposite_directions_distance{tag}', fig),
                         (f'betti_h1_distance{tag}_reference', reference)]:
        target = output / 'figures' / f'{name}.png'
        figure.savefig(target, dpi=600, metadata={'Software': 'NeuralFieldManifold plot-only renderer'})
        exports[name] = {'sha256': digest(target), 'dpi': 600,
                         'inches': figure.get_size_inches().tolist()}
        plt.close(figure)
    columns = ['row_index', 'original_trial_number', 'direction', 'opposite_pair',
               'distance', 'betti_h1', 'plot_x']
    plotted[columns].to_csv(output / 'tables' / f'trial_counts_distance{tag}.csv', index=False)
    stats = counts.groupby('direction').betti_h1.agg(
        n='count', mean='mean', sd='std', median='median',
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    stats.to_csv(output / 'tables/direction_summary.csv')
    summary.loc[summary.homology_dimension.eq(1)].to_csv(
        output / 'tables/reference_curves.csv', index=False)
    common = ('Monkey T, y070316009-12, channel 7; 198 short-delay trials, 33 per direction. '
              'Final 500 ms before GO; observed 494-point clouds, m=3 and tau=3 ms. '
              'All trials included regardless of geometric-fit usability. ')
    caption = (common + f'H1 counts at exact Euclidean distance {distance:g}: birth <= {distance:g} < death, '
               'computed from saved intervals, not interpolated from the Betti-curve grid. '
               'Adjacent groups are spatially opposite targets: 1/4, 2/5, 3/6. '
               'Directions 1-6 are upper right, right, lower right, lower left, left, upper left. '
               'Every dot is one trial; jitter is horizontal only. White circles are medians; '
               'black segments are interquartile ranges, not confidence intervals. '
               'These are independent trial groups, not paired observations; no trial-connecting lines. '
               'The threshold was selected after viewing these data. This is an exploratory '
               'single-recording comparison, with no significance tests or claim of separate clusters. '
               'Distances use existing preprocessed signal units, not per-cloud RMS rescaling. '
               'No persistence calculation, geometric fit, or decoding was rerun.')
    (output / 'captions' / f'betti_h1_opposite_directions_distance{tag}.txt').write_text(caption + '\n')
    (output / 'captions' / f'betti_h1_distance{tag}_reference.txt').write_text(
        common + 'Original full-range H1 mean Betti curves and interquartile bands. '
        f'The dashed vertical line marks the exploratory distance threshold {distance:g}. '
        'Curves use the saved 512-point grid; trial counts in the accompanying panel '
        f'are evaluated at exact distance {distance:g} from complete saved intervals.\n')
    (output / 'README.md').write_text(
        f'# H1 counts at distance {distance:g}\n\n'
        'Exploratory comparison of opposite reach directions: 1/4, 2/5, 3/6. '
        'All 198 trials are shown, with median and interquartile range. '
        'The distance was chosen after inspecting the existing curves; no significance testing '
        'or clustering claim is made. PNGs are 600 dpi.\n\n'
        f'Counts are read from saved birth/death intervals using birth <= {distance:g} < death. '
        'No Ripser, fitter, or decoder is invoked. The companion curve panel marks the threshold. '
        'CSV files preserve all displayed values and horizontal jitter.\n\n'
        f'Regenerate: `{sys.executable} {Path(__file__).resolve()} --distance {distance}`\n')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError(f'Protected file changed: {path}')
    if 'ripser' in sys.modules:
        raise RuntimeError('Unexpected Ripser import in plot-only process.')
    shutil.copy2(__file__, output / 'source' / Path(__file__).name)
    (output / 'provenance.json').write_text(json.dumps({
        'distance': distance, 'opposite_pairs': PAIRS, 'n_trials': len(counts),
        'plot_only': True, 'threshold_selection': 'post_hoc_exploratory',
        'renderer_sha256': digest(__file__), 'outputs': exports,
        'protected_files_verified_unchanged': len(protected), 'protected_sha256': protected,
    }, indent=2) + '\n')
    print(stats.round(3).to_string())
    print(f'Saved two PNGs in {output / "figures"}; {len(protected)} protected files unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', type=float, default=DISTANCE)
    args = parser.parse_args()
    generate(distance=args.distance)
