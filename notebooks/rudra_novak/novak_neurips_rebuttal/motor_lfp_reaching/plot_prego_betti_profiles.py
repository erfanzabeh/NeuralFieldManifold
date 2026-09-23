"""Direction-wise Betti deviations and ranked H1 lifetimes from saved results."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_prego_persistent_homology import COLORS, ROOT, SOURCE, digest, plt, style


def deviation_table(curves):
    h1 = curves.loc[curves.homology_dimension.eq(1)].copy()
    if h1.duplicated(['row_index', 'grid_index']).any():
        raise ValueError('Duplicate trial/grid observations.')
    per_grid = h1.groupby('grid_index').size()
    if per_grid.empty or per_grid.nunique() != 1:
        raise ValueError('Incomplete common trial grid.')
    if not h1.groupby('grid_index').distance.nunique().eq(1).all():
        raise ValueError('Distances disagree across trials.')
    if not h1.groupby('row_index').direction.nunique().eq(1).all():
        raise ValueError('Trial direction changes within the grid.')
    group = h1.groupby(['direction', 'grid_index'], as_index=False).agg(
        distance=('distance', 'first'), n_trials=('betti', 'size'),
        mean_betti=('betti', 'mean'))
    pooled = h1.groupby('grid_index').betti.mean()
    group['pooled_mean_betti'] = group.grid_index.map(pooled)
    group['delta_betti'] = group.mean_betti - group.pooled_mean_betti
    return group.sort_values(['direction', 'grid_index']).reset_index(drop=True)


def ranked_lifetimes(intervals, trials):
    if not trials.status.eq('success').all() or not trials.row_index.is_unique:
        raise ValueError('Expected unique successful topology trials.')
    h1 = intervals.loc[intervals.homology_dimension.eq(1)].copy()
    if not h1.row_index.isin(trials.row_index).all():
        raise ValueError('Unknown trial in persistence intervals.')
    if (not np.isfinite(h1.birth).all() or np.isnan(h1.death).any()
            or (h1.death < h1.birth).any()):
        raise ValueError('Invalid H1 interval.')
    if np.isinf(h1.death).any():
        raise ValueError('Infinite H1 intervals require explicit handling, not a finite-rank plot.')
    h1['duration'] = h1.death - h1.birth
    ordered = h1.sort_values(['row_index', 'duration'], ascending=[True, False])
    ordered['rank'] = ordered.groupby('row_index').cumcount() + 1
    result = trials[['row_index', 'original_trial_number', 'direction']].copy()
    result['h1_interval_count'] = result.row_index.map(h1.groupby('row_index').size()).fillna(0).astype(int)
    for rank, name in [(1, 'longest_h1'), (2, 'second_longest_h1')]:
        values = ordered.loc[ordered['rank'].eq(rank)].set_index('row_index').duration
        result[name] = result.row_index.map(values).fillna(0.)
    result['lifetime_gap'] = result.longest_h1 - result.second_longest_h1
    return result.sort_values('row_index').reset_index(drop=True)


def heatmap_panel(table, zoom=False):
    style()
    matrix = table.pivot(index='direction', columns='distance', values='delta_betti').sort_index()
    limit = max(1., float(np.ceil(np.abs(matrix.to_numpy()).max())))
    fig, ax = plt.subplots(figsize=(5.3, 2.9))
    fig.subplots_adjust(left=.13, right=.80, bottom=.22, top=.94)
    mesh = ax.pcolormesh(matrix.columns.to_numpy(), matrix.index.to_numpy(), matrix.to_numpy(),
                         shading='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit,
                         edgecolors='none', antialiased=False)
    ax.set(xlabel='Distance threshold', ylabel='Reach direction', yticks=range(1, 7),
           xlim=(0, 1.) if zoom else (0, matrix.columns.max()), ylim=(6.5, .5))
    ax.tick_params(axis='y', length=0)
    color_ax = fig.add_axes([.835, .22, .025, .72])
    bar = fig.colorbar(mesh, cax=color_ax, ticks=np.linspace(-limit, limit, 5))
    bar.set_label(r'$\Delta\beta_1$ (mean count)', fontsize=9, labelpad=7)
    bar.outline.set_linewidth(.6)
    return fig, ax, mesh


def lifetime_panel(table):
    style()
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    fig.subplots_adjust(left=.18, right=.96, bottom=.17, top=.82)
    upper = float(table.longest_h1.max()) * 1.07
    ax.plot([0, upper], [0, upper], color='#777777', lw=.85, ls='--', zorder=0)
    for direction, color in enumerate(COLORS, 1):
        group = table.loc[table.direction.eq(direction)]
        ax.scatter(group.second_longest_h1, group.longest_h1, color=color, s=18,
                   alpha=.70, edgecolors='white', linewidths=.3, zorder=2)
    ax.set(xlabel='Second-longest H1 lifetime', ylabel='Longest H1 lifetime',
           xlim=(0, upper), ylim=(0, upper))
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.legend(handles=[Line2D([], [], color=c, marker='o', lw=0, markersize=4, label=str(i))
                        for i, c in enumerate(COLORS, 1)],
               loc='upper center', bbox_to_anchor=(.54, .985), ncol=6,
               frameon=False, handletextpad=.2, columnspacing=.65,
               title='Reach direction', title_fontsize=8.5)
    return fig, ax


def generate():
    output = ROOT / 'betti_profiles_and_loop_ranks_v1'
    protected = {str(p): digest(p) for base in (ROOT, SOURCE) for p in base.rglob('*')
                 if p.is_file() and output not in p.parents}
    curves = pd.read_csv(ROOT / 'tables/betti_curves.csv', float_precision='round_trip')
    intervals = pd.read_csv(ROOT / 'tables/persistence_intervals.csv', float_precision='round_trip')
    trials = pd.read_csv(ROOT / 'tables/trial_measurements.csv', float_precision='round_trip')
    if len(trials) != 198 or trials.direction.value_counts().to_dict() != dict.fromkeys(range(1, 7), 33):
        raise ValueError('Expected 198 frozen trials, 33 per direction.')
    deviations = deviation_table(curves)
    ranks = ranked_lifetimes(intervals, trials)
    if len(deviations) != 6*512 or not deviations.n_trials.eq(33).all():
        raise ValueError('Expected a complete six-direction, 512-point curve grid.')
    saved_summary = pd.read_csv(ROOT / 'tables/betti_summary.csv', float_precision='round_trip')
    reference = saved_summary.loc[saved_summary.homology_dimension.eq(1)].sort_values(['direction', 'grid_index'])
    np.testing.assert_allclose(deviations.mean_betti, reference['mean'], rtol=1e-12)
    np.testing.assert_allclose(deviations.groupby('grid_index').delta_betti.mean(), 0, atol=1e-12)
    for row in ranks.itertuples():
        with np.load(ROOT / 'diagrams' / f'trial_{row.row_index:03d}.npz') as saved:
            actual = sorted((float(d-b) for b, d in saved['H1']), reverse=True)
        np.testing.assert_allclose([row.longest_h1, row.second_longest_h1], actual[:2], rtol=1e-12)
    np.testing.assert_allclose(ranks.longest_h1, trials.sort_values('row_index').longest_H1_lifetime, rtol=1e-12)
    for folder in ('figures', 'tables', 'captions', 'source'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    deviations.to_csv(output / 'tables/betti_direction_deviations.csv', index=False)
    ranks.to_csv(output / 'tables/ranked_h1_lifetimes.csv', index=False)
    summary = ranks.groupby('direction').agg(
        n=('row_index', 'size'), longest_median=('longest_h1', 'median'),
        second_longest_median=('second_longest_h1', 'median'),
        gap_median=('lifetime_gap', 'median'), gap_q25=('lifetime_gap', lambda x: x.quantile(.25)),
        gap_q75=('lifetime_gap', lambda x: x.quantile(.75)))
    summary.to_csv(output / 'tables/lifetime_direction_summary.csv')
    exports = {}
    for name, fig in [
        ('betti_h1_direction_deviation', heatmap_panel(deviations)[0]),
        ('betti_h1_direction_deviation_zoom', heatmap_panel(deviations, zoom=True)[0]),
        ('h1_longest_vs_second_longest', lifetime_panel(ranks)[0])]:
        target = output / 'figures' / f'{name}.png'
        fig.savefig(target, dpi=600, metadata={'Software': 'NeuralFieldManifold saved-results renderer'})
        exports[name] = {'sha256': digest(target), 'inches': fig.get_size_inches().tolist(), 'dpi': 600}
        plt.close(fig)
    shutil.copy2(ROOT / 'figures/betti_h1.png', output / 'figures/betti_h1_raw_reference.png')
    common = ('Monkey T, y070316009-12, channel 7; 198 short-delay trials, 33 per direction. '
              'Final 500 ms before GO, m=3, tau=3 ms, 494 observed points per trial. '
              'Saved results only; no persistence calculation, fitting, or decoding rerun. '
              'All topology trials included regardless of geometric-fit success. ')
    heatmap_caption = (common + 'Each cell is the direction mean H1 Betti count minus the pooled '
        '198-trial mean at the same distance. The pooled mean is trial-weighted, not row-standardized. '
        'Red indicates more loops than the pooled mean; blue indicates fewer. '
        'A shared symmetric color scale uses actual count differences. Values use the original '
        '512-point H1 distance grid without smoothing or separate direction normalization. '
        'This mean-contrast display does not show trial variability or establish significance. '
        'Refer to the unchanged raw mean/IQR curve panel and earlier trial distributions. ')
    (output / 'captions/betti_h1_direction_deviation.txt').write_text(heatmap_caption + 'Full distance range.\n')
    (output / 'captions/betti_h1_direction_deviation_zoom.txt').write_text(
        heatmap_caption + 'Display-only x-axis zoom to distances 0-1, with the same color limits.\n')
    (output / 'captions/h1_longest_vs_second_longest.txt').write_text(
        common + 'Each point is one trial: x is the second-longest finite H1 lifetime and y is the '
        'longest, each defined as death distance minus birth distance. Colors denote direction. '
        'No jitter, fitted trend, lifetime threshold, or trial exclusion is used. '
        'Both axes have equal limits and equal aspect; the dashed line is equality. '
        'Points must lie on or above equality because lifetimes were ranked. '
        'Their position above this line alone is not evidence for significant topology or one torus. '
        'All 198 trials have at least two finite H1 intervals, with no infinite H1 interval. '
        'These are distance-scale lifetimes, not temporal durations; no RMS rescaling was applied.\n')
    shutil.copy2(ROOT / 'captions/betti_h1.txt', output / 'captions/betti_h1_raw_reference.txt')
    report = {
        'n_trials': len(ranks), 'n_trials_per_direction': 33,
        'longest_median': float(ranks.longest_h1.median()),
        'second_longest_median': float(ranks.second_longest_h1.median()),
        'paired_gap_median': float(ranks.lifetime_gap.median()),
        'paired_gap_q25': float(ranks.lifetime_gap.quantile(.25)),
        'paired_gap_q75': float(ranks.lifetime_gap.quantile(.75)),
        'minimum_h1_interval_count': int(ranks.h1_interval_count.min()),
        'delta_betti_min': float(deviations.delta_betti.min()),
        'delta_betti_max': float(deviations.delta_betti.max()),
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Betti profiles and ranked H1 lifetimes\n\n'
        'Direction-deviation heatmaps show the difference from the pooled mean, not a new topology '
        'classification. The full-range and 0-1 zoom panels use the same symmetric count scale. '
        'The original raw mean/IQR panel is included unchanged to retain variability context.\n\n'
        f'All 198 trials are included. Median longest H1 lifetime: {report["longest_median"]:.3f}; '
        f'median second-longest: {report["second_longest_median"]:.3f}. '
        f'The median within-trial gap is {report["paired_gap_median"]:.3f}. '
        'The equality-line ordering is guaranteed by ranking; it is not evidence of significant '
        'loops, a single underlying torus, or direction decoding. No surrogate calibration '
        'or significance testing was performed.\n\n'
        'PNG exports are 600 dpi. Underlying values, captions, and provenance are saved alongside them. '
        'Earlier analyses and images were not modified.\n')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError(f'Protected file changed: {path}')
    if 'ripser' in sys.modules:
        raise RuntimeError('Unexpected Ripser import in plot-only renderer.')
    shutil.copy2(__file__, output / 'source' / Path(__file__).name)
    (output / 'provenance.json').write_text(json.dumps({
        'renderer_sha256': digest(__file__), 'plot_only': True,
        'protected_files_verified_unchanged': len(protected), 'protected_sha256': protected,
        'outputs': exports,
        'raw_reference_sha256': digest(output / 'figures/betti_h1_raw_reference.png'),
    }, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(summary.round(4).to_string())
    print(f'Saved PNGs in {output / "figures"}; {len(protected)} protected files unchanged.')


if __name__ == '__main__':
    generate()
