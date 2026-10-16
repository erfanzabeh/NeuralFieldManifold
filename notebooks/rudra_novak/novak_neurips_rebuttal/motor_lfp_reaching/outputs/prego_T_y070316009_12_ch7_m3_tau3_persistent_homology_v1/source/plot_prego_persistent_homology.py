"""PNG-only regeneration from saved PH tables, observed clouds and existing fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

UNIT = Path(__file__).resolve().parent
ROOT = UNIT / 'outputs/prego_T_y070316009_12_ch7_m3_tau3_persistent_homology_v1'
SOURCE = UNIT / 'outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1'
COLORS = ['#0072b2', '#d55e00', '#009e73', '#cc79a7', '#e69f00', '#56b4e9']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'xtick.major.width': .7, 'ytick.major.width': .7})


def distribution_panel(table, column, label):
    style()
    table = table.loc[np.isfinite(table[column])].copy().sort_values('row_index')
    rng = np.random.default_rng(220922)
    table['plot_x'] = table.direction + rng.uniform(-.19, .19, len(table))
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.subplots_adjust(left=.21, right=.985, bottom=.19, top=.97)
    for direction, color in enumerate(COLORS, 1):
        subset = table[table.direction == direction]
        values = subset[column].to_numpy()
        if not len(values):
            continue
        ax.scatter(subset.plot_x, values, s=13, color=color, alpha=.60,
                   edgecolors='white', linewidths=.25, zorder=2)
        q1, median, q3 = np.quantile(values, [.25, .5, .75])
        ax.plot([direction]*2, [q1, q3], color='#252525', lw=2.5, solid_capstyle='butt', zorder=4)
        ax.scatter([direction], [median], s=19, facecolors='white', edgecolors='#252525', linewidths=.9, zorder=5)
    ax.set(xlim=(.5, 6.5), xticks=range(1, 7), xlabel='Reach direction', ylabel=label)
    ax.set_ylim(bottom=0, top=max(float(table[column].max())*1.10, .01))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    return fig, ax, table


def betti_panel(summary, h):
    style()
    table = summary[summary.homology_dimension == h]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.subplots_adjust(left=.18, right=.985, bottom=.19, top=.83)
    for direction, color in enumerate(COLORS, 1):
        group = table[table.direction == direction].sort_values('grid_index')
        ax.fill_between(group.distance, group.q25, group.q75, color=color, alpha=.10, lw=0)
        ax.plot(group.distance, group['mean'], color=color, lw=1.15, label=str(direction))
    ax.set(xlabel='Distance threshold', ylabel=rf'$\beta_{h}$',
           xlim=(0, table.distance.max()))
    upper = max(table['mean'].max(), table.q75.max())
    ax.set_ylim(0, max(.05, upper*1.08))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.legend(handles=[Line2D([], [], color=c, label=str(i), lw=1.3) for i, c in enumerate(COLORS, 1)],
               loc='upper center', bbox_to_anchor=(.57, .99), ncol=6, frameon=False,
               handlelength=1., columnspacing=.85, handletextpad=.4, title='Reach direction', title_fontsize=8)
    return fig


def outlines(fit):
    theta = np.linspace(0, 2*np.pi, 361)
    c, u, v = (np.asarray(fit[k]) for k in ('center', 'u_axis', 'v_axis'))
    return tuple(c + fit[a]*np.cos(theta)[:, None]*u + fit[b]*np.sin(theta)[:, None]*v
                 for a, b in [('R1', 'R2'), ('R1_in', 'R2_in')])


def validate_examples(chosen, trials):
    expected = trials.loc[trials.groupby('direction').original_trial_number.idxmin()].reset_index(drop=True)
    pd.testing.assert_frame_equal(chosen[expected.columns].reset_index(drop=True), expected, check_dtype=False)


def geometry_panel(cloud, outline, direction, limits):
    style()
    fig = plt.figure(figsize=(3.6, 3.25))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.subplots_adjust(left=.03, right=.88, bottom=.09, top=.90)
    ax.scatter(*cloud.T, s=2.8, color=COLORS[direction-1], alpha=.65, depthshade=False,
               edgecolors='none', zorder=2)
    if outline is not None:
        for line in outline:
            ax.plot(*line.T, color='#333333', lw=.9, alpha=.95, zorder=3)
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
        axis.set_major_locator(MaxNLocator(3))
    ax.set(xlim=limits, ylim=limits, zlim=limits,
           xlabel=r'$x(t)$', ylabel=r'$x(t-3\,\mathrm{ms})$', zlabel=r'$x(t-6\,\mathrm{ms})$')
    ax.tick_params(pad=0, labelsize=8)
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3
    ax.zaxis.labelpad = 3
    ax.grid(False)
    fig.text(.5, .95, f'Direction {direction}', ha='center', va='top', fontsize=9)
    return fig


def persistence_panel(table, direction, extent):
    style()
    fig, ax = plt.subplots(figsize=(3.3, 3.25))
    fig.subplots_adjust(left=.20, right=.965, bottom=.17, top=.82)
    ax.plot([0, extent], [0, extent], color='#aaaaaa', ls='--', lw=.7, zorder=0)
    infinite_y = extent*1.09
    ax.axhline(infinite_y, color='#bcbcbc', lw=.6, ls=':')
    styles = [('#999999', 'o', 9), (COLORS[direction-1], 'o', 18), ('#252525', 'D', 15)]
    for h, (color, marker, size) in enumerate(styles):
        group = table[table.homology_dimension == h]
        finite = group[np.isfinite(group.death)]
        ax.scatter(finite.birth, finite.death, s=size, c=color, marker=marker,
                   alpha=.75, linewidths=0, clip_on=False)
        infinite = group[np.isinf(group.death)]
        if len(infinite):
            ax.scatter(infinite.birth, np.full(len(infinite), infinite_y), s=20,
                       c=color, marker='^', linewidths=0, clip_on=False)
    ticks = np.linspace(0, extent, 4)
    ax.set(xlim=(-.025*extent, extent*1.025), ylim=(-.025*extent, extent*1.16),
           xlabel='Birth distance', ylabel='Death distance', xticks=ticks)
    ax.set_yticks([*ticks, infinite_y], [*(f'{v:.2g}' for v in ticks), r'$\infty$'])
    ax.set_xticklabels([f'{v:.2g}' for v in ticks])
    fig.legend(handles=[Line2D([], [], color=c, marker=m, lw=0, markersize=4, label=f'H{h}')
                        for h, (c, m, _) in enumerate(styles)], loc='upper center',
               bbox_to_anchor=(.57, .99), ncol=3, frameon=False, handletextpad=.3,
               title=f'Direction {direction}', title_fontsize=9)
    return fig


def save_panel(name, fig, data, caption, manifest):
    target = ROOT / 'figures' / f'{name}.png'
    fig.savefig(target, dpi=600, metadata={'Software': 'NeuralFieldManifold PH plot-only renderer'})
    data.to_csv(ROOT / 'tables' / f'plot_{name}.csv', index=False)
    (ROOT / 'captions' / f'{name}.txt').write_text(caption.strip() + '\n')
    manifest[name] = dict(sha256=digest(target), inches=fig.get_size_inches().tolist(), dpi=600,
                          data_sha256=digest(ROOT / 'tables' / f'plot_{name}.csv'))
    plt.close(fig)


def generate():
    for folder in ('figures', 'captions'):
        (ROOT / folder).mkdir(exist_ok=True)
    paths = [SOURCE/'clouds.npz', SOURCE/'tables/fit_diagnostics.csv']
    paths += list((ROOT/'diagrams').glob('*.npz'))
    paths += [ROOT/'tables'/f'{n}.csv' for n in ('trial_measurements', 'betti_summary',
                                              'example_trials', 'persistence_intervals')]
    before = {str(p): digest(p) for p in paths}
    measures = pd.read_csv(ROOT/'tables/trial_measurements.csv')
    betti = pd.read_csv(ROOT/'tables/betti_summary.csv')
    intervals = pd.read_csv(ROOT/'tables/persistence_intervals.csv')
    examples = pd.read_csv(ROOT/'tables/example_trials.csv')
    validate_examples(examples, pd.read_csv(SOURCE/'inputs/selected_trials.csv'))
    assert examples.row_index.tolist() == json.loads((ROOT/'config.json').read_text())['example_rows']
    clouds = np.load(SOURCE/'clouds.npz', allow_pickle=False)['clouds']
    accounting = f'{(measures.status == "success").sum()} successful topology clouds; 197 usable geometric fits.'
    common = ('Monkey T, y070316009-12, channel 7; 198 short-delay trials, 33 per direction. '
              'Final 500 ms before GO, m=3, tau=3 ms, 494 unchanged observed points per trial. '
              'Ripser 0.6.15, Euclidean full filtration through H2, coefficients in F2, all points. ')
    manifest = {}
    for name, column, label in [
        ('loop_lifetime', 'longest_H1_lifetime', 'Longest H1 lifetime'),
        ('loop_lifetime_normalized', 'normalized_H1_lifetime', 'Longest H1 lifetime / RMS radius'),
        ('radius_r1', 'R1', 'R1 (normalized signal units)'),
        ('radius_r2', 'R2', 'R2 (normalized signal units)')]:
        radius = column in ('R1', 'R2')
        subset = measures[measures.geometry_usable] if radius else measures[measures.status == 'success']
        fig, _, plotted = distribution_panel(subset, column, label)
        definition = ('Saved canonical outer ellipse radius; all fit-valid trials included. '
                      'Original trial 128 is excluded from radii only, not from PH. ' if radius else
                      'Longest finite H1 interval, zero only for successful diagrams with no H1 intervals. ')
        if column == 'normalized_H1_lifetime':
            definition += 'Division by centered cloud RMS radius occurs after persistence; clouds were not rescaled. '
        save_panel(name, fig, plotted, common+definition+
                   'All trial values are shown with display-only horizontal jitter. White circles: medians; '
                   'black segments: interquartile ranges, not confidence intervals. '+accounting, manifest)
    for h in range(3):
        fig = betti_panel(betti, h)
        save_panel(f'betti_h{h}', fig, betti[betti.homology_dimension == h], common+
            f'H{h} Betti curves count intervals with birth <= distance < death, including infinite bars. '
            'Lines show direction-wise trial means; bands show the interquartile range, not uncertainty of the mean. '
            'All directions share a 512-point distance grid; dimensions have independent distance extents. '
            'These are scale-dependent counts, not thresholded topological labels. '+accounting, manifest)
    geometry = {}
    all_values = []
    for row in examples.row_index:
        fit_path = SOURCE/'checkpoints'/f'trial_{row:03d}_one_torus.json'
        before[str(fit_path)] = digest(fit_path)
        saved = json.loads(fit_path.read_text())
        outline = outlines(saved['fit']) if measures.iloc[row].geometry_usable else None
        geometry[row] = outline
        all_values.append(clouds[row].ravel())
        if outline is not None:
            all_values.extend(line.ravel() for line in outline)
    values = np.concatenate(all_values)
    lo, hi = float(values.min()), float(values.max())
    pad = (hi-lo)*.06
    limits = (lo-pad, hi+pad)
    sample_intervals = intervals[intervals.row_index.isin(examples.row_index)]
    extent = float(sample_intervals.loc[np.isfinite(sample_intervals.death), 'death'].max())*1.04
    for trial in examples.itertuples(index=False):
        row, direction = int(trial.row_index), int(trial.direction)
        fig = geometry_panel(clouds[row], geometry[row], direction, limits)
        table = pd.DataFrame(clouds[row], columns=['x_t', 'x_t_minus3ms', 'x_t_minus6ms'])
        table.insert(0, 'latest_sample_index', np.arange(6, 500))
        caption = (common+f'Original trial {trial.original_trial_number}, direction {direction}; '
            'lowest original trial ID within this direction, selected without fit scores or topology. '
            'Dots show the complete observed cloud; black inner/outer ellipse outlines reuse the existing '
            '1-torus annular-band fit. All six examples use identical coordinate limits and viewpoints. '
            'The outline is a geometric model, not a persistence constraint; no refitting was performed.')
        save_panel(f'geometry_direction_{direction}', fig, table, caption, manifest)
        table = intervals[intervals.row_index == row]
        fig = persistence_panel(table, direction, extent)
        save_panel(f'persistence_direction_{direction}', fig, table,
            common+f'Original trial {trial.original_trial_number}, direction {direction}. '
            'All persistence intervals are displayed; the triangle on the infinity line denotes an essential interval. '
            'All six diagrams use identical axes. Coincident points may overlap. No lifetime cutoff or null '
            'significance classification is applied.', manifest)
    for path, checksum in before.items():
        assert digest(path) == checksum, path
    protected = json.loads((ROOT/'protected_manifest.json').read_text())
    for path, checksum in protected.items():
        assert digest(path) == checksum, path
    shutil.copy2(__file__, ROOT/'source'/Path(__file__).name)
    (ROOT/'plot_manifest.json').write_text(json.dumps(dict(
        renderer_sha256=digest(__file__), matplotlib_version=matplotlib.__version__,
        input_hashes=before, outputs=manifest, geometry_limits=limits, persistence_extent=extent,
        protected_files_unchanged=len(protected), plot_only=True), indent=2)+'\n')
    print(f'Saved {len(manifest)} PNG panels in {ROOT}/figures', flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    generate()
