"""Plot frozen R1/R2 pairs by documented physical reach side; no new fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from plot_prego_fixed_geometry_decoding import style, save_panel, digest
from plot_prego_fixed_geometry_additive import OUTPUT
from plot_prego_radius_pair import SOURCE, valid_pairs

STEM = 'geometry_r1_r2_left_right'
COLORS = {'Leftward': '#0072b2', 'Rightward': '#d55e00'}
POSITIONS = {1: 'upper right', 2: 'right', 3: 'lower right',
             4: 'lower left', 5: 'left', 6: 'upper left'}
SIDES = {1: 'Rightward', 2: 'Rightward', 3: 'Rightward',
         4: 'Leftward', 5: 'Leftward', 6: 'Leftward'}
REPO_URL = 'https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior'
SOURCES = {'direction_order': REPO_URL+'/raw/master/README.md',
           'target_layout': REPO_URL+'/raw/master/Setup%26Task.pdf'}
CONVERTED = SOURCE.parent/('recording_198_trial_dossiers_v1/'
                          'monkeyT_session-y070316009-12_lfp-7/inputs/full_converted_recording.npz')


def group_directions(table):
    if not table.direction.isin(SIDES).all():
        raise ValueError('Unknown direction; expected integer codes 1 through 6')
    result = table.copy()
    result['target_event_code'] = result.direction.astype(int)+100
    result['target_position'] = result.direction.map(POSITIONS)
    result['reach_side'] = result.direction.map(SIDES)
    return result


def load_grouped_trials():
    table = group_directions(pd.read_csv(SOURCE/'tables/features_geometry.csv'))
    pairs = valid_pairs(table)
    if len(table) != 198 or len(pairs) != 197:
        raise ValueError('Unexpected frozen trial accounting')
    diagnostics = pd.read_csv(SOURCE/'tables/fit_diagnostics.csv')
    np.testing.assert_array_equal(pairs.row_index, diagnostics.loc[diagnostics.usable, 'row_index'])
    with np.load(CONVERTED) as original:
        indices = table.raw_trial_index.to_numpy()
        np.testing.assert_array_equal(table.direction, original['direction'][indices])
        np.testing.assert_array_equal(table.original_trial_number, original['original_trial_number'][indices])
        np.testing.assert_array_equal(table.direction, original['condition_code'][indices])
        np.testing.assert_array_equal(original['trial_type_code'][indices], np.full(198, 91))
    return table, pairs


def radius_plot(table):
    style()
    xy = table[['R1', 'R2']].to_numpy()
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = (hi-lo)*.20
    gx = np.linspace(max(0, lo[0]-pad[0]), hi[0]+pad[0], 180)
    gy = np.linspace(max(0, lo[1]-pad[1]), hi[1]+pad[1], 180)
    xx, yy = np.meshgrid(gx, gy)
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    fig = plt.figure(figsize=(6.0, 5.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4.5], width_ratios=[4.5, 1],
                           left=.12, right=.98, bottom=.12, top=.81, wspace=.04, hspace=.04)
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    right = fig.add_subplot(grid[1, 1], sharey=ax)
    densities, levels, x_densities, y_densities = [], [], [], []
    for side, color in COLORS.items():
        values = xy[table.reach_side.to_numpy() == side]
        density = gaussian_kde(values.T, bw_method='scott')(grid_points).reshape(xx.shape)
        ordered = np.sort(density.ravel())[::-1]
        mass = np.cumsum(ordered)/ordered.sum()
        thresholds = [ordered[np.searchsorted(mass, q)] for q in (.8, .5)]
        xd = gaussian_kde(values[:, 0], bw_method='scott')(gx)
        yd = gaussian_kde(values[:, 1], bw_method='scott')(gy)
        densities.append(density)
        levels.append(thresholds)
        x_densities.append(xd)
        y_densities.append(yd)
        ax.contour(gx, gy, density, levels=thresholds, colors=[color],
                   linewidths=[.9, 1.2], alpha=.75)
        top.plot(gx, xd, color=color, lw=1.1)
        top.fill_between(gx, xd, color=color, alpha=.12)
        right.plot(yd, gy, color=color, lw=1.1)
        right.fill_betweenx(gy, yd, color=color, alpha=.12)
    # Interleave draw order without moving or jittering any fitted radius pair.
    points = table.sample(frac=1, random_state=42)
    ax.scatter(points.R1, points.R2, c=[COLORS[g] for g in points.reach_side],
               s=19, alpha=.66, edgecolors='white', linewidths=.35, zorder=4)
    ax.set(xlabel='R1', ylabel='R2', xlim=(gx[0], gx[-1]), ylim=(gy[0], gy[-1]))
    ax.grid(False)
    top.axis('off')
    right.axis('off')
    fig.text(.12, .958, 'Geometry by reach side', fontsize=11, weight='bold', va='top')
    counts = table.reach_side.value_counts()
    labels = {g: f'{g} (n={counts[g]})\nDirections '+(', '.join(str(d) for d in SIDES if SIDES[d] == g))
              for g in COLORS}
    fig.legend(handles=[Line2D([], [], color=c, lw=1.7, label=labels[g]) for g, c in COLORS.items()],
               ncol=2, frameon=False, loc='upper left', bbox_to_anchor=(.10, .919),
               fontsize=9, handlelength=1.5, columnspacing=2.0)
    fig.text(.12, .818, 'Fitted radii; normalized signal units', fontsize=8, va='bottom')
    density = dict(groups=np.asarray(list(COLORS)), x=gx, y=gy, density=np.asarray(densities),
                   levels=np.asarray(levels), x_density=np.asarray(x_densities), y_density=np.asarray(y_densities))
    return fig, ax, density


def generate(root):
    root = Path(root)
    protected = {str(p): digest(p) for base in (SOURCE, root) for p in base.rglob('*')
                 if p.is_file() and not p.name.startswith(STEM)}
    table, pairs = load_grouped_trials()
    for folder in ('figures', 'previews', 'tables'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    fig, _, density = radius_plot(pairs)
    caption = (
        'Monkey T, session y070316009-12, channel 7; short-delay trials, final 500 ms before GO; '
        'K=1, m=3, tau=3 ms. Direct fitted outer ellipse radii: R1 is the larger and R2 the smaller. '
        'The data repository documents target codes 101-106 as clockwise from the upper-right target. '
        'Its Setup&Task diagram establishes the six-target layout. Thus directions 1, 2, 3 are '
        'upper right, right, lower right (Rightward); 4, 5, 6 are lower left, left, upper left (Leftward). '
        'Sides refer to spatial targets in the task display, not the reaching hand or recorded hemisphere. '
        '198 selected trials, 99 per side; 197 valid radius pairs shown: 99 leftward, 98 rightward. '
        'Original trial 128, direction 3, had a nonconverged fit and is omitted only from this plot. '
        'No points are imputed, shifted, or refit. Each dot is one trial. '
        'Contours summarize approximately 50% and 80% of each group Gaussian KDE mass on the displayed grid; '
        'marginals are one-dimensional KDEs. Scott bandwidths; each group density normalized separately. '
        'Contours are descriptive distributions, not confidence regions or evidence of decoding accuracy. '
        'Raw saved radii are unchanged; no LDA/PCA projection and no binary decoder. '
        'Sources: '+SOURCES['direction_order']+' ; '+SOURCES['target_layout'])
    save_panel(root, STEM, fig, pairs, caption)
    np.savez_compressed(root/f'tables/{STEM}_density.npz', **density)
    mapping = group_directions(pd.DataFrame({'direction': list(POSITIONS)}))
    mapping.to_csv(root/f'tables/{STEM}_mapping.csv', index=False)
    table.assign(plotted=table.row_index.isin(pairs.row_index)).to_csv(
        root/f'tables/{STEM}_trial_accounting.csv', index=False)
    for old, expected in protected.items():
        assert digest(old) == expected, old
    dependencies = [Path(__file__), Path(__file__).with_name('plot_prego_radius_pair.py'),
                    Path(__file__).with_name('plot_prego_fixed_geometry_decoding.py')]
    manifest = dict(sources=SOURCES, mapping=mapping.to_dict(orient='records'),
                    input_sha256={str(p): digest(p) for p in
                                  [SOURCE/'tables/features_geometry.csv', SOURCE/'tables/fit_diagnostics.csv', CONVERTED]},
                    renderer_sha256={str(p): digest(p) for p in dependencies},
                    included_counts=pairs.reach_side.value_counts().to_dict(),
                    excluded_original_trials=table.loc[~table.row_index.isin(pairs.row_index), 'original_trial_number'].tolist(),
                    previous_files_unchanged=len(protected),
                    outputs={str(p.relative_to(root)): digest(p) for p in root.rglob(STEM+'*') if p.is_file()
                             and p.name != STEM+'_manifest.json'})
    (root/f'{STEM}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Saved {root}/figures/{STEM}.*', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    generate(parser.parse_args().output)
