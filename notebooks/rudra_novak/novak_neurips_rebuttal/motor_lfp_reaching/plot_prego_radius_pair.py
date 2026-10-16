"""Descriptive R1-versus-R2 plot from frozen fitted radii; no decoder or fitter."""
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
from plot_prego_fixed_geometry_additive import DIRECTION_COLORS, OUTPUT

SOURCE = Path(__file__).resolve().parent/'outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1'


def valid_pairs(table):
    keep = np.isfinite(table[['R1', 'R2']].to_numpy()).all(axis=1)
    result = table.loc[keep].copy()
    if (result.R2 <= 0).any() or (result.R1 < result.R2).any():
        raise ValueError('Expected positive, canonically ordered outer radii')
    return result


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
    for direction, color in enumerate(DIRECTION_COLORS, 1):
        values = xy[table.direction.to_numpy() == direction]
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
        top.fill_between(gx, xd, color=color, alpha=.09)
        right.plot(yd, gy, color=color, lw=1.1)
        right.fill_betweenx(gy, yd, color=color, alpha=.09)
    # Interleave directions to avoid systematically drawing one class over others.
    points = table.sample(frac=1, random_state=42)
    colors = [DIRECTION_COLORS[int(d)-1] for d in points.direction]
    ax.scatter(points.R1, points.R2, c=colors, s=19, alpha=.66,
               edgecolors='white', linewidths=.35, zorder=4)
    ax.set(xlabel='R1', ylabel='R2', xlim=(gx[0], gx[-1]), ylim=(gy[0], gy[-1]))
    ax.grid(False)
    top.axis('off')
    right.axis('off')
    fig.text(.12, .958, 'Geometry by reach direction', fontsize=11, weight='bold', va='top')
    fig.legend(handles=[Line2D([], [], color=c, lw=1.7, label=str(d))
                        for d, c in enumerate(DIRECTION_COLORS, 1)],
               title='Reach direction', title_fontsize=8.5, ncol=6, frameon=False,
               loc='upper left', bbox_to_anchor=(.10, .926), fontsize=9,
               handlelength=1.25, columnspacing=1.3)
    fig.text(.12, .818, f'{len(table)} valid trials; normalized signal units', fontsize=8, va='bottom')
    density = dict(x=gx, y=gy, density=np.asarray(densities), levels=np.asarray(levels),
                   x_density=np.asarray(x_densities), y_density=np.asarray(y_densities))
    return fig, ax, density


def generate(root):
    root = Path(root)
    path = SOURCE/'tables/features_geometry.csv'
    source_hash = digest(path)
    table = pd.read_csv(path)
    pairs = valid_pairs(table)
    if len(table) != 198 or len(pairs) != 197:
        raise ValueError('Unexpected frozen trial accounting')
    diagnostics = pd.read_csv(SOURCE/'tables/fit_diagnostics.csv')
    np.testing.assert_array_equal(pairs.row_index, diagnostics.loc[diagnostics.usable, 'row_index'])
    existing = {str(p): digest(p) for p in root.rglob('*') if p.is_file() and
                'geometry_r1_r2' not in p.name and p.name != 'radius_plot_manifest.json'}
    for folder in ('figures', 'previews', 'tables'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    fig, _, density = radius_plot(pairs)
    caption = (
        'Direct fitted-radius geometry for Monkey T, session y070316009-12, channel 7; '
        'short-delay trials, final 500 ms before GO; K=1, m=3, tau=3 ms. '
        'R1 and R2 are the larger and smaller outer ellipse radii of the planar annular-band fit, '
        'in normalized signal units. Each point is one valid trial, colored by true reach direction. '
        'All 197 valid pairs are shown (33 per direction except direction 3, with 32). '
        'Original trial 128 (saved row 69) had a nonconverged fit and is omitted only from this '
        'descriptive plot, not from the 198-trial decoding cohort. No missing points are imputed here. '
        'Contours enclose approximately 50% and 80% of each direction-specific Gaussian KDE mass '
        'on the displayed grid; marginal curves show the corresponding one-dimensional densities. '
        'All KDEs use Scott bandwidths. These are smoothed data distributions, not confidence regions. '
        'Axes are actual saved radii, not LDA/PCA components, and are not standardized again. '
        'Labels control colors and grouping only. There is no train/test distinction because no '
        'supervised coordinate transform is learned. No refitting or decoding was run. '
        'This view uses only two of the 14 decoder features; overlap does not exclude information '
        'in the other features.')
    columns = ['row_index', 'original_trial_number', 'direction', 'heldout_fold_zero_based', 'R1', 'R2']
    save_panel(root, 'geometry_r1_r2', fig, pairs[columns], caption)
    np.savez_compressed(root/'tables/geometry_r1_r2_density.npz', **density)
    assert digest(path) == source_hash
    for old, expected in existing.items():
        assert digest(old) == expected, old
    manifest = dict(source=str(path), source_sha256=source_hash,
                    renderer_sha256=digest(__file__), rows=197,
                    excluded_rows=table.loc[~table.index.isin(pairs.index), 'row_index'].tolist(),
                    previous_files_unchanged=len(existing),
                    outputs={str(p.relative_to(root)): digest(p) for p in root.rglob('*geometry_r1_r2*') if p.is_file()})
    (root/'radius_plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Saved {root}/figures/geometry_r1_r2.*', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    generate(parser.parse_args().output)
