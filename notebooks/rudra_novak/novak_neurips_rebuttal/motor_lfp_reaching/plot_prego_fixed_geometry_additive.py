"""Plot the additive comparison and colored LDA view from saved results only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from plot_prego_fixed_geometry_decoding import style, intervals, save_panel, digest

OUTPUT = Path(__file__).resolve().parent/'outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_additive_v1'
METHODS = ('relevant_band', 'geometry_relevant_band', 'average_psd',
           'geometry_average_psd', 'all_bands', 'geometry')
LABELS = ['Relevant band\n(13-30 Hz)', 'Torus +\nrelevant band', 'Average\nPSD',
          'Torus +\naverage PSD', 'All band\npowers', 'Geometry']
COLORS = ['#e1937a', '#a12e29', '#b0afa9', '#a12e29', '#92918d', '#7f1209']
DIRECTION_COLORS = ['#0072b2', '#d55e00', '#009e73', '#cc79a7', '#e69f00', '#56b4e9']


def comparison_plot(table):
    style()
    if table.method.tolist() != list(METHODS):
        raise ValueError('Unexpected representation order')
    fig, ax = plt.subplots(figsize=(7.1, 3.75), layout='constrained')
    x = np.array([0, 1, 2.4, 3.4, 4.8, 5.8])
    ax.bar(x, table.macro_f1, width=.72, color=COLORS, edgecolor='#6f1009', linewidth=.7)
    intervals(ax, x, table.ci_low, table.ci_high, '#3a0a06', cap=.065)
    for pos, row in zip(x, table.itertuples()):
        ax.text(pos, row.ci_high+.015, f'{row.macro_f1:.3f}', ha='center', fontsize=9)
    ax.axhline(1/6, linestyle='--', color='#777777', lw=1, label='Chance reference: 1/6')
    ax.set(xticks=x, xticklabels=LABELS, ylabel='Macro-F1',
           ylim=(0, max(.35, table.ci_high.max()+.10)), xlim=(-.65, 6.45))
    ax.grid(False)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    return fig, ax


def density_arrays(xy, labels, folds):
    train = folds != 0
    lo, hi = xy[train].min(axis=0), xy[train].max(axis=0)
    pad = np.maximum((hi-lo)*.25, .5)
    gx = np.linspace(lo[0]-pad[0], hi[0]+pad[0], 180)
    gy = np.linspace(lo[1]-pad[1], hi[1]+pad[1], 180)
    xx, yy = np.meshgrid(gx, gy)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    densities, x_density, y_density, levels = [], [], [], []
    for direction in range(1, 7):
        values = xy[train & (labels == direction)]
        density = gaussian_kde(values.T, bw_method='scott')(positions).reshape(xx.shape)
        descending = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(descending)/descending.sum()
        threshold = [descending[np.searchsorted(cumulative, mass)] for mass in (.8, .5)]
        densities.append(density)
        levels.append(threshold)
        x_density.append(gaussian_kde(values[:, 0], bw_method='scott')(gx))
        y_density.append(gaussian_kde(values[:, 1], bw_method='scott')(gy))
    return dict(x=gx, y=gy, density=np.asarray(densities), levels=np.asarray(levels),
                x_density=np.asarray(x_density), y_density=np.asarray(y_density))


def projection_plot(table, density):
    style()
    fig = plt.figure(figsize=(6.0, 5.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4.5], width_ratios=[4.5, 1],
                           left=.12, right=.98, bottom=.12, top=.81, wspace=.04, hspace=.04)
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    right = fig.add_subplot(grid[1, 1], sharey=ax)
    for direction, color in enumerate(DIRECTION_COLORS, 1):
        values = table[table.direction == direction]
        training = values[values.role == 'training']
        heldout = values[values.role == 'held_out']
        j = direction-1
        ax.contour(density['x'], density['y'], density['density'][j], levels=density['levels'][j],
                   colors=[color], linewidths=[.9, 1.2], alpha=.70)
        ax.scatter(training.LD1, training.LD2, s=12, facecolors='none', edgecolors=color,
                   linewidths=.55, alpha=.38, zorder=3)
        ax.scatter(heldout.LD1, heldout.LD2, s=30, color=color, edgecolors='white',
                   linewidths=.45, zorder=4)
        top.plot(density['x'], density['x_density'][j], color=color, lw=1.0)
        top.fill_between(density['x'], density['x_density'][j], color=color, alpha=.09)
        right.plot(density['y_density'][j], density['y'], color=color, lw=1.0)
        right.fill_betweenx(density['y'], density['y_density'][j], color=color, alpha=.09)
    # Limits include every displayed point; density estimation uses training trials only.
    xlo, xhi = min(density['x'][0], table.LD1.min()), max(density['x'][-1], table.LD1.max())
    ylo, yhi = min(density['y'][0], table.LD2.min()), max(density['y'][-1], table.LD2.max())
    ax.set(xlabel='LD1', ylabel='LD2', xlim=(xlo-.04, xhi+.04), ylim=(ylo-.04, yhi+.04))
    ax.grid(False)
    for marginal in (top, right):
        marginal.axis('off')
    fig.text(.12, .958, 'Geometry: LDA projection', fontsize=11, weight='bold', va='top')
    fig.legend(handles=[Line2D([], [], color=c, lw=1.7, label=str(i)) for i, c in enumerate(DIRECTION_COLORS, 1)],
               title='Reach direction', title_fontsize=8.5, ncol=6, frameon=False, loc='upper left',
               bbox_to_anchor=(.10, .926), fontsize=9, handlelength=1.25, columnspacing=1.3)
    fig.text(.12, .818, 'Contours / open points: training     Filled points: held out', fontsize=8, va='bottom')
    return fig, ax


def generate(root):
    root = Path(root)
    if not json.loads((root/'run_status.json').read_text())['complete']:
        raise ValueError('Run the analysis first; plotting never fits models')
    summary = pd.read_csv(root/'tables/summary.csv')
    projection = pd.read_csv(root/'tables/geometry_lda_projection.csv')
    paths = [root/'tables/summary.csv', root/'tables/geometry_lda_projection.csv', root/'config.json']
    before = {str(p): digest(p) for p in paths}
    for folder in ('figures', 'previews'):
        (root/folder).mkdir(exist_ok=True)
    fig, _ = comparison_plot(summary)
    comparison_caption = (
        'Six-direction decoding from 198 short-delay trials (33/direction), Monkey T session '
        'y070316009-12, channel 7, final 500 ms before GO. Fixed K=1, m=3, tau=3 ms. '
        'Bars: pooled held-out macro-F1, using the same saved five folds and training-only '
        'median imputation/scaling with shrinkage LDA. Whiskers: 95% percentile intervals '
        'from 2,000 class-stratified resamples of held-out predictions, conditional on fitted models. '
        'Relevant band: log10 integrated beta PSD (13-30 Hz), 1D. Average PSD: log10 arithmetic '
        'mean density across 2-55 Hz, 1D. Torus + baseline: concatenate the unchanged 14D '
        'geometry with the 1D baseline, 15D. All band powers: 5D; geometry: 14D. '
        'The dashed line at 1/6=0.1667 is the nominal balanced uniform-guessing reference, '
        'not a method-specific empirical macro-F1 null or a significance threshold. '
        'No shuffled-label markers, gridlines, or significance stars are displayed. '
        'No geometry was refitted. One unusable fit remains in the trial cohort via training-fold '
        'imputation. This is exploratory single-recording performance; bar differences are not significance tests.')
    save_panel(root, 'feature_comparison', fig, summary, comparison_caption)
    density = density_arrays(projection[['LD1', 'LD2']].to_numpy(), projection.direction.to_numpy(),
                             projection.heldout_fold_zero_based.to_numpy())
    np.savez_compressed(root/'tables/projection_density.npz', **density)
    fig, _ = projection_plot(projection, density)
    n_train = int((projection.role == 'training').sum())
    n_test = int((projection.role == 'held_out').sum())
    projection_caption = (
        'Descriptive supervised LDA projection of the 14 geometric features, colored by true reach '
        f'direction. A single common projection is fitted on the {n_train} training trials of saved '
        f'fold 0; {n_test} held-out trials are transformed without refitting. Open faint points: training; '
        'filled points: held out. All 198 trials are shown. Contours and marginal densities use '
        'training trials only; two-dimensional contours enclose approximately 50% and 80% of each '
        'estimated class density on the displayed training-derived grid. Gaussian KDE uses Scott bandwidths. '
        'The projection uses eigen-solver LDA with automatic shrinkage solely to expose two visualization '
        'axes; classifier scores use the unchanged lsqr shrinkage LDA on full feature vectors. '
        'LD1 and LD2 are centered and scaled using training coordinates only. No PCA is used. '
        'Axis signs are deterministic; axes from different folds are not pooled. One unusable geometric '
        'fit is imputed with training medians. Training separation is supervised and is not evidence '
        'of held-out accuracy; the held-out points represent one fold, not all five-fold predictions.')
    save_panel(root, 'geometry_lda_projection', fig, projection, projection_caption)
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError('Plotting changed analysis tables')
    (root/'plot_manifest.json').write_text(json.dumps(dict(inputs=before, renderer_sha256=digest(__file__),
        outputs={str(p.relative_to(root)): digest(p) for p in sorted((root/'figures').glob('*'))}), indent=2)+'\n')
    report = ['# Updated Comparison and Direction-Colored Geometry', '',
              'Two requested figures only. The original experiment and its plots remain unchanged.', '',
              '## Comparison', '', '| Representation | Dimensions | Macro-F1 | Conditional 95% interval |',
              '|---|---:|---:|---:|']
    for row in summary.itertuples():
        report.append(f'| {row.method.replace("_", " ")} | {row.dimensions} | {row.macro_f1:.3f} | '
                      f'{row.ci_low:.3f}-{row.ci_high:.3f} |')
    report += ['', comparison_caption, '',
               'All spectral features here refer to the saved raw pre-GO epochs with linear detrending. '
               'Average PSD uses the same Hann/Welch settings as this run\'s frozen all-band features '
               '(500 samples, 250 overlap, nfft=10000), then takes log10 of the mean density at '
               '2-55 Hz. It is not the average of five log band powers. Unlike the legacy movement '
               'analysis\'s average-PSD helper, this definition does not re-normalize amplitude per trial '
               'and does not change the frozen all-band comparison. Old movement-aligned scores are not reused.',
               '', '## LDA View', '', projection_caption, '',
               'The 14D features and K/m/tau settings are unchanged; the plot is a view of feature '
               'space, not a 2D replacement for fitting or decoding. Colors identify directions, not '
               'decoding performance. Displayed training densities must not be read as generalization.',
               '', '## Files and Reproduction', '',
               '- `figures/`: each panel has PDF, editable-text SVG, 600-dpi PNG, caption and CSV.',
               '- `models/`, `features.npz`, `heldout_predictions.npz`, `bootstrap.npz`: full numerical record.',
               '- `projection.npz` and its CSV: common-axis coordinates, roles, labels, folds and projection transform.',
               '- `tables/projection_density.npz`: saved KDE grids, marginal densities and contour levels.',
               '- `provenance.json`, `parent_manifest.json`, `numerical_environment.json`, `source/`: tracking.',
               '- Analysis: `run_prego_fixed_geometry_additive.py`; plot-only: `plot_prego_fixed_geometry_additive.py`.',
               '- Saved geometry/all-band held-out predictions and bootstrap values exactly match the parent run.',
               '- No new significance tests were run; no claim of additive superiority follows from bar ordering.', '']
    (root/'REPORT.md').write_text('\n'.join(report))
    print(f'Both panels saved to {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    generate(parser.parse_args().output)
