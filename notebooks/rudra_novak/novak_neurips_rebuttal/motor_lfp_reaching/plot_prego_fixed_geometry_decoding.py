"""Render only frozen scores and features. Never fit or invoke a decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent/'outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1'
COLORS = dict(peak_power='#f1d2ca', peak_frequency='#e6a491', power_frequency='#d85a30',
              all_bands='#92918d', geometry='#7f1209')
LABELS = dict(peak_power='Peak\npower', peak_frequency='Peak\nfrequency',
              power_frequency='Power +\nfrequency', all_bands='All band\npowers', geometry='Geometry')
EDGE, INK, NULL = '#6f1009', '#242424', '#707070'
CMAP = LinearSegmentedColormap.from_list('salmon_red', ['#ffffff', '#f1d2ca', '#d85a30', '#7f1209'])


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.linewidth': .8})


def tidy(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e8e8e8', linewidth=.6)
    ax.tick_params(length=3, width=.8)


def intervals(ax, x, lo, hi, color, cap=.035, linewidth=1.1, zorder=4):
    ax.vlines(x, lo, hi, color=color, lw=linewidth, zorder=zorder)
    ax.hlines(lo, np.asarray(x)-cap, np.asarray(x)+cap, color=color, lw=linewidth, zorder=zorder)
    ax.hlines(hi, np.asarray(x)-cap, np.asarray(x)+cap, color=color, lw=linewidth, zorder=zorder)


def confusion_plot(matrix):
    fig, ax = plt.subplots(figsize=(3.5, 3.25), layout='constrained')
    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=np.arange(1, 7),
           yticklabels=np.arange(1, 7), xlabel='Predicted direction', ylabel='True direction')
    ax.tick_params(length=0, pad=5)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                    color='white' if matrix[i, j] > .57 else INK)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.7)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.045, ticks=[0, .5, 1])
    bar.set_label('Prediction fraction', fontsize=9)
    bar.outline.set_visible(False)
    return fig, ax


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_panel(root, name, fig, table, caption):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(root/'figures'/f'{name}.{extension}', dpi=600)
    # A screen-sized preview is extra; manuscript PNGs remain 600 dpi.
    fig.savefig(root/'previews'/f'{name}.png', dpi=180)
    table.to_csv(root/'figures'/f'{name}.csv', index=False)
    (root/'figures'/f'{name}.caption.txt').write_text(caption+'\n')
    plt.close(fig)


def generate(root):
    root = Path(root)
    if not json.loads((root/'run_status.json').read_text())['complete']:
        raise ValueError('Analysis is not complete; plotting cannot run analysis')
    if not json.loads((root/'validation.json').read_text())['passed']:
        raise ValueError('Analysis must pass validation before plotting')
    style()
    sources = [root/'config.json', root/'validation.json', root/'tables/summary.csv',
               root/'tables/per_direction.csv', root/'tables/features_geometry.csv']
    sources += [root/'tables'/f'confusion_fraction_{m}.csv' for m in ('power_frequency', 'geometry')]
    before = {str(p): digest(p) for p in sources}
    for folder in ('figures', 'previews'):
        (root/folder).mkdir(exist_ok=True)
    summary = pd.read_csv(root/'tables/summary.csv')
    per_direction = pd.read_csv(root/'tables/per_direction.csv')
    geometry = pd.read_csv(root/'tables/features_geometry.csv')
    quality = json.loads((root/'validation.json').read_text())
    methods = list(COLORS)
    if summary.method.tolist() != methods:
        raise ValueError('Method order differs from the fixed analysis')
    scope = ('Monkey T, session y070316009-12, LFP channel 7; 198 short-delay trials, '
             '33 per direction. Final 500 ms before GO. Fixed K=1, m=3, tau=3 ms (494 points/trial). '
             'Five saved trial-level folds; one held-out prediction per trial. '
             'Observed-score confidence intervals are 95% percentile intervals from 2,000 class-stratified resamples of '
             'held-out predictions, conditional on the fitted models; folds are not independent recordings. ')
    fit_note = (f'{quality["usable_fits"]}/198 usable fits; {quality["unusable_fits"]} unusable feature rows '
                'retained through training-fold median imputation. Boundary-flagged fits were retained. ')

    fig, ax = plt.subplots(figsize=(6.6, 3.65), layout='constrained')
    x = np.arange(5)
    ax.bar(x, summary.macro_f1, width=.68, color=[COLORS[m] for m in methods], edgecolor=EDGE, linewidth=.7)
    intervals(ax, x, summary.macro_f1_ci_low, summary.macro_f1_ci_high, '#3a0a06', cap=.065)
    intervals(ax, x+.24, summary.null_f1_low, summary.null_f1_high, NULL, cap=.035, linewidth=1)
    ax.scatter(x+.24, summary.null_f1_median, color=NULL, s=18, marker='D', zorder=6)
    limit = max(summary.macro_f1_ci_high.max(), summary.null_f1_high.max())
    for j, row in summary.iterrows():
        ax.text(j, row.macro_f1_ci_high+.018, f'{row.macro_f1:.3f}', ha='center', fontsize=9)
    ax.set(xticks=x, xticklabels=[LABELS[m] for m in methods], ylim=(0, min(1.08, max(.30, limit+.10))))
    tidy(ax, 'Macro-F1')
    ax.legend(handles=[Line2D([], [], color=NULL, marker='D', ms=4, lw=1,
                             label='Shuffled labels: median and 95% null interval')],
              loc='upper left', frameon=False, fontsize=8)
    p = summary.loc[summary.method == 'geometry', 'permutation_p'].iloc[0]
    caption = (scope+'Bars show pooled held-out macro-F1, not means across folds. '
               'Feature dimensions: peak power 1D; peak frequency 1D; power + frequency 2D; '
               'all band powers 5D; geometry 14D. Gray diamonds and whiskers show method-specific '
               'null medians and central 95% intervals from 1,000 within-fold label permutations, '
               f'with LDA refitted each time. Geometry one-sided permutation p={p:.6f}, computed as '
               '(1 + null scores at least as large as observed)/1001; other nulls are descriptive. '+fit_note)
    save_panel(root, 'feature_comparison', fig, summary, caption)

    for method in ('power_frequency', 'geometry'):
        table = pd.read_csv(root/'tables'/f'confusion_fraction_{method}.csv', index_col=0)
        matrix = table.to_numpy()
        np.testing.assert_allclose(matrix.sum(axis=1), 1)
        fig, ax = confusion_plot(matrix)
        caption = (scope+f'Method: {method.replace("_", " ")}. Rows are true directions and columns are '
                   'predicted directions. Each row is normalized by its 33 held-out trials; diagonal '
                   'entries are class recall, not F1. Both matrices use the identical 0-1 scale.')
        long = table.reset_index().melt(id_vars='true_direction', var_name='predicted_direction', value_name='prediction_fraction')
        save_panel(root, f'confusion_{method}', fig, long, caption)

    fig, ax = plt.subplots(figsize=(6.0, 3.5), layout='constrained')
    for method, offset in (('power_frequency', -.12), ('geometry', .12)):
        table = per_direction[per_direction.method == method].sort_values('direction')
        xpos = np.arange(1, 7)+offset
        intervals(ax, xpos, table.ci_low, table.ci_high, COLORS[method], cap=.04)
        ax.scatter(xpos, table.f1, color=COLORS[method], s=30, zorder=5, clip_on=False,
                   label='Power + frequency' if method == 'power_frequency' else 'Geometry')
    selected = per_direction[per_direction.method.isin(['power_frequency', 'geometry'])]
    ax.set(xticks=np.arange(1, 7), xlabel='Reach direction', xlim=(.5, 6.5),
           ylim=(0, min(1.05, max(.35, selected.ci_high.max()+.10))))
    tidy(ax, 'Per-direction F1')
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=9)
    save_panel(root, 'per_direction_f1', fig, selected, scope+
               'Dots show pooled held-out F1 for each direction. Offset within-direction points compare '
               'power + frequency (coral) with geometry (dark red). Whiskers are conditional bootstrap '
               'intervals; no direction-by-direction significance tests were performed. A class with '
               'zero observed true positives has a degenerate [0, 0] conditional interval, which '
               'does not imply zero uncertainty about performance on future trials.')

    rng = np.random.default_rng(61000)
    for radius in ('R1', 'R2'):
        fig, ax = plt.subplots(figsize=(4.25, 3.2), layout='constrained')
        groups = [geometry.loc[geometry.direction == d, radius].dropna().to_numpy() for d in range(1, 7)]
        ax.boxplot(groups, positions=np.arange(1, 7), widths=.52, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='#f1d2ca', edgecolor=EDGE, linewidth=.8),
                   medianprops=dict(color=EDGE, linewidth=1.2), whiskerprops=dict(color=EDGE, linewidth=.8),
                   capprops=dict(color=EDGE, linewidth=.8))
        for d, values in enumerate(groups, 1):
            ax.scatter(d+rng.uniform(-.18, .18, len(values)), values, s=9, color=COLORS['geometry'],
                       alpha=.62, linewidths=0, zorder=3)
        ax.set(xlabel='Reach direction', xticks=np.arange(1, 7), xlim=(.5, 6.5))
        ax.set_ylim(bottom=0)
        tidy(ax, f'{radius} (normalized signal units)')
        description = 'major' if radius == 'R1' else 'minor'
        caption = (f'Outer {description}-axis radius of the fitted planar annular band, ordered R1 >= R2. '
                   'All finite trial measurements are shown, with median, interquartile range, and '
                   '1.5-IQR boxplot whiskers; extreme points are not hidden. Missing measurements are '
                   'not replaced by imputed values in this descriptive plot. '+fit_note+
                   'These radii summarize the same normalized, three-coordinate clouds used for decoding; '
                   'they do not establish topology or directional separation. No band half-width is plotted.')
        save_panel(root, f'geometry_{radius}', fig, geometry[['row_index', 'original_trial_number', 'direction', radius]], caption)

    if {str(p): digest(p) for p in sources} != before:
        raise ValueError('Plotting altered frozen analysis inputs')
    manifest = dict(source_tables=before, renderer_sha256=digest(__file__),
                    outputs={str(p.relative_to(root)): digest(p) for p in sorted((root/'figures').glob('*'))})
    (root/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write_report(root, summary, quality, scope)
    return manifest


def write_report(root, summary, quality, scope):
    lines = ['# Fixed-Geometry Decoding: T, Channel 7', '', scope, '',
             '## Results', '', '| Representation | Dimensions | Macro-F1 (95% conditional interval) | Accuracy |',
             '|---|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.method.replace("_", " ")} | {row.dimensions} | '
                     f'{row.macro_f1:.3f} ({row.macro_f1_ci_low:.3f}-{row.macro_f1_ci_high:.3f}) | {row.accuracy:.3f} |')
    geo = summary[summary.method == 'geometry'].iloc[0]
    lines += ['', f'Geometry permutation p = {geo.permutation_p:.6f} (1,000 within-fold shuffles).',
              'This is a test against shuffled labels, not a significance test against another representation.', '',
              '## Fit Accounting', '',
              f'- {quality["usable_fits"]}/198 usable fits; {quality["unusable_fits"]} unusable fits.',
              f'- Unusable reasons: {quality["unusable_reasons"]}. All 198 trials still receive predictions.',
              f'- Active-bound fits: {quality["active_bound_fits"]}; near-bound fits: {quality["near_bound_fits"]}. '
              'These flags overlap and were not exclusion criteria.', '',
              '## Inputs and Features', '',
              '- Geometry reuses the exact frozen 500-sample epochs, previously detrended, median/MAD normalized, '
              'and zero-phase filtered at 2-55 Hz. No filtering was repeated.',
              '- Clouds are [x(t), x(t-3), x(t-6)], 494 points per trial. Each trial is fitted independently '
              'using the existing 1-torus planar elliptical annular-band fitter (lam=0.1, hole_ratio=0.5, '
              'Huber loss, three angular regularization harmonics, max_nfev=6000). The three regularization '
              'harmonics are not K. There is no PCA, PINN, or shape-matrix representation. The native full-dimensional '
              'SVD initialization is unchanged; it does not discard coordinates.',
              '- Decoder columns, in saved order: R1, R2, MSE, mean_error, frac_inside, normal_x/y/z, '
              'u_x/y/z, v_x/y/z (14 total). R1 >= R2. The normal and major-axis vectors have their largest '
              'absolute component positive; the minor-axis vector is their cross product. No center or width feature.',
              '- MSE is native mean squared 3D distance to the fitted annular set; mean_error is the native '
              'in-plane containment-distance average; frac_inside is the fraction whose planar projection falls '
              'inside the band. These are distinct, not three equivalent fit errors. Band width remains internal '
              'to the unchanged fitter and therefore affects fit-quality measures, although it is not a decoder column.',
              '- Spectral features use the saved raw pre-GO epochs, with linear detrending but without geometric '
              'normalization or its bandpass. Peak power is log10 peak PSD and peak frequency is its frequency '
              'within 12-40 Hz: Hamming Welch, 300 samples, 200 overlap, nfft=10000. All-band features are log10 '
              'integrated powers in 2-4, 4-8, 8-13, 13-30, and 30-55 Hz: Hann Welch, 500 samples, 250 overlap, '
              'nfft=10000. Zero padding interpolates the frequency grid, not spectral resolution.', '',
              '## Evaluation and Interpretation', '',
              '- All methods use identical trials and saved folds. Median imputation and scaling are fitted '
              'on training trials only, followed by lsqr LDA with automatic covariance shrinkage. Each trial '
              'contributes exactly one held-out prediction. Macro-F1 is the unweighted average of the six class F1 scores.',
              '- The null preserves the class counts within each saved fold and refits all 25 method/fold pipelines '
              'for each of 1,000 shuffles. Only geometry has a reported inferential p-value. Null percentiles are '
              'descriptive reference intervals, not uncertainty intervals on the observed score.',
              '- The 2,000 bootstrap samples resample 33 held-out predictions per true class, jointly across methods. '
              'Intervals are conditional on the fitted models: no refitting, no recording/day/animal uncertainty.',
              '- This is an exploratory, selected single-recording analysis. Tau=3 ms was visually chosen from '
              'earlier examples, outside nested validation; these results are not a confirmatory assessment of '
              'that choice. Trial-level folds are not time-blocked. Permutations assume exchangeability within '
              'folds and do not preserve temporal trial dependence.',
              '- No superiority claim follows from bar ordering or overlapping/nonoverlapping intervals. '
              'These results do not establish the correct topology or demonstrate cross-recording or cross-animal generalization.',
              '- Direction 4 has zero observed true positives for geometry. Its [0, 0] conditional bootstrap '
              'interval cannot quantify uncertainty about future performance; resampling these predictions '
              'cannot create previously unobserved correct classifications.',
              '- No retuning was performed in response to these scores.', '',
              '## Files', '',
              '- `figures/`: six standalone PDF, editable-text SVG, and 600-dpi PNG panels, each with CSV and caption.',
              '- `previews/`: screen-sized copies of each panel.',
              '- `tables/`: pooled scores, per-direction F1, all five confusion matrices, features and trial fit diagnostics.',
              '- `checkpoints/`: all 198 native fit parameters, diagnostics, failures and input-cloud hashes.',
              '- `models/`, `training_audit.json`: fitted fold pipelines and training preprocessing statistics.',
              '- `heldout_predictions.npz`, `permutations.npz`, `bootstrap.npz`: predictions, shuffled labels/predictions, '
              'bootstrap trial indices and their scores.',
              '- `config.json`, `provenance.json`, `source/`, `validation.json`: frozen definitions, source/input hashes, '
              'analysis source snapshot and numerical checks.', '',
              '## Regeneration', '',
              'Plot-only: run `plot_prego_fixed_geometry_decoding.py --output <this directory>`. '
              'It reads frozen tables and cannot invoke fitting or decoding.',
              'Validation-only: run `run_prego_fixed_geometry_decoding.py --output <this directory> --verify-only`. '
              'Analysis resume requires `--resume` and unchanged configuration/source/input hashes.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.output)
    print(f'Figures and REPORT.md: {args.output}', flush=True)
