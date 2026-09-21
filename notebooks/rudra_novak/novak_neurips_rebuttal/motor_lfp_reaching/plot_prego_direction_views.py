"""Plot saved direction predictions and geometric measurements; no analysis reruns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from plot_prego_fixed_geometry_decoding import style, save_panel, digest, CMAP

BASE = Path(__file__).resolve().parent/'outputs'
SOURCE = BASE/'prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1'
OUTPUT = BASE/'prego_T_y070316009_12_ch7_K1_m3_tau3_14D_direction_views_v1'
POSITIONS = ['Upper right', 'Right', 'Lower right', 'Lower left', 'Left', 'Upper left']
SHORT_POSITIONS = ['UR', 'R', 'LR', 'LL', 'L', 'UL']
TARGET_XY = np.array([[.5, np.sqrt(3)/2], [1., 0.], [.5, -np.sqrt(3)/2],
                      [-.5, -np.sqrt(3)/2], [-1., 0.], [-.5, np.sqrt(3)/2]])
ORIENTATION = [f'{v}_{axis}' for v in ('normal', 'u', 'v') for axis in ('x', 'y', 'z')]
METHODS = {'geometry': 'Geometry', 'all_bands': 'All band powers'}
RED, CORAL, INK = '#7f1209', '#d85a30', '#242424'
SCOPE = ('Monkey T, session y070316009-12, LFP channel 7; 198 short-delay trials, '
         '33 per direction; final 500 ms before GO. K=1, m=3, tau=3 ms. '
         'Exploratory single-recording results. ')
MAPPING_SOURCE = ('https://gin.g-node.org/kilavik.b/'
                  'Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior')


def load_inputs():
    if not json.loads((SOURCE/'validation.json').read_text())['passed']:
        raise ValueError('Frozen source validation did not pass')
    table = pd.read_csv(SOURCE/'tables/features_geometry.csv')
    diag = pd.read_csv(SOURCE/'tables/fit_diagnostics.csv')
    np.testing.assert_array_equal(table.row_index, diag.row_index)
    if len(table) != 198 or not table.delay.eq('short').all():
        raise ValueError('Unexpected frozen cohort')
    np.testing.assert_array_equal(table.groupby('direction').size(), [33]*6)
    usable = table.loc[diag.usable].copy()
    assert len(usable) == 197
    assert np.isfinite(usable[['R1', 'R2']+ORIENTATION].to_numpy()).all()
    matrices, counts = {}, {}
    for method in METHODS:
        for stem, destination in [('confusion_fraction', matrices), ('confusion_counts', counts)]:
            saved = pd.read_csv(SOURCE/f'tables/{stem}_{method}.csv', index_col=0)
            np.testing.assert_array_equal(saved.index, range(1, 7))
            assert saved.columns.tolist() == [str(i) for i in range(1, 7)]
            destination[method] = saved.to_numpy()
        np.testing.assert_array_equal(counts[method].sum(axis=1), [33]*6)
        np.testing.assert_allclose(matrices[method], counts[method]/33, atol=1e-14)
        np.testing.assert_allclose(matrices[method].sum(axis=1), 1)
    return usable, matrices, counts


def feature_summary(table, features):
    """Descriptive display aggregation only; no testing, resampling, or fitted curve."""
    rows = []
    for feature in features:
        for direction in range(1, 7):
            values = table.loc[table.direction == direction, feature].dropna()
            rows.append(dict(feature=feature, direction=direction,
                             position=POSITIONS[direction-1], n=len(values),
                             mean=values.mean(), sd=values.std(ddof=1)))
    return pd.DataFrame(rows)


def draw_target_map(ax, fractions, true_direction):
    for d, ((x, y), fraction) in enumerate(zip(TARGET_XY, fractions), 1):
        circle = Circle((x, y), .32, facecolor=CMAP(fraction),
                        edgecolor=INK if d == true_direction else '#a7a7a7',
                        linewidth=2 if d == true_direction else .6)
        ax.add_patch(circle)
        ax.text(x, y, f'{fraction:.2f}', fontsize=8, ha='center', va='center',
                color='white' if fraction > .45 else INK)
        ax.text(1.50*x, 1.50*y, str(d), ha='center', va='center', fontsize=8)
    ax.plot(0, 0, '.', color='#999999', ms=3)
    ax.set(xlim=(-1.68, 1.68), ylim=(-1.52, 1.52), aspect='equal')
    ax.set_axis_off()
    ax.set_title(f'True direction {true_direction}', fontsize=9, pad=3)


def target_legend(fig, y):
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=INK,
                               markerfacecolor='white', markeredgewidth=1.8,
                               label='Bold outline: instructed target')],
               loc='lower center', bbox_to_anchor=(.47, y), frameon=False, fontsize=8)


def target_plot(matrix, title):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    fig.subplots_adjust(left=.025, right=.87, bottom=.12, top=.82, hspace=.35, wspace=.16)
    for direction, ax in enumerate(axes.flat, 1):
        draw_target_map(ax, matrix[direction-1], direction)
    fig.text(.05, .95, title, fontsize=12, weight='bold', va='top')
    fig.text(.05, .895, 'Held-out predictions; 33 trials per instructed direction', fontsize=9, va='top')
    cax = fig.add_axes([.91, .26, .016, .43])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cax, ticks=[0, .5, 1])
    bar.set_label('Prediction fraction', fontsize=9)
    bar.outline.set_visible(False)
    target_legend(fig, .015)
    return fig


def target_comparison(matrices):
    style()
    fig, axes = plt.subplots(2, 6, figsize=(11.6, 5.0))
    fig.subplots_adjust(left=.025, right=.91, bottom=.13, top=.82, hspace=.55, wspace=.14)
    for row, (method, label) in enumerate(METHODS.items()):
        for direction, ax in enumerate(axes[row], 1):
            draw_target_map(ax, matrices[method][direction-1], direction)
        fig.text(.04, .885 if row == 0 else .465, label, fontsize=11, weight='bold')
    fig.text(.04, .97, 'Where does the decoder predict the reach?', fontsize=12, weight='bold', va='top')
    cax = fig.add_axes([.94, .24, .012, .46])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cax, ticks=[0, .5, 1])
    bar.set_label('Prediction fraction', fontsize=9)
    bar.outline.set_visible(False)
    target_legend(fig, .01)
    return fig


def radius_profile(ax, features, summary, feature):
    values = summary[summary.feature == feature].sort_values('direction')
    for direction in range(1, 7):
        trials = features.loc[features.direction == direction].sort_values('original_trial_number')
        # Horizontal offsets expose overlapping trial points; radii are unchanged.
        x = direction+np.linspace(-.13, .13, len(trials))
        ax.scatter(x, trials[feature], s=9, c='#a5a5a5', alpha=.38, edgecolors='none', zorder=1)
    ax.errorbar(values.direction, values['mean'], yerr=values.sd, fmt='o-',
                color=RED, linewidth=1.2, markersize=4, capsize=3, elinewidth=1, zorder=3)
    upper = max(features[feature].max(), (values['mean']+values.sd).max())*1.08
    ax.set(xticks=np.arange(1, 7), xticklabels=[f'{d}\n{p}' for d, p in enumerate(SHORT_POSITIONS, 1)],
           xlim=(.6, 6.4), ylim=(0, upper), xlabel='Reach direction (clockwise)',
           ylabel=f'{feature} (normalized signal units)')
    ax.set_title('Outer major radius (R1)' if feature == 'R1' else 'Outer minor radius (R2)',
                 fontsize=10, pad=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(False)


def radius_plot(features, summary, radii):
    style()
    combined = len(radii) == 2
    fig, axes = plt.subplots(1, len(radii), figsize=(7.2 if combined else 4.6, 3.5), squeeze=False)
    fig.subplots_adjust(left=.09 if combined else .16, right=.98, bottom=.22, top=.73,
                        wspace=.38)
    for ax, radius in zip(axes.flat, radii):
        radius_profile(ax, features, summary, radius)
    fig.text(.09 if combined else .16, .965, 'Geometric measurements by reach direction',
             fontsize=11, weight='bold', va='top')
    fig.text(.09 if combined else .16, .89, 'Mean +/- SD; gray dots show individual trials', fontsize=8.5, va='top')
    return fig


def orientation_plot(summary):
    style()
    matrix = summary.pivot(index='feature', columns='direction', values='mean').loc[ORIENTATION].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    fig.subplots_adjust(left=.22, right=.82, bottom=.21, top=.81)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    for i in range(9):
        for j in range(6):
            label = f'{matrix[i, j]:.3f}' if abs(matrix[i, j]) >= .0005 else '0.000'
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='white' if abs(matrix[i, j]) > .6 else INK)
    labels = [f'{v}{axis}' for v in ('n', 'u', 'v') for axis in ('x', 'y', 'z')]
    ax.set(xticks=np.arange(6), xticklabels=[f'{d}\n{p}' for d, p in enumerate(SHORT_POSITIONS, 1)],
           yticks=np.arange(9), yticklabels=labels, xlabel='Reach direction (clockwise)')
    ax.tick_params(length=0, axis='both', pad=6, labelsize=8)
    ax.hlines([2.5, 5.5], -.5, 5.5, color='white', lw=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.055, ticks=[-1, 0, 1])
    bar.set_label('Mean vector component', fontsize=9)
    bar.outline.set_visible(False)
    fig.text(.08, .96, 'Orientation components by reach direction', fontsize=11, weight='bold', va='top')
    fig.text(.08, .895, 'Saved components; fixed -1 to 1 scale', fontsize=9, va='top')
    fig.text(.08, .08, 'n: plane normal    u: major axis    v: minor axis', fontsize=8)
    fig.text(.08, .04, 'x, y, z are lag coordinates, not physical reach axes', fontsize=8)
    return fig, ax


def confusion_long(matrices, counts):
    return pd.DataFrame([dict(method=method, true_direction=i+1, predicted_direction=j+1,
                              predicted_position=POSITIONS[j], count=int(counts[method][i, j]),
                              prediction_fraction=matrix[i, j])
                         for method, matrix in matrices.items() for i in range(6) for j in range(6)])


def generate(root, panel='all'):
    root = Path(root)
    # Read-only snapshots protect both the frozen analysis and prior plot-only outputs.
    protected_roots = [SOURCE, BASE/'prego_T_y070316009_12_ch7_K1_m3_tau3_14D_additive_v1']
    if any(root.resolve().is_relative_to(base.resolve()) or base.resolve().is_relative_to(root.resolve())
           for base in protected_roots):
        raise ValueError('Plot outputs must not overlap the frozen analysis or prior plots')
    protected = {str(p): digest(p) for base in protected_roots for p in base.rglob('*') if p.is_file()}
    features, matrices, counts = load_inputs()
    for folder in ('figures', 'previews', 'tables'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    features.to_csv(root/'tables/plotted_geometry_trials.csv', index=False)
    mapping = pd.DataFrame(dict(direction=np.arange(1, 7), target_event_code=np.arange(101, 107),
                               position=POSITIONS, schematic_x=TARGET_XY[:, 0], schematic_y=TARGET_XY[:, 1]))
    mapping.to_csv(root/'tables/target_mapping.csv', index=False)
    if panel in ('targets', 'all'):
        long = confusion_long(matrices, counts)
        caption = (SCOPE+'Spatial rendering of the existing row-normalized confusion matrices. '
                   'Each small diagram conditions on one true direction. Target fill and numbers show the fraction '
                   'of its 33 held-out trials assigned to each predicted direction. The bold outline marks the '
                   'instructed target; its value is class recall, not F1 or predicted probability confidence. '
                   'All 198 trials are included, exactly as in saved predictions, including the one trial with '
                   'unusable geometry handled by the already-fitted training-fold imputation. '
                   'Both methods use a fixed 0-1 color scale. The dataset numbers targets clockwise from upper right: '
                   '1 upper right, 2 right, 3 lower right, 4 lower left, 5 left, 6 upper left. '
                   'Locations are a schematic, not measured hand positions. Direction 4 has zero correct '
                   'geometry predictions; this is preserved without adjustment. No fitting or decoding was rerun. '
                   'Mapping references: '+MAPPING_SOURCE+'/raw/master/README.md and '
                   +MAPPING_SOURCE+'/raw/master/Setup%26Task.pdf')
        for method, label in METHODS.items():
            save_panel(root, 'target_map_'+method, target_plot(matrices[method], label),
                       long[long.method == method], caption)
        save_panel(root, 'target_map_comparison', target_comparison(matrices), long, caption)
    if panel in ('radii', 'all'):
        summary = feature_summary(features, ['R1', 'R2'])
        caption = (SCOPE+'Descriptive direction profiles of the larger (R1) and smaller (R2) outer ellipse radii. '
                   'Red dots and straight connecting segments show arithmetic means; whiskers show sample SD '
                   '(ddof=1), not confidence intervals. No tuning function was fitted. Gray dots show all valid '
                   'individual trial measurements with small horizontal offsets for visibility; radius values '
                   'are unchanged. Measurements are in the normalized geometric-input signal units. '
                   '197 usable fits: n=33 in directions 1, 2, 4, 5, 6 and n=32 in direction 3. '
                   'The nonconverged original trial 128 is omitted from feature plots only; no imputation. '
                   'UR/R/LR/LL/L/UL mean upper right/right/lower right/lower left/left/upper left. '
                   'The zero-based vertical axes use different ranges for the two radii. '
                   'Descriptive plots do not establish significant tuning or explain held-out decoding by themselves.')
        for radius in ('R1', 'R2'):
            save_panel(root, 'direction_profile_'+radius, radius_plot(features, summary, [radius]),
                       summary[summary.feature == radius], caption)
        save_panel(root, 'direction_profiles_radii', radius_plot(features, summary, ['R1', 'R2']), summary, caption)
    if panel in ('orientation', 'all'):
        summary = feature_summary(features, ORIENTATION)
        fig, _ = orientation_plot(summary)
        caption = (SCOPE+'Heatmap of arithmetic means of all nine saved orientation components; no feature '
                   'selection, row standardization, preferred-direction alignment, or new projection. '
                   'Normal n, major-axis u, and minor-axis v each have x/y/z components along '
                   '[x(t), x(t-3 ms), x(t-6 ms)], not anatomical or physical reach coordinates. '
                   'The source fit canonicalized axis order and vector signs; those saved components are unchanged. '
                   'A mean vector component is not a newly fitted orientation or a mean reach angle. '
                   'Color limits are fixed at -1 to 1; cell text gives means to three decimals. '
                   'The accompanying CSV also includes n and sample SD. All 197 usable fits, with n=32 for '
                   'direction 3 and n=33 otherwise; nonconverged original trial 128 omitted without imputation. '
                   'No significance tests or directional-tuning fits were performed.')
        save_panel(root, 'orientation_direction_heatmap', fig, summary, caption)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    outputs = {str(p.relative_to(root)): digest(p) for folder in ('figures', 'tables')
               for p in (root/folder).glob('*') if p.is_file()}
    manifest = dict(source=str(SOURCE), requested_panel=panel,
                    renderer_sha256=digest(__file__), helper_sha256=digest(Path(__file__).with_name('plot_prego_fixed_geometry_decoding.py')),
                    source_tables={str(p): digest(p) for p in (SOURCE/'tables').glob('*.csv')},
                    existing_files_unchanged=len(protected), outputs=outputs,
                    operations=['read frozen CSVs', 'plot saved confusion fractions',
                                'display arithmetic feature means and sample SDs', 'export artwork'],
                    prohibited_operations_run=[], prediction_trials=198, plotted_feature_trials=197)
    (root/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report = ('# Plot-only direction views\n\n'+SCOPE+'\n\n'
              'These are visualizations of frozen tables, not a new experiment. No decoder evaluations, '
              'torus fits, feature extraction, bootstrap, permutation tests, or tuning models were run.\n\n'
              '## Included\n\n'
              '- Target-space held-out predictions: geometry and all-band powers, standalone plus comparison.\n'
              '- R1 and R2 direction profiles: raw trial dots and arithmetic mean +/- sample SD.\n'
              '- All nine orientation components: mean values on a fixed -1 to 1 scale.\n\n'
              '## Not run\n\n'
              '- Feature-family ablation: requires new decoder evaluations.\n'
              '- Time-resolved decoding: requires new windows, features, fits, and decoder evaluations.\n\n'
              '## Accounting\n\n'
              'Prediction maps retain all 198 trials (33 per direction). Feature plots use 197 usable fits; '
              'original trial 128 in direction 3 remains omitted from these descriptive feature views only. '
              'Unusable values were not imputed for plotting. Boundary-flagged but usable fits remain included.\n\n'
              'Direction order: 1 upper right, 2 right, 3 lower right, 4 lower left, 5 left, 6 upper left. '
              'Target positions are schematic. Source definitions: '+MAPPING_SOURCE+'/raw/master/README.md '
              'and '+MAPPING_SOURCE+'/raw/master/Setup%26Task.pdf.\n\n'
              'PDF and SVG preserve vector artwork and text; PNGs are 600 dpi. Small PNGs in previews/ are '
              'for screen viewing. Every panel has its own caption and underlying display table; raw plotted '
              'features are also saved in tables/plotted_geometry_trials.csv.\n\n'
              f'Verified {len(protected)} existing source/plot files unchanged.\n')
    (root/'README.md').write_text(report)
    print(f'Saved plot-only outputs to {root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--panel', choices=['all', 'targets', 'radii', 'orientation'], default='all')
    args = parser.parse_args()
    generate(args.output, args.panel)
