"""Select contrasting single-trial examples and plot frozen clouds and fits only."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

from plot_prego_persistent_homology import (
    ROOT, SOURCE, digest, geometry_panel, outlines, plt,
)

OUTPUT = ROOT / 'geometry_diverse_examples_v1'
RULES = (
    ('normalized_H1_lifetime', 'min', 'Shortest size-normalized H1 lifetime'),
    ('normalized_H1_lifetime', 'max', 'Longest size-normalized H1 lifetime'),
    ('axis_ratio', 'min', 'Most elongated fitted outer ellipse'),
    ('axis_ratio', 'max', 'Least elongated fitted outer ellipse'),
    ('R1', 'min', 'Smallest fitted major radius'),
    ('R1', 'max', 'Largest fitted major radius'),
)


def select_examples(measures):
    pool = measures.loc[measures.geometry_usable & measures.status.eq('success')].copy()
    pool['axis_ratio'] = pool.R2 / pool.R1
    columns = ['normalized_H1_lifetime', 'axis_ratio', 'R1']
    if not np.isfinite(pool[columns]).all().all() or (pool.R1 <= 0).any():
        raise ValueError('Invalid saved selection measurements.')
    if len(pool) < 6 or not pool.row_index.is_unique:
        raise ValueError('Need at least six unique usable trials.')
    chosen = []
    # Sequential extrema, without replacement; ties use original trial identity.
    for metric, extremum, reason in RULES:
        row = pool.sort_values([metric, 'original_trial_number'],
                               ascending=[extremum == 'min', True]).iloc[0].copy()
        row['selection_metric'] = metric
        row['selection_extremum'] = extremum
        row['selection_reason'] = reason
        row['selection_value'] = row[metric]
        chosen.append(row)
        pool = pool.loc[pool.row_index.ne(row.row_index)]
    result = pd.DataFrame(chosen).reset_index(drop=True)
    result.insert(0, 'example', np.arange(1, 7))
    return result


def common_limits(clouds, boundaries):
    points = np.concatenate([*clouds, *(line for pair in boundaries for line in pair)])
    extent = float(np.abs(points).max()) * 1.06
    return (-extent, extent)


def make_panel(cloud, boundary, direction, trial, limits):
    fig = geometry_panel(cloud, boundary, direction, limits)
    fig.texts[0].set_text(f'Trial {trial} | Direction {direction}')
    return fig


def make_preview(paths, output):
    # Two rows, with each low/high selection pair occupying one column.
    with Image.open(paths[0]) as panel:
        width, height = panel.size
    canvas = Image.new('RGB', (3 * width, 2 * height), 'white')
    for index, path in enumerate(paths):
        with Image.open(path) as panel:
            canvas.paste(panel.convert('RGB'), ((index // 2) * width, (index % 2) * height))
    canvas.save(output / 'geometry_examples_six_panel.png', dpi=(600, 600))
    canvas.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
    canvas.save(output / 'geometry_examples_preview.png')


def generate():
    protected = {str(path): digest(path) for root in (ROOT, SOURCE)
                 for path in root.rglob('*') if path.is_file() and OUTPUT not in path.parents}
    measures = pd.read_csv(ROOT / 'tables/trial_measurements.csv', float_precision='round_trip')
    selected = select_examples(measures)
    with np.load(SOURCE / 'clouds.npz', allow_pickle=False) as data:
        clouds = data['clouds']
    if clouds.shape != (198, 494, 3) or not np.isfinite(clouds).all():
        raise ValueError('Unexpected frozen clouds.')
    metadata = pd.read_csv(SOURCE / 'inputs/selected_trials.csv')
    boundaries = []
    for row in selected.itertuples():
        source_row = metadata.loc[metadata.row_index.eq(row.row_index)].iloc[0]
        assert source_row.original_trial_number == row.original_trial_number
        assert source_row.direction == row.direction
        saved = json.loads((SOURCE / 'checkpoints' / f'trial_{row.row_index:03d}_one_torus.json').read_text())
        assert saved['row_index'] == row.row_index
        assert saved['original_trial_number'] == row.original_trial_number
        assert saved['direction'] == row.direction
        np.testing.assert_allclose([saved['fit']['R1'], saved['fit']['R2']], [row.R1, row.R2])
        boundaries.append(outlines(saved['fit']))
    limits = common_limits(clouds[selected.row_index], boundaries)
    for folder in ('figures', 'tables', 'captions', 'source'):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    selected['png'] = [f'geometry_example_{r.example:02d}_trial_{r.original_trial_number}_direction_{r.direction}.png'
                       for r in selected.itertuples()]
    selected.to_csv(OUTPUT / 'tables/selected_examples.csv', index=False)
    paths = []
    plot_data = []
    caption = ('Monkey T, y070316009-12, channel 7. One short-delay trial, final 500 ms before GO. '
               'The 494 colored points are the unchanged saved lag coordinates '
               '[x(t), x(t-3 ms), x(t-6 ms)]; color indicates the actual reach direction. '
               'Black curves are the existing annular-band fit outer and inner ellipses, '
               'not persistence generators. All six examples use identical axis limits, '
               'aspect ratio, viewing angle, opacity, and point size. No per-example rescaling, '
               'rotation, centering, projection, or filtering was applied. '
               'Selections illustrate contrasting individual trials, not typical conditions '
               'or evidence of direction separation. ')
    for row, boundary in zip(selected.itertuples(), boundaries):
        cloud = clouds[row.row_index]
        fig = make_panel(cloud, boundary, int(row.direction), int(row.original_trial_number), limits)
        plotted = np.column_stack(fig.axes[0].collections[0]._offsets3d)
        np.testing.assert_array_equal(plotted, cloud)
        assert fig.axes[0].get_xlim() == fig.axes[0].get_ylim() == fig.axes[0].get_zlim() == limits
        target = OUTPUT / 'figures' / row.png
        fig.savefig(target, dpi=600, metadata={'Software': 'NeuralFieldManifold saved-results renderer'})
        plt.close(fig)
        paths.append(target)
        (OUTPUT / 'captions' / f'{target.stem}.txt').write_text(
            caption + f'Original trial {row.original_trial_number}, direction {row.direction}. '
            f'Selection: {row.selection_reason}; value {row.selection_value:.6g}.\n')
        points = pd.DataFrame(cloud, columns=['x_t', 'x_t_minus_3ms', 'x_t_minus_6ms'])
        points.insert(0, 'point_index', np.arange(494))
        points.insert(0, 'original_trial_number', row.original_trial_number)
        points.insert(0, 'row_index', row.row_index)
        plot_data.append(points)
    pd.concat(plot_data, ignore_index=True).to_csv(OUTPUT / 'tables/plotted_coordinates.csv', index=False)
    make_preview(paths, OUTPUT / 'figures')
    details = '\n'.join(f'- Trial {r.original_trial_number}, direction {r.direction}: '
                        f'{r.selection_reason.lower()} ({r.selection_value:.3f}).'
                        for r in selected.itertuples())
    (OUTPUT / 'README.md').write_text(
        '# Selected geometry examples\n\n'
        'Six contrasting single-trial examples selected from the 197 usable saved fits. '
        'Selection uses minimum/maximum normalized H1 lifetime, R2/R1, and R1, in that order, '
        'without replacement. Ties favor the lower original trial number. '
        'Direction labels and decoding outcomes play no role in selection. '
        'No new persistence, fitting, or decoding was performed.\n\n'
        + details + '\n\n'
        'All plots preserve the original cloud coordinates and existing inner/outer fitted ellipses; '
        'the view and scale are common. Normalized H1 lifetime is death minus birth divided by '
        'centered cloud RMS radius; it is used only for example selection, not to rescale the plots. '
        'Radii are in the existing normalized-signal units, not raw LFP voltage. '
        'These deliberately selected extremes show within-recording variability, not typical '
        'geometry for a direction or statistically distinct topologies. The fitted annular '
        'shape is shared, so different-looking examples need not have different topology.\n\n'
        'Individual panels and full-resolution contact sheet are 600-dpi PNGs. '
        'Preview columns: loop persistence, elongation, size; selected minima on the top row '
        'and maxima on the bottom row (for R2/R1, minimum means most elongated). '
        'All original analyses remain unchanged.\n')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError(f'Protected file changed: {path}')
    forbidden = [name for name in sys.modules
                 if name == 'ripser' or name.startswith(('ripser.', 'sklearn.', 'NeuralFieldManifold.fits'))]
    if forbidden:
        raise RuntimeError(f'Unexpected analysis import: {forbidden}')
    shutil.copy2(__file__, OUTPUT / 'source' / Path(__file__).name)
    (OUTPUT / 'provenance.json').write_text(json.dumps({
        'plot_only': True, 'selection_rules': RULES,
        'renderer_sha256': digest(__file__),
        'helper_sha256': digest(Path(__file__).with_name('plot_prego_persistent_homology.py')),
        'protected_files_verified_unchanged': len(protected), 'protected_sha256': protected,
        'limits_all_axes': limits, 'elevation': 25, 'azimuth': -60,
        'points_per_trial': 494, 'dpi': 600,
        'output_sha256': {p.name: digest(p) for p in (OUTPUT / 'figures').glob('*.png')},
    }, indent=2) + '\n')
    print(selected[['example', 'original_trial_number', 'direction', 'selection_reason', 'selection_value']].to_string(index=False))
    print(f'{len(protected)} protected files unchanged. PNGs: {OUTPUT / "figures"}')


if __name__ == '__main__':
    generate()
