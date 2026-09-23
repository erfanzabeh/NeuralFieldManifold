"""Independent numerical/PNG audit; only reads existing results (no Ripser import)."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1'


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def verify():
    trials = pd.read_csv(SOURCE / 'inputs/selected_trials.csv')
    measures = pd.read_csv(ROOT / 'tables/trial_measurements.csv')
    intervals = pd.read_csv(ROOT / 'tables/persistence_intervals.csv', float_precision='round_trip')
    curves = pd.read_csv(ROOT / 'tables/betti_curves.csv', float_precision='round_trip')
    summary = pd.read_csv(ROOT / 'tables/betti_summary.csv')
    distributions = pd.read_csv(ROOT / 'tables/distribution_summary.csv')
    clouds = np.load(SOURCE / 'clouds.npz', allow_pickle=False)['clouds']
    assert len(trials) == len(measures) == 198
    assert len(list((ROOT / 'checkpoints').glob('trial_*.json'))) == 198
    np.testing.assert_array_equal(trials.row_index, measures.row_index)
    np.testing.assert_array_equal(trials.original_trial_number, measures.original_trial_number)
    for row, cloud in enumerate(clouds):
        m = measures.iloc[row]
        meta = json.loads((ROOT / 'checkpoints' / f'trial_{row:03d}.json').read_text())
        assert meta['status'] == m.status
        if meta['status'] != 'success':
            assert pd.isna(m.longest_H1_lifetime) and pd.notna(m.failure_reason)
            assert not (curves.row_index == row).any()
            continue
        path = ROOT / 'diagrams' / f'trial_{row:03d}.npz'
        assert checksum(path) == meta['diagram_sha256']
        rms = np.linalg.norm(cloud - np.average(cloud, axis=0), 'fro') / np.sqrt(494)
        np.testing.assert_allclose(m.centered_rms, rms)
        with np.load(path, allow_pickle=False) as data:
            h1 = data['H1']
            finite = h1[np.isfinite(h1).all(axis=1)]
            lifetime = 0. if len(h1) == 0 else np.max(finite[:, 1]-finite[:, 0]) if len(finite) else np.nan
            np.testing.assert_allclose(m.longest_H1_lifetime, lifetime)
            np.testing.assert_allclose(m.normalized_H1_lifetime, lifetime/rms)
            for h in range(3):
                diagram = data[f'H{h}']
                table = intervals[(intervals.row_index == row) & (intervals.homology_dimension == h)]
                np.testing.assert_allclose(table[['birth', 'death']], diagram, atol=1e-14)
                sampled = curves[(curves.row_index == row) & (curves.homology_dimension == h)]
                assert len(sampled) == 512
                # Independent event-count implementation, not the analysis broadcast expression.
                counts = (np.searchsorted(np.sort(diagram[:, 0]), sampled.distance, side='right') -
                          np.searchsorted(np.sort(diagram[:, 1]), sampled.distance, side='right'))
                np.testing.assert_array_equal(sampled.betti, counts)
                if h == 0:
                    assert sampled.betti.iloc[0] == 494
                    assert sampled.betti.iloc[-1] == 1
    for (h, d), part in curves.groupby(['homology_dimension', 'direction']):
        pivot = part.pivot(index='row_index', columns='grid_index', values='betti').to_numpy()
        saved = summary[(summary.homology_dimension == h) & (summary.direction == d)].sort_values('grid_index')
        np.testing.assert_allclose(saved['mean'], pivot.mean(axis=0))
        np.testing.assert_allclose(saved[['q25', 'median', 'q75']].to_numpy().T,
                                   np.quantile(pivot, [.25, .5, .75], axis=0))
        assert (saved.n_trials == len(pivot)).all()
    for record in distributions.itertuples(index=False):
        valid = measures.geometry_usable if record.measure in ('R1', 'R2') else measures.status == 'success'
        values = measures.loc[valid & (measures.direction == record.direction), record.measure].dropna()
        assert record.n == len(values)
        np.testing.assert_allclose([record.q25, record.median, record.q75], values.quantile([.25, .5, .75]))
    assert measures.geometry_usable.sum() == 197
    assert len(measures[measures.original_trial_number == 128]) == 1
    assert not bool(measures.loc[measures.original_trial_number == 128, 'geometry_usable'].iloc[0])
    chosen = pd.read_csv(ROOT / 'tables/example_trials.csv')
    np.testing.assert_array_equal(chosen.original_trial_number,
                                  trials.groupby('direction').original_trial_number.min())
    plot_manifest = json.loads((ROOT / 'plot_manifest.json').read_text())
    assert len(plot_manifest['outputs']) == 19
    inspected = []
    for name, metadata in plot_manifest['outputs'].items():
        path = ROOT / 'figures' / f'{name}.png'
        assert checksum(path) == metadata['sha256']
        image = Image.open(path)
        np.testing.assert_allclose(image.info['dpi'], [600, 600], atol=.01)
        assert image.width >= 1900 and image.height >= 1700
        pixel = np.array(image.convert('RGB'))
        assert np.std(pixel) > 5 and (pixel < 200).mean() > .005
        inspected.append(dict(name=name, width=image.width, height=image.height, dpi=image.info['dpi']))
        data = pd.read_csv(ROOT / 'tables' / f'plot_{name}.csv')
        if name in ('loop_lifetime', 'loop_lifetime_normalized', 'radius_r1', 'radius_r2'):
            key = {'loop_lifetime': 'longest_H1_lifetime', 'loop_lifetime_normalized': 'normalized_H1_lifetime',
                   'radius_r1': 'R1', 'radius_r2': 'R2'}[name]
            np.testing.assert_allclose(data[key], measures.set_index('row_index').loc[data.row_index, key])
        if name.startswith('geometry_direction_'):
            direction = int(name.rsplit('_', 1)[1])
            row = int(chosen.loc[chosen.direction == direction, 'row_index'].iloc[0])
            np.testing.assert_allclose(data[['x_t', 'x_t_minus3ms', 'x_t_minus6ms']], clouds[row], atol=1e-14)
    assert not list((ROOT / 'figures').glob('*.pdf')) and not list((ROOT / 'figures').glob('*.svg'))
    protected = json.loads((ROOT / 'protected_manifest.json').read_text())
    for path, expected in protected.items():
        assert checksum(path) == expected, path
    result = dict(passed=True, trials=len(measures), successes=int((measures.status == 'success').sum()),
                  failures=int((measures.status != 'success').sum()), radius_trials=197,
                  csv_intervals=len(intervals), betti_rows=len(curves), png_checks=inspected,
                  protected_files_unchanged=len(protected), independent_event_count_validation=True)
    (ROOT / 'validation/independent_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'png_checks'}, indent=2))


if __name__ == '__main__':
    verify()
