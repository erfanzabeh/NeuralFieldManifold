"""Independent, checkpointed PH of frozen observed pre-GO clouds; no fitting/decoding."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / 'outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1'
OUTPUT = UNIT / 'outputs/prego_T_y070316009_12_ch7_m3_tau3_persistent_homology_v1'
PARAMETERS = dict(maxdim=2, coeff=2, thresh=float('inf'), n_perm=None,
                  metric='euclidean', distance_matrix=False, do_cocycles=False)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.shape).encode() + x.dtype.str.encode() + x.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen configuration or provenance changed: {path}')
    else:
        write_json(path, value)


def check_calculation_unchanged(original, current):
    def calculation_tree(source):
        nodes = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name not in (
                'initialize', 'verify', 'verify_table', 'verify_analysis_source', 'check_calculation_unchanged'):
                nodes[node.name] = ast.dump(node)
            elif isinstance(node, ast.Assign):
                nodes[ast.unparse(node.targets[0])] = ast.dump(node)
        return nodes
    if calculation_tree(original) != calculation_tree(current):
        raise ValueError('Calculation changed: use a new experiment, not existing checkpoints')


def verify_analysis_source():
    provenance = json.loads((OUTPUT / 'analysis_provenance.json').read_text())
    original_path = OUTPUT / 'source' / Path(__file__).name
    original_hash = provenance['analysis_source'][str(Path(__file__).resolve())]
    assert digest(original_path) == original_hash, 'Frozen initial source changed'
    if digest(__file__) != original_hash:
        check_calculation_unchanged(original_path.read_text(), Path(__file__).read_text())
        revision = digest(__file__)
        snapshot = OUTPUT / 'source' / f'verification_revision_{revision}.py'
        if not snapshot.exists():
            shutil.copy2(__file__, snapshot)
        assert digest(snapshot) == revision
        immutable_json(snapshot.with_suffix('.json'), dict(
            original_source_sha256=original_hash, revised_source_sha256=revision,
            calculation_ast_unchanged=True,
            reason='Missing-value CSV validation and exact example-selection verification; no calculation changes'))


def verify_table(path, reference):
    saved = pd.read_csv(path, float_precision='round_trip')
    reference = reference.copy()
    for column in reference:
        if reference[column].dtype == object:
            saved[column] = saved[column].fillna('')
            reference[column] = reference[column].fillna('')
    pd.testing.assert_frame_equal(saved, reference, check_dtype=False, atol=1e-12, rtol=1e-10)


def environment():
    return dict(python=platform.python_version(), machine=platform.machine(),
                packages={k: importlib.metadata.version(k)
                          for k in ('ripser', 'numpy', 'scipy', 'pandas', 'scikit-learn')})


def load_inputs():
    config = json.loads((SOURCE / 'config.json').read_text())
    alignment = json.loads((SOURCE / 'inputs/alignment.json').read_text())
    assert (config['m'], config['tau_ms'], config['fs_hz']) == (3, 3, 1000)
    assert config['window_ms'] == [-500, 0]
    assert alignment['sampling_rate_hz'] == 1000
    assert alignment['end_sample_exclusive'] == alignment['go_sample_zero_based']
    assert alignment['end_sample_exclusive'] - alignment['start_sample'] == 500
    with np.load(SOURCE / 'clouds.npz', allow_pickle=False) as saved:
        clouds = saved['clouds']
        np.testing.assert_array_equal(saved['latest_sample_index'], np.arange(6, 500))
    trials = pd.read_csv(SOURCE / 'inputs/selected_trials.csv')
    np.testing.assert_array_equal(trials.row_index, np.arange(198))
    assert trials.original_trial_number.is_unique and trials.raw_trial_index.is_unique
    assert (trials.delay == 'short').all()
    np.testing.assert_array_equal(trials.groupby('direction').size().reindex(range(1, 7)), [33]*6)
    with np.load(SOURCE / 'inputs/frozen_epochs.npz', allow_pickle=False) as frozen:
        assert frozen['processed'].shape == (198, 500)
        for column, start in enumerate((6, 3, 0)):
            np.testing.assert_array_equal(clouds[:, :, column], frozen['processed'][:, start:start+494])
        for column, key in [('direction', 'labels'), ('original_trial_number', 'original_trial_number'),
                            ('raw_trial_index', 'raw_trial_indices'), ('heldout_fold_zero_based', 'folds')]:
            np.testing.assert_array_equal(trials[column], frozen[key])
        np.testing.assert_array_equal(frozen['time_ms'], np.arange(-500, 0))
    assert clouds.shape == (198, 494, 3) and np.isfinite(clouds).all()
    return clouds, trials


def pilot_rows(trials):
    return trials.sort_values('original_trial_number').head(6).row_index.astype(int).tolist()


def example_rows(trials):
    return trials.sort_values('original_trial_number').groupby('direction').first().row_index.astype(int).tolist()


def compute_diagrams(cloud):
    # Lazy import keeps validation of frozen summaries separate from computation.
    import ripser
    if ripser.__version__ != '0.6.15':
        raise ValueError('This experiment requires Ripser.py 0.6.15')
    cloud = np.asarray(cloud)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError('Expected finite observed coordinates in three dimensions')
    return ripser.ripser(cloud, **PARAMETERS)['dgms']


def centered_rms(cloud):
    return float(np.sqrt(np.mean(np.sum((cloud - cloud.mean(axis=0))**2, axis=1))))


def longest_finite_lifetime(diagram):
    if len(diagram) == 0:
        return 0.0
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    return float(np.max(finite[:, 1] - finite[:, 0])) if len(finite) else float('nan')


def betti_curve(diagram, grid):
    return ((diagram[:, :1] <= grid) & (grid < diagram[:, 1:2])).sum(axis=0)


def validate_diagrams(diagrams, n_points):
    assert len(diagrams) == 3
    for h, d in enumerate(diagrams):
        assert d.ndim == 2 and d.shape[1] == 2
        assert np.isfinite(d[:, 0]).all() and not np.isnan(d).any()
        assert (d[:, 0] >= 0).all() and (d[:, 1] > d[:, 0]).all()
        if h:
            assert np.isfinite(d).all(), 'Full finite-cloud filtration must kill H1/H2'
    assert np.isinf(diagrams[0][:, 1]).sum() == 1
    assert len(diagrams[0]) <= n_points


def checkpoint_paths(row):
    return (OUTPUT / 'checkpoints' / f'trial_{row:03d}.json',
            OUTPUT / 'diagrams' / f'trial_{row:03d}.npz')


def checkpoint(row, cloud, trial):
    meta_path, data_path = checkpoint_paths(row)
    meta = json.loads(meta_path.read_text())
    assert meta['row_index'] == row
    assert meta['original_trial_number'] == int(trial.original_trial_number)
    assert meta['cloud_sha256'] == array_digest(cloud)
    assert meta['config_sha256'] == digest(OUTPUT / 'config.json')
    assert meta['status'] in ('success', 'failed', 'timeout')
    diagrams = None
    if meta['status'] == 'success':
        assert digest(data_path) == meta['diagram_sha256']
        with np.load(data_path, allow_pickle=False) as data:
            diagrams = [data[f'H{h}'] for h in range(3)]
        validate_diagrams(diagrams, len(cloud))
    return meta, diagrams


def initialize(clouds, trials):
    for folder in ('checkpoints', 'diagrams', 'inputs', 'tables', 'logs', 'source', 'validation'):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    immutable_json(OUTPUT / 'config.json', dict(
        recording='monkeyT_session-y070316009-12_lfp-7', source=str(SOURCE),
        n_trials=198, per_direction=33, n_points=494, m=3, tau_samples=3, tau_ms=3, fs_hz=1000,
        window_ms=[-500, 0], ripser_version='0.6.15',
        ripser_parameters=dict(PARAMETERS, thresh='infinity'), pilot_rows=pilot_rows(trials),
        example_rows=example_rows(trials), timeout_seconds=600, sequential=True,
        input_transform='none', geometric_refitting=False, decoding=False,
        loop_summary='longest finite H1 lifetime; zero only for empty successful H1',
        normalized_summary='lifetime / centered cloud RMS radius; input unchanged',
        betti_grid_points=512, betti_interval='birth <= distance < death',
        empty_dimension_grid='H0 finite-death maximum only if dimension has no finite deaths',
        inference='descriptive; no tests or threshold-based counts'))
    immutable_json(OUTPUT / 'environment.json', environment())
    if environment()['packages']['ripser'] != '0.6.15':
        raise ValueError('Ripser version mismatch')
    protected = OUTPUT / 'protected_manifest.json'
    if not protected.exists():
        paths = [p for p in (UNIT / 'outputs').rglob('*') if p.is_file()
                 and OUTPUT not in p.parents and '__pycache__' not in p.parts]
        paths += [p for p in UNIT.glob('*.py') if p.name not in (
            'run_prego_persistent_homology.py', 'plot_prego_persistent_homology.py')]
        paths += list((UNIT.parents[3] / 'NeuralFieldManifold/fits').glob('*.py'))
        write_json(protected, {str(p): digest(p) for p in sorted(set(paths))})
    verify_protected()
    provenance_path = OUTPUT / 'analysis_provenance.json'
    original_source = (json.loads(provenance_path.read_text())['analysis_source'] if provenance_path.exists()
                       else {str(Path(__file__).resolve()): digest(__file__)})
    immutable_json(provenance_path, {
        'analysis_source': original_source,
        'inputs': {str(SOURCE / p): digest(SOURCE / p) for p in (
            'clouds.npz', 'inputs/frozen_epochs.npz', 'inputs/selected_trials.csv',
            'inputs/alignment.json', 'tables/fit_diagnostics.csv', 'tables/features_geometry.csv')},
        'cloud_hashes': [array_digest(x) for x in clouds]})
    snapshot = OUTPUT / 'source' / Path(__file__).name
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    verify_analysis_source()
    for source, destination in [(SOURCE / 'inputs/selected_trials.csv', OUTPUT / 'inputs/selected_trials.csv'),
                                (SOURCE / 'inputs/alignment.json', OUTPUT / 'inputs/alignment.json')]:
        if destination.exists():
            assert digest(source) == digest(destination)
        else:
            shutil.copy2(source, destination)


def verify_protected():
    manifest = json.loads((OUTPUT / 'protected_manifest.json').read_text())
    changed = [p for p, value in manifest.items() if not Path(p).exists() or digest(p) != value]
    if changed:
        raise ValueError(f'Protected files changed: {changed[:10]}')
    return len(manifest)


def worker(row):
    clouds, trials = load_inputs()
    cloud, trial = clouds[row], trials.iloc[row]
    meta_path, data_path = checkpoint_paths(row)
    started = time.monotonic()
    meta = dict(row_index=row, original_trial_number=int(trial.original_trial_number),
                cloud_sha256=array_digest(cloud), config_sha256=digest(OUTPUT / 'config.json'),
                n_points=len(cloud))
    try:
        diagrams = compute_diagrams(cloud)
        validate_diagrams(diagrams, len(cloud))
        temporary = data_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **{f'H{h}': d for h, d in enumerate(diagrams)})
        temporary.replace(data_path)
        meta.update(status='success', reason='', diagram_sha256=digest(data_path))
    except Exception:
        meta.update(status='failed', reason=traceback.format_exc())
    meta['elapsed_seconds'] = time.monotonic() - started
    write_json(meta_path, meta)


def run_trial(row, clouds, trials):
    meta_path, _ = checkpoint_paths(row)
    if meta_path.exists():
        return checkpoint(row, clouds[row], trials.iloc[row])[0], True
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    status, reason = 'failed', ''
    with (OUTPUT / 'logs' / f'trial_{row:03d}.log').open('w') as log:
        try:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                     '--worker-row', str(row)], stdout=log, stderr=log,
                                    env=env, timeout=600, check=False)
            reason = f'Worker exit code {result.returncode}; see logs/trial_{row:03d}.log'
        except subprocess.TimeoutExpired:
            status, reason = 'timeout', 'Exceeded predeclared 600-second per-trial limit; no approximation'
    if not meta_path.exists():
        write_json(meta_path, dict(row_index=row,
            original_trial_number=int(trials.iloc[row].original_trial_number),
            cloud_sha256=array_digest(clouds[row]), config_sha256=digest(OUTPUT / 'config.json'),
            status=status, reason=reason, n_points=494, elapsed_seconds=time.monotonic()-started))
    return checkpoint(row, clouds[row], trials.iloc[row])[0], False


def build_tables(clouds, trials):
    diagnostics = pd.read_csv(SOURCE / 'tables/fit_diagnostics.csv')
    radii = pd.read_csv(SOURCE / 'tables/features_geometry.csv')
    for table in (diagnostics, radii):
        for key in ('row_index', 'original_trial_number', 'direction'):
            np.testing.assert_array_equal(table[key], trials[key])
    usable = diagnostics.usable.to_numpy(dtype=bool)
    np.testing.assert_array_equal(np.isfinite(radii[['R1', 'R2']]).all(axis=1), usable)
    assert usable.sum() == 197
    records, diagram_rows, diagrams = [], [], {}
    for row, trial in trials.iterrows():
        meta, d = checkpoint(row, clouds[row], trial)
        item = dict(trial, status=meta['status'], failure_reason=meta['reason'],
                    elapsed_seconds=meta['elapsed_seconds'], centered_rms=centered_rms(clouds[row]),
                    geometry_usable=bool(usable[row]), R1=radii.iloc[row].R1, R2=radii.iloc[row].R2,
                    longest_H1_lifetime=np.nan, normalized_H1_lifetime=np.nan)
        if d is not None:
            diagrams[row] = d
            item['longest_H1_lifetime'] = longest_finite_lifetime(d[1])
            item['normalized_H1_lifetime'] = (item['longest_H1_lifetime']/item['centered_rms']
                                               if item['centered_rms'] > 0 else np.nan)
            for h, bars in enumerate(d):
                item[f'H{h}_intervals'] = len(bars)
                item[f'H{h}_infinite'] = int(np.isinf(bars[:, 1]).sum())
                for j, (birth, death) in enumerate(bars):
                    diagram_rows.append(dict(row_index=row, original_trial_number=int(trial.original_trial_number),
                        direction=int(trial.direction), homology_dimension=h, interval=j,
                        birth=birth, death=death, lifetime=death-birth))
        records.append(item)
    measurements = pd.DataFrame(records)
    intervals = pd.DataFrame(diagram_rows)
    if not diagrams:
        raise ValueError('No successful diagrams to summarize')
    curve_rows, summary_rows, grid_notes = [], [], {}
    for h in range(3):
        deaths = [d[h][:, 1][np.isfinite(d[h][:, 1])] for d in diagrams.values()]
        finite = np.concatenate(deaths)
        fallback = len(finite) == 0
        extent = float(finite.max()) if len(finite) else float(max(
            d[0][np.isfinite(d[0][:, 1]), 1].max() for d in diagrams.values()))
        grid = np.linspace(0, extent, 512)
        grid_notes[f'H{h}'] = dict(max_distance=extent, fallback_to_H0=fallback)
        curves = {}
        for row, d in diagrams.items():
            curves[row] = betti_curve(d[h], grid)
            curve_rows.extend(dict(row_index=row, direction=int(trials.iloc[row].direction),
                                   homology_dimension=h, grid_index=j, distance=float(e), betti=int(b))
                              for j, (e, b) in enumerate(zip(grid, curves[row])))
        for direction in range(1, 7):
            values = np.array([curve for row, curve in curves.items() if trials.iloc[row].direction == direction])
            if not len(values):
                continue
            for j, e in enumerate(grid):
                q1, median, q3 = np.quantile(values[:, j], [.25, .5, .75])
                summary_rows.append(dict(homology_dimension=h, direction=direction, grid_index=j,
                    distance=e, n_trials=len(values), mean=values[:, j].mean(),
                    q25=q1, median=median, q75=q3))
    distributions = []
    for measure in ('longest_H1_lifetime', 'normalized_H1_lifetime', 'R1', 'R2'):
        for direction in range(1, 7):
            keep = measurements.direction == direction
            keep &= measurements.geometry_usable if measure in ('R1', 'R2') else measurements.status == 'success'
            values = measurements.loc[keep, measure].dropna().to_numpy()
            if len(values):
                q1, med, q3 = np.quantile(values, [.25, .5, .75])
                distributions.append(dict(measure=measure, direction=direction, n=len(values),
                    mean=values.mean(), q25=q1, median=med, q75=q3, minimum=values.min(), maximum=values.max()))
    return dict(trial_measurements=measurements, persistence_intervals=intervals,
                betti_curves=pd.DataFrame(curve_rows), betti_summary=pd.DataFrame(summary_rows),
                distribution_summary=pd.DataFrame(distributions)), grid_notes


def summarize(clouds, trials):
    tables, grid_notes = build_tables(clouds, trials)
    for name, table in tables.items():
        table.to_csv(OUTPUT / 'tables' / f'{name}.csv', index=False)
    write_json(OUTPUT / 'grid_definitions.json', grid_notes)
    selected = trials.set_index('row_index').loc[example_rows(trials)].reset_index()
    selected.to_csv(OUTPUT / 'tables/example_trials.csv', index=False)
    return tables


def verify():
    clouds, trials = load_inputs()
    expected, grids = build_tables(clouds, trials)
    for name, table in expected.items():
        verify_table(OUTPUT / 'tables' / f'{name}.csv', table)
    verify_table(OUTPUT / 'tables/example_trials.csv',
                 trials.set_index('row_index').loc[example_rows(trials)].reset_index())
    assert json.loads((OUTPUT / 'grid_definitions.json').read_text()) == grids
    config = json.loads((OUTPUT / 'config.json').read_text())
    assert config['ripser_parameters'] == dict(PARAMETERS, thresh='infinity')
    assert json.loads((OUTPUT / 'environment.json').read_text()) == environment()
    provenance = json.loads((OUTPUT / 'analysis_provenance.json').read_text())
    verify_analysis_source()
    for path, value in provenance['inputs'].items():
        assert digest(path) == value, path
    for i, x in enumerate(clouds):
        assert array_digest(x) == provenance['cloud_hashes'][i]
    protected = verify_protected()
    measures = expected['trial_measurements']
    report = dict(passed=True, trials=198, direction_counts=trials.direction.value_counts().sort_index().to_dict(),
        successes=int((measures.status == 'success').sum()), failures=int((measures.status != 'success').sum()),
        radius_usable=int(measures.geometry_usable.sum()), protected_files_unchanged=protected,
        all_saved_tables_reproduced=True, clouds_exactly_match_frozen_epochs=True,
        infinite_intervals_preserved=True, decoder_calls=0, geometric_fit_calls=0,
        limitations='Descriptive single-recording PH; no topology significance or parameter retuning.')
    write_json(OUTPUT / 'validation/numerical_checks.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'all'), default='pilot')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--worker-row', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_row is not None:
        if not 0 <= args.worker_row < 198:
            parser.error('Invalid trial row')
        worker(args.worker_row)
        return
    if args.verify_only:
        print(json.dumps(verify(), indent=2), flush=True)
        return
    clouds, trials = load_inputs()
    initialize(clouds, trials)
    pilot = pilot_rows(trials)
    records, reused = [], 0
    for row in pilot:
        result, old = run_trial(row, clouds, trials)
        reused += int(old)
        records.append(result)
        print(f'Pilot original trial {result["original_trial_number"]}: {result["status"]}, '
              f'{result["elapsed_seconds"]:.2f}s' + (' (saved)' if old else ''), flush=True)
        if result['status'] != 'success':
            write_json(OUTPUT / 'run_status.json', dict(status='pilot_stopped', results=records))
            verify_protected()
            raise SystemExit('Pilot did not complete; stopping without approximation')
    timing = [r['elapsed_seconds'] for r in records]
    write_json(OUTPUT / 'pilot.json', dict(rows=pilot, original_trial_numbers=[r['original_trial_number'] for r in records],
        elapsed_seconds=timing, projected_198_compute_seconds=float(np.mean(timing)*198), all_success=True))
    if args.phase == 'pilot':
        verify_protected()
        return
    for n, row in enumerate([r for r in range(198) if r not in pilot], 1):
        result, old = run_trial(row, clouds, trials)
        reused += int(old)
        if n % 10 == 0 or result['status'] != 'success' or n == 192:
            print(f'Accounted {n+6}/198; last original trial {result["original_trial_number"]}: '
                  f'{result["status"]}, {result["elapsed_seconds"]:.2f}s', flush=True)
    tables = summarize(clouds, trials)
    verified = verify()
    write_json(OUTPUT / 'run_status.json', dict(status='complete', reused_checkpoints=reused, **verified))
    print(tables['distribution_summary'].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
