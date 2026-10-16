"""Run the fixed single-recording experiment, without plotting or parameter search."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from motor_lfp_utils import BANDS, lag_embed
from prego_decoding import spectral_features, permute_within_folds
from prego_fixed_geometry_decoding import (
    OUTPUT, METHODS, DIMENSIONS, GEOMETRY_COLUMNS, geometry_features, feature_sets,
    decode, score_predictions, bootstrap_predictions, permutation_result,
)
from prego_geometric_fits import INPUT, RECORDING, UNIT, array_hash, sha256, write_json
from run_prego_geometric_fits import initialize as initialize_saved
from run_prego_geometric_fits import load_inputs, validate_checkpoint, worker

SOURCE = UNIT/'outputs/prego_T_y070316009_12_ch7_tau16_geometric_fits_v1'


def numerical_environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__, machine=platform.machine())


def initialize(root, config, provenance, resume):
    root = Path(root)
    environment = root/'numerical_environment.json'
    if root.exists() and resume:
        if not environment.exists() or json.loads(environment.read_text()) != numerical_environment():
            raise ValueError('Numerical environment changed; use a new experiment')
    initialize_saved(root, config, provenance, resume)
    if not environment.exists():
        write_json(environment, numerical_environment())


def frozen_clouds(processed):
    if processed.shape != (198, 500) or not np.isfinite(processed).all():
        raise ValueError('Expected 198 finite frozen 500-sample epochs')
    clouds = np.stack([lag_embed(x, dim=3, tau=3) for x in processed])
    assert clouds.shape == (198, 494, 3)
    for column, offset in enumerate((6, 3, 0)):
        np.testing.assert_array_equal(clouds[:, :, column], processed[:, offset:offset+494])
    return clouds


def validate_null(saved, labels, folds, indices):
    np.testing.assert_array_equal(saved['index'], indices)
    for row, index in enumerate(indices):
        shuffled = permute_within_folds(labels, folds, 52000+int(index))
        np.testing.assert_array_equal(saved['labels'][row], shuffled)
        assert saved['predictions'][row].shape == (len(labels), 5)
        scores = score_predictions(shuffled, saved['predictions'][row])
        for key in ('macro_f1', 'accuracy'):
            np.testing.assert_allclose(saved[key][row], scores[key], atol=1e-14)


def write_tables(root, trials, features, predictions, bootstrap, null):
    labels = trials.direction.to_numpy()
    scores = score_predictions(labels, predictions)
    summary, directions, prediction_rows = [], [], []
    for j, (method, dimension) in enumerate(zip(METHODS, DIMENSIONS)):
        row = dict(method=method, dimensions=dimension, n_trials=len(labels))
        for key in ('macro_f1', 'accuracy'):
            low, high = np.quantile(bootstrap[key][:, j], [.025, .975])
            row.update({key: scores[key][j], key+'_ci_low': low, key+'_ci_high': high})
        lo, median, hi = np.quantile(null['macro_f1'][:, j], [.025, .5, .975])
        row.update(null_f1_low=lo, null_f1_median=median, null_f1_high=hi,
                   permutation_p=(1+np.count_nonzero(null['macro_f1'][:, j] >= scores['macro_f1'][j]))/1001
                   if method == 'geometry' else np.nan)
        summary.append(row)
        for d in range(6):
            low, high = np.quantile(bootstrap['per_direction_f1'][:, j, d], [.025, .975])
            directions.append(dict(method=method, direction=d+1, f1=scores['per_direction_f1'][j, d],
                                   ci_low=low, ci_high=high, n_true=int((labels == d+1).sum())))
        for key in ('confusion_counts', 'confusion_fraction'):
            pd.DataFrame(scores[key][j], index=np.arange(1, 7), columns=np.arange(1, 7)).rename_axis(
                'true_direction').to_csv(root/'tables'/f'{key}_{method}.csv')
        for i, trial in enumerate(trials.to_dict('records')):
            prediction_rows.append(dict(trial, method=method, predicted_direction=int(predictions[i, j])))
        pd.concat([trials.reset_index(drop=True), pd.DataFrame(features[method], columns=(GEOMETRY_COLUMNS
                   if method == 'geometry' else [f'{method}_{k+1}' for k in range(dimension)]))], axis=1).to_csv(
                       root/'tables'/f'features_{method}.csv', index=False)
    pd.DataFrame(summary).to_csv(root/'tables/summary.csv', index=False)
    pd.DataFrame(directions).to_csv(root/'tables/per_direction.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(root/'tables/heldout_predictions.csv', index=False)
    pd.DataFrame(null['macro_f1'], columns=METHODS).rename_axis('permutation_index').to_csv(root/'tables/null_macro_f1.csv')
    pd.DataFrame(bootstrap['macro_f1'], columns=METHODS).rename_axis('resample_index').to_csv(root/'tables/bootstrap_macro_f1.csv')
    return pd.DataFrame(summary)


def verify(root):
    root = Path(root)
    if json.loads((root/'numerical_environment.json').read_text()) != numerical_environment():
        raise ValueError('Numerical environment changed')
    config = json.loads((root/'config.json').read_text())
    assert config['geometry_columns'] == GEOMETRY_COLUMNS and config['tau_ms'] == 3
    frozen = np.load(root/'inputs/frozen_epochs.npz')
    data, trials, _ = load_inputs(INPUT)
    with np.load(SOURCE/'clouds.npz') as source:
        np.testing.assert_array_equal(frozen['processed'], source['processed'])
    for field in data:
        np.testing.assert_array_equal(frozen[field], data[field])
    clouds = frozen_clouds(frozen['processed'])
    np.testing.assert_array_equal(np.load(root/'clouds.npz')['clouds'], clouds)
    features_file = np.load(root/'features.npz')
    features = {method: features_file[method] for method in METHODS}
    geometry, flags, reasons = [], [], []
    for i in range(198):
        result = validate_checkpoint(root/'checkpoints'/f'trial_{i:03d}_one_torus.json', i, 'one_torus', clouds[i])
        vector, reason = geometry_features(result)
        geometry.append(vector)
        flags.append(result.get('summary', {}).get('flags', 'fit_failed'))
        reasons.append(reason)
    np.testing.assert_allclose(features['geometry'], geometry, equal_nan=True)
    assert features['geometry'].shape == (198, 14)
    expected = feature_sets(*spectral_features(data['segments'], 'T'), np.asarray(geometry))
    for method in METHODS:
        np.testing.assert_allclose(features[method], expected[method], equal_nan=True)
    labels, folds = data['labels'], data['folds']
    saved = np.load(root/'heldout_predictions.npz')
    np.testing.assert_array_equal(saved['labels'], labels)
    np.testing.assert_array_equal(saved['folds'], folds)
    predictions = saved['predictions']
    for j, method in enumerate(METHODS):
        for fold in range(5):
            train, test = folds != fold, folds == fold
            model = joblib.load(root/'models'/f'{method}_fold{fold}.joblib')
            assert list(model.named_steps) == ['impute', 'scale', 'lda']
            np.testing.assert_allclose(model['impute'].statistics_, np.nanmedian(features[method][train], axis=0))
            imputed_train = model['impute'].transform(features[method][train])
            np.testing.assert_allclose(model['scale'].mean_, imputed_train.mean(axis=0), atol=1e-14)
            np.testing.assert_array_equal(model.predict(features[method][test]), predictions[test, j])
    metrics = score_predictions(labels, predictions)
    np.testing.assert_allclose(metrics['confusion_fraction'].sum(axis=2), 1)
    null = np.load(root/'permutations.npz')
    validate_null(null, labels, folds, np.arange(1000))
    boot = np.load(root/'bootstrap.npz')
    assert boot['indices'].shape == (2000, 198)
    for k, indices in enumerate(boot['indices']):
        np.testing.assert_array_equal(np.bincount(labels[indices], minlength=7)[1:], np.full(6, 33))
        scores = score_predictions(labels[indices], predictions[indices])
        for name in ('macro_f1', 'accuracy', 'per_direction_f1'):
            np.testing.assert_allclose(boot[name][k], scores[name], atol=1e-14)
    summary = pd.read_csv(root/'tables/summary.csv')
    for key in ('macro_f1', 'accuracy'):
        np.testing.assert_allclose(summary[key], metrics[key])
        interval = np.quantile(boot[key], [.025, .975], axis=0)
        np.testing.assert_allclose(summary[key+'_ci_low'], interval[0])
        np.testing.assert_allclose(summary[key+'_ci_high'], interval[1])
    np.testing.assert_allclose(summary.null_f1_median, np.median(null['macro_f1'], axis=0))
    assert np.isclose(summary.permutation_p.iloc[-1], (1+(null['macro_f1'][:, -1] >= metrics['macro_f1'][-1]).sum())/1001)
    for j, method in enumerate(METHODS):
        for key in ('confusion_counts', 'confusion_fraction'):
            np.testing.assert_allclose(pd.read_csv(root/'tables'/f'{key}_{method}.csv', index_col=0), metrics[key][j])
    for name in ('provenance.json', 'prior_outputs_manifest.json'):
        manifest = json.loads((root/name).read_text())
        changed = [path for path, digest in manifest.items() if sha256(path) != digest]
        if changed:
            raise ValueError(f'Changed source or prior result: {changed}')
    report = dict(passed=True, n_trials=198, n_directions=6, trials_per_direction=33, folds=5,
                  n_features=14, embedded_points=494, usable_fits=sum(not r for r in reasons),
                  unusable_fits=sum(bool(r) for r in reasons),
                  unusable_reasons=pd.Series([r for r in reasons if r]).value_counts().to_dict(),
                  active_bound_fits=sum('active_optimizer_bound' in f for f in flags),
                  near_bound_fits=sum('near_optimizer_bound' in f for f in flags),
                  preserved_prior_files=len(json.loads((root/'prior_outputs_manifest.json').read_text())),
                  permutation_count=1000, bootstrap_count=2000,
                  checks=['exact frozen epochs and saved trial folds', '494 original-coordinate points per trial',
                          '14 canonical features; no width', 'all trials receive OOF predictions',
                          'saved estimators have impute/scale/lda only', 'training-only medians and scaling',
                          'saved models reproduce OOF predictions', 'spectral features reproduce existing extraction',
                          'confusion rows sum to one', 'all null and bootstrap metrics reproduce predictions',
                          'source/input hashes and previous results unchanged'])
    write_json(root/'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.output
    if args.verify_only:
        with threadpool_limits(limits=1):
            print(verify(root), flush=True)
        return
    started = time.monotonic()
    data, trials, alignment = load_inputs(INPUT)
    with np.load(SOURCE/'clouds.npz') as frozen:
        processed = frozen['processed']
        for key in ('labels', 'original_trial_number'):
            np.testing.assert_array_equal(frozen[key], data[key])
    clouds = frozen_clouds(processed)
    config = dict(recording=RECORDING, n_trials=198, n_per_direction=33, fs_hz=1000,
                  window_ms=[-500, 0], K=1, m=3, tau_ms=3, points_per_trial=494,
                  geometry_columns=GEOMETRY_COLUMNS, dimensions=list(DIMENSIONS), methods=list(METHODS),
                  fitter='one_torus_fit', lam=.1, hole_ratio=.5, angular_regularization_harmonics=3,
                  max_nfev=6000, loss='huber', width_feature=False, pca=False, pinn=False,
                  geometry_input='Frozen detrended, median/MAD-normalized, zero-phase 2-55 Hz epochs; no re-filtering',
                  spectral_input='Saved raw pre-GO epochs, linear detrend; no geometry normalization or bandpass',
                  spectral_peak=dict(band_hz=[12, 40], window='hamming', nperseg=300, noverlap=200,
                                     nfft=10000, power='log10 peak PSD', frequency='Hz at peak'),
                  spectral_bands=dict(bands_hz=BANDS, window='hann', nperseg=500, noverlap=250,
                                      nfft=10000, power='log10 integrated PSD'),
                  lda=dict(imputation='training median', scaling='training StandardScaler', solver='lsqr', shrinkage='auto'),
                  folds='saved five stratified trial folds; no regeneration',
                  permutations=1000, permutation_seed=52000, permutation_scheme='within each saved fold',
                  bootstraps=2000, bootstrap_seed=42000, bootstrap_scheme='class-stratified held-out predictions',
                  inference='Exploratory single selected recording; delay visually chosen before this run; conditional intervals')
    # JSON round-trip removes tuple/list differences when resuming.
    config = json.loads(json.dumps(config))
    native = Path(importlib.import_module('NeuralFieldManifold.fits.one_torus').__file__)
    sources = [Path(__file__), native]+[UNIT/name for name in (
        'prego_fixed_geometry_decoding.py', 'prego_geometric_fits.py', 'prego_decoding.py',
        'motor_lfp_utils.py', 'run_prego_geometric_fits.py')]
    inputs = [SOURCE/'clouds.npz', SOURCE/'config.json']+[INPUT/name for name in (
        'selected_raw_prego_epochs.npz', 'selected_trials.csv', 'audited_alignment.json', 'full_converted_recording.npz')]
    provenance = {str(p.resolve()): sha256(p) for p in sources+inputs}
    initialize(root, config, provenance, args.resume)
    for folder in ('models', 'null_checkpoints'):
        (root/folder).mkdir(exist_ok=True)
    for path in sources:
        if not (root/'source'/path.name).exists():
            shutil.copy2(path, root/'source'/path.name)
    manifest = root/'prior_outputs_manifest.json'
    if not manifest.exists():
        print('Hashing earlier outputs without modifying them', flush=True)
        write_json(manifest, {str(p.resolve()): sha256(p) for p in (UNIT/'outputs').rglob('*')
                             if p.is_file() and root.resolve() not in p.resolve().parents and '__pycache__' not in p.parts})
    np.savez_compressed(root/'inputs/frozen_epochs.npz', **data, processed=processed)
    trials.to_csv(root/'inputs/selected_trials.csv', index=False)
    write_json(root/'inputs/alignment.json', alignment)
    np.savez_compressed(root/'clouds.npz', clouds=clouds, latest_sample_index=np.arange(6, 500))
    if not (root/'logs/environment.json').exists():
        write_json(root/'logs/environment.json', dict(**numerical_environment(), jobs=args.jobs))
    todo = []
    for trial in trials.to_dict('records'):
        row = int(trial['row_index'])
        path = root/'checkpoints'/f'trial_{row:03d}_one_torus.json'
        if path.exists():
            validate_checkpoint(path, row, 'one_torus', clouds[row])
        else:
            todo.append((row, trial))
    print(f'Fitting {len(todo)} trial clouds: m=3, tau=3 ms, 494 points each', flush=True)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        iterator = Parallel(n_jobs=args.jobs, return_as='generator_unordered')(
            delayed(worker)(root, row, 'one_torus', clouds[row], trial) for row, trial in todo)
        for count, result in enumerate(iterator, 1):
            if count % 25 == 0 or count == len(todo):
                print(f'Fits: {count}/{len(todo)}', flush=True)
    geometry, diagnostics = [], []
    for trial in trials.to_dict('records'):
        row = int(trial['row_index'])
        result = validate_checkpoint(root/'checkpoints'/f'trial_{row:03d}_one_torus.json', row, 'one_torus', clouds[row])
        vector, reason = geometry_features(result)
        geometry.append(vector)
        diagnostics.append(dict(trial, **result.get('summary', {}), fit_status=result['status'],
                                unusable_reason=reason, usable=not bool(reason), warnings=';'.join(result['warnings'])))
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(root/'tables/fit_diagnostics.csv', index=False)
    features = feature_sets(*spectral_features(data['segments'], 'T'), np.asarray(geometry))
    np.savez_compressed(root/'features.npz', **features, labels=data['labels'], folds=data['folds'])
    print(f'Usable fits: {diagnostics.usable.sum()}/198. Running saved-fold LDA.', flush=True)
    labels, folds = data['labels'], data['folds']
    with threadpool_limits(limits=1):
        predictions, models, audit = decode(features, labels, folds, retain_models=True)
    for (method, fold), estimator in models.items():
        joblib.dump(estimator, root/'models'/f'{method}_fold{fold}.joblib')
    write_json(root/'training_audit.json', audit)
    np.savez_compressed(root/'heldout_predictions.npz', predictions=predictions, labels=labels, folds=folds,
                        original_trial_number=data['original_trial_number'])
    blocks = []
    for start in range(0, 1000, 50):
        path = root/'null_checkpoints'/f'permutations_{start:04d}_{start+49:04d}.npz'
        if not path.exists():
            with parallel_config(backend='loky', inner_max_num_threads=1):
                results = Parallel(n_jobs=args.jobs)(delayed(permutation_result)(i, features, labels, folds)
                                                     for i in range(start, start+50))
            block = {key: np.stack([r[key] for r in results]) for key in results[0]}
            temporary = path.with_suffix('.partial.npz')
            np.savez_compressed(temporary, **block)
            temporary.replace(path)
        with np.load(path) as saved:
            block = {key: saved[key] for key in saved.files}
        validate_null(block, labels, folds, np.arange(start, start+50))
        blocks.append(block)
        print(f'Label permutations: {start+50}/1000', flush=True)
    null = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    np.savez_compressed(root/'permutations.npz', **null)
    bootstrap = bootstrap_predictions(labels, predictions)
    np.savez_compressed(root/'bootstrap.npz', **bootstrap)
    summary = write_tables(root, trials, features, predictions, bootstrap, null)
    with threadpool_limits(limits=1):
        verification = verify(root)
    write_json(root/'run_status.json', dict(complete=True, elapsed_seconds=time.monotonic()-started,
                                          validation=verification))
    print(summary.to_string(index=False), flush=True)
    print(f'Complete: {root}', flush=True)


if __name__ == '__main__':
    main()
