"""Add the requested matched representations and one fixed LDA visualization.

Consumes frozen tau=3 geometry. No torus fitting or parameter selection occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from prego_fixed_geometry_decoding import OUTPUT as SOURCE, decode, score_predictions, bootstrap_predictions
from prego_geometric_fits import UNIT, sha256, write_json
from run_prego_fixed_geometry_decoding import initialize

OUTPUT = UNIT/'outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_additive_v1'
METHODS = ('relevant_band', 'geometry_relevant_band', 'average_psd',
           'geometry_average_psd', 'all_bands', 'geometry')
DIMENSIONS = (1, 15, 1, 15, 5, 14)


def average_psd(segments):
    x = signal.detrend(np.asarray(segments, dtype=float), axis=1, type='linear')
    frequency, density = signal.welch(x, fs=1000, window='hann', nperseg=500,
                                      noverlap=250, nfft=10000, detrend=False, axis=1)
    mean = density[:, (frequency >= 2)&(frequency <= 55)].mean(axis=1)
    return np.log10(np.maximum(mean, 1e-30))[:, None]


def build_features(bands, geometry, average):
    n = len(geometry)
    if bands.shape != (n, 5) or geometry.shape != (n, 14) or average.shape != (n, 1):
        raise ValueError('Expected 5D bands, 14D geometry, and 1D average PSD')
    beta = bands[:, 3:4]
    return dict(relevant_band=beta, geometry_relevant_band=np.hstack([geometry, beta]),
                average_psd=average, geometry_average_psd=np.hstack([geometry, average]),
                all_bands=bands, geometry=geometry)


def fit_projection(x, labels, folds):
    train = folds != 0
    model = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                      ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=2))])
    model.fit(x[train], labels[train])
    weights = model['lda'].scalings_[:, :2]
    signs = np.sign(weights[np.argmax(np.abs(weights), axis=0), np.arange(2)])
    signs[signs == 0] = 1
    transformed = model.transform(x)*signs
    mean, scale = transformed[train].mean(axis=0), transformed[train].std(axis=0)
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError('Degenerate LDA projection; no alternative projection is substituted')
    return (transformed-mean)/scale, model, signs, mean, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.output
    config = dict(source=str(SOURCE), n_trials=198, directions=6, K=1, m=3, tau_ms=3,
                  methods=list(METHODS), dimensions=list(DIMENSIONS), geometry_refit=False,
                  relevant_band='log10 integrated beta PSD, 13-30 Hz; saved all_bands column 3',
                  average_psd='log10 mean PSD density over 2-55 Hz; raw pre-GO epochs linearly detrended; '
                              'Hann Welch nperseg500 noverlap250 nfft10000; no normalization or filtering',
                  decoder='saved trial folds; training median + StandardScaler + LDA(lsqr, shrinkage=auto)',
                  bootstrap='2000 stratified prediction resamples, seed42000; conditional on fitted models',
                  new_permutation_tests=False, nominal_reference=1/6,
                  projection='eigen shrinkage LDA; fit only folds!=0; first two LDs; training-centered/unit-SD axes',
                  projection_holdout_fold=0, pca=False, width_feature=False,
                  inference='exploratory single selected recording; no additive superiority tests')
    paths = [SOURCE/name for name in ('features.npz', 'inputs/frozen_epochs.npz', 'inputs/selected_trials.csv',
             'config.json', 'heldout_predictions.npz', 'bootstrap.npz', 'validation.json')]
    sources = [Path(__file__)]+[UNIT/name for name in ('prego_fixed_geometry_decoding.py', 'prego_decoding.py',
               'prego_geometric_fits.py', 'run_prego_fixed_geometry_decoding.py', 'run_prego_geometric_fits.py')]
    provenance = {str(p.resolve()): sha256(p) for p in paths+sources}
    initialize(root, config, provenance, args.resume)
    old_files = {str(p): sha256(p) for p in SOURCE.rglob('*') if p.is_file()}
    write_json(root/'parent_manifest.json', old_files)
    for path in sources:
        if not (root/'source'/path.name).exists():
            shutil.copy2(path, root/'source'/path.name)
    frozen = np.load(SOURCE/'features.npz')
    epochs = np.load(SOURCE/'inputs/frozen_epochs.npz')
    trials = pd.read_csv(SOURCE/'inputs/selected_trials.csv')
    labels, folds = frozen['labels'], frozen['folds']
    np.testing.assert_array_equal(labels, trials.direction)
    np.testing.assert_array_equal(folds, trials.heldout_fold_zero_based)
    np.testing.assert_array_equal(epochs['labels'], labels)
    assert len(labels) == 198 and trials.groupby('direction').size().to_dict() == dict.fromkeys(range(1, 7), 33)
    features = build_features(frozen['all_bands'], frozen['geometry'], average_psd(epochs['segments']))
    np.savez_compressed(root/'features.npz', **features, labels=labels, folds=folds)
    trials.to_csv(root/'inputs/selected_trials.csv', index=False)
    with threadpool_limits(limits=1):
        predictions, models, audit = decode(features, labels, folds, retain_models=True)
        xy, projection, signs, mean, scale = fit_projection(features['geometry'], labels, folds)
    original_predictions = np.load(SOURCE/'heldout_predictions.npz')['predictions']
    np.testing.assert_array_equal(predictions[:, -2:], original_predictions[:, -2:])
    (root/'models').mkdir(exist_ok=True)
    for (method, fold), model in models.items():
        joblib.dump(model, root/'models'/f'{method}_fold{fold}.joblib')
    joblib.dump(projection, root/'models/visualization_lda_fold0.joblib')
    write_json(root/'training_audit.json', audit)
    np.savez_compressed(root/'heldout_predictions.npz', predictions=predictions, labels=labels, folds=folds)
    np.savez_compressed(root/'projection.npz', xy=xy, signs=signs, train_mean=mean, train_scale=scale,
                        labels=labels, folds=folds, imputed_fit=~np.isfinite(features['geometry']).all(axis=1))
    projection_table = trials.copy()
    projection_table['LD1'], projection_table['LD2'] = xy.T
    projection_table['role'] = np.where(folds == 0, 'held_out', 'training')
    projection_table['imputed_fit'] = ~np.isfinite(features['geometry']).all(axis=1)
    projection_table.to_csv(root/'tables/geometry_lda_projection.csv', index=False)
    scores = score_predictions(labels, predictions)
    boot = bootstrap_predictions(labels, predictions)
    old_boot = np.load(SOURCE/'bootstrap.npz')
    np.testing.assert_array_equal(boot['indices'], old_boot['indices'])
    np.testing.assert_array_equal(boot['macro_f1'][:, -2:], old_boot['macro_f1'][:, -2:])
    np.savez_compressed(root/'bootstrap.npz', **boot)
    rows = []
    prediction_rows = []
    for j, (method, dim) in enumerate(zip(METHODS, DIMENSIONS)):
        ci = np.quantile(boot['macro_f1'][:, j], [.025, .975])
        rows.append(dict(method=method, dimensions=dim, n_trials=198, macro_f1=scores['macro_f1'][j],
                         ci_low=ci[0], ci_high=ci[1], accuracy=scores['accuracy'][j]))
        pd.concat([trials, pd.DataFrame(features[method], columns=[f'feature_{k}' for k in range(dim)])],
                   axis=1).to_csv(root/'tables'/f'features_{method}.csv', index=False)
        prediction_rows.extend([dict(row, method=method, prediction=int(predictions[i, j]))
                                for i, row in enumerate(trials.to_dict('records'))])
    summary = pd.DataFrame(rows)
    summary.to_csv(root/'tables/summary.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(root/'tables/heldout_predictions.csv', index=False)
    np.testing.assert_allclose(scores['confusion_fraction'].sum(axis=2), 1)
    for path, digest in old_files.items():
        if sha256(path) != digest:
            raise ValueError(f'Changed parent experiment file: {path}')
    for path, digest in provenance.items():
        if sha256(path) != digest:
            raise ValueError(f'Changed source/input: {path}')
    write_json(root/'validation.json', dict(passed=True, n_trials=198, prediction_count=int(predictions.size),
               feature_dimensions=list(DIMENSIONS), parent_files_unchanged=len(old_files),
               original_geometry_and_all_band_predictions='exact match', original_bootstrap_indices='exact match',
               original_geometry_and_all_band_bootstraps='exact match', projection_training_trials=int((folds != 0).sum()),
               projection_heldout_trials=int((folds == 0).sum()), refitted_geometry=False))
    write_json(root/'run_status.json', dict(complete=True))
    print(summary.to_string(index=False), flush=True)
    print(f'Saved {root}', flush=True)


if __name__ == '__main__':
    main()
