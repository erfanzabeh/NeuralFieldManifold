"""Package frozen 198-trial results. No fitting, decoding, or source modification."""
from pathlib import Path
import hashlib
import json
import shutil
import textwrap
import zipfile

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

BASE = Path('/home/nochen/code/NeuralFieldManifold/notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching')
DEST = BASE / 'outputs/recording_198_trial_dossiers_v1'
RUNS = {'decoding_no_pca': 'prego_psd_ktorus_nopca_v1',
        'ols_audit': 'prego_ols_audit_v1', 'historical_pca_predecessor': 'prego_single_channel'}
METHODS = ['peak_power', 'peak_frequency', 'power_frequency', 'all_bands',
           'ktorus', 'ktorus_power_frequency', 'ktorus_all_bands']
LABELS = ['Peak power', 'Peak frequency', 'Power + frequency', 'All band powers',
          'K-mode geometry', 'Geometry + power/frequency', 'Geometry + all bands']
COLORS = ['#66a5a0', '#a4c8c3', '#287c79', '#878785', '#ba5947', '#b88a3b', '#745989']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})


def selected_recordings(cohort):
    return sorted(cohort.loc[cohort.n_trials == 198, 'recording'].tolist())


def prediction_metrics(frame):
    if frame.duplicated(['method', 'raw_trial_index']).any():
        raise ValueError('Repeated held-out prediction for a method/trial')
    return pd.DataFrame([dict(method=m, macro_f1=f1_score(g.true_direction, g.predicted_direction,
                        labels=range(1, 7), average='macro', zero_division=0), n_trials=len(g))
                        for m, g in frame.groupby('method')])


def validate_trials(a, b):
    for key in ['raw_trial_indices', 'labels', 'original_trial_number', 'folds']:
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'Trial identity/order mismatch: {key}')
    if len(a['labels']) != 198 or not np.array_equal(np.bincount(a['labels'])[1:7], [33]*6):
        raise ValueError('Expected exactly 33 trials in each of six directions')
    if set(a['folds']) != set(range(5)):
        raise ValueError('Expected all five held-out folds')


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def md_table(frame):
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    lines = [header, '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def save_figure(fig, folder, name, caption, book):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(folder / (name + '.svg'), bbox_inches='tight')
    fig.savefig(folder / (name + '.png'), bbox_inches='tight', dpi=220)
    (folder / (name + '.caption.txt')).write_text(caption + '\n')
    book.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def decoding_plot(ax, scores, nulls, title):
    v = scores.set_index('method').loc[METHODS, 'macro_f1'].to_numpy()
    pos = np.arange(len(METHODS))
    ax.barh(pos, v, color=COLORS, height=.58)
    for i, method in enumerate(METHODS):
        q = nulls.loc[nulls.method == method, 'macro_f1'].quantile([.025, .5, .975]).to_numpy()
        ax.errorbar(q[1], i + .31, xerr=[[q[1]-q[0]], [q[2]-q[1]]], fmt='o',
                    color='#454545', markersize=3, capsize=2, linewidth=1)
        ax.text(v[i]+.006, i, f'{v[i]:.3f}', va='center', fontsize=9)
    ax.set(yticks=pos, yticklabels=LABELS, xlabel='Held-out macro-F1', xlim=(0, .36), title=title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.13)
    ax.set_axisbelow(True)


def extract_tables(source, records, alias, ledger):
    for path in sorted((source / 'tables').glob('*.csv')):
        columns = pd.read_csv(path, nrows=0).columns
        key = 'recording' if 'recording' in columns else 'lfp_uid' if 'lfp_uid' in columns else None
        if key is None:
            continue
        ledger[str(path)] = sha(path)
        pieces = []
        for chunk in pd.read_csv(path, chunksize=20000):
            part = chunk.loc[chunk[key].isin(records)]
            if not part.empty:
                pieces.append(part)
        if pieces:
            frame = pd.concat(pieces, ignore_index=True)
            for recording, group in frame.groupby(key):
                target = DEST / recording / alias / 'tables'
                target.mkdir(parents=True, exist_ok=True)
                group.to_csv(target / path.name, index=False)


def copy_tree(source, destination, ledger):
    shutil.copytree(source, destination)
    for p in source.rglob('*'):
        if p.is_file():
            digest = sha(p)
            ledger[str(p)] = digest
            assert sha(destination / p.relative_to(source)) == digest


def main():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite {DEST}')
    cohort = pd.read_csv(BASE / 'outputs/prego_ols_audit_v1/tables/cohort.csv')
    records = selected_recordings(cohort)
    if len(records) != 4:
        raise ValueError(f'Expected the four inspected 198-trial recordings, found {records}')
    DEST.mkdir(parents=True)
    ledger = {}
    for alias, run in RUNS.items():
        source = BASE / 'outputs' / run
        print('Gathering', run, flush=True)
        context = DEST / 'source_context' / alias
        context.mkdir(parents=True)
        for name in ['config.json', 'manifest.json', 'README.md', 'RESULTS.md', 'validation.json', 'verification.json']:
            p = source / name
            if p.exists():
                shutil.copy2(p, context / name)
                ledger[str(p)] = sha(p)
        for relative in ['source', 'logs/source', 'logs/integrity_source']:
            p = source / relative
            if p.is_dir():
                copy_tree(p, context / relative, ledger)
        for recording in records:
            copy_tree(source / 'cache' / recording, DEST / recording / alias / 'cache', ledger)
        extract_tables(source, records, alias, ledger)
    manifest = pd.read_csv(BASE / 'tables/lfp_manifest.csv').set_index('lfp_uid')
    ledger[str(BASE / 'tables/lfp_manifest.csv')] = sha(BASE / 'tables/lfp_manifest.csv')
    summary = []
    verification = []
    for recording in records:
        print('Building report', recording, flush=True)
        folder = DEST / recording
        dec = folder / 'decoding_no_pca'
        ols = folder / 'ols_audit'
        figs = folder / 'figures'
        data = arrays(dec / 'cache/trials.npz')
        processed = arrays(ols / 'cache/trials.npz')
        validate_trials(data, processed)
        info = json.loads((ols / 'cache/metadata.json').read_text())
        info['sampling_rate_hz'] = 1000
        source = BASE / str(manifest.loc[recording, 'converted_path'])
        inputs = folder / 'inputs'
        inputs.mkdir()
        shutil.copy2(source, inputs / 'full_converted_recording.npz')
        ledger[str(source)] = sha(source)
        assert sha(inputs / 'full_converted_recording.npz') == ledger[str(source)]
        with np.load(source, allow_pickle=True) as z:
            ix = data['raw_trial_indices']
            assert np.array_equal(z['original_trial_number'][ix], data['original_trial_number'])
            assert np.array_equal(z['direction'][ix], data['labels'])
            assert (z['delay_label'][ix].astype(str) == 'short').all()
            raw = z['lfp'][ix, info['start_sample']:info['end_sample_exclusive']]
            assert raw.shape == (198, 500)
            np.savez_compressed(inputs / 'selected_raw_prego_epochs.npz', segments=raw,
                                time_ms=np.arange(-500, 0), **{k: data[k] for k in
                                ['raw_trial_indices', 'original_trial_number', 'labels', 'folds']})
        (inputs / 'audited_alignment.json').write_text(json.dumps(info, indent=2)+'\n')
        trials = pd.DataFrame({'row_index': np.arange(198), 'raw_trial_index': data['raw_trial_indices'],
                  'original_trial_number': data['original_trial_number'], 'direction': data['labels'],
                  'heldout_fold_zero_based': data['folds'], 'delay': 'short'})
        trials.to_csv(inputs / 'selected_trials.csv', index=False)
        counts = pd.crosstab(trials.heldout_fold_zero_based, trials.direction).reindex(index=range(5), columns=range(1, 7))
        counts.to_csv(inputs / 'test_counts_by_fold_and_direction.csv')
        bounds = ['12_25Hz', '25_40Hz'] if info['monkey'] == 'M' else ['12_40Hz']
        spectral = trials.copy()
        for key, suffix in [('power', 'log10_peak_psd'), ('frequency', 'peak_frequency_hz')]:
            for col, name in enumerate(bounds):
                spectral[name+'_'+suffix] = data[key][:, col]
        for col, name in enumerate(['delta_2_4Hz', 'theta_4_8Hz', 'alpha_8_13Hz', 'beta_13_30Hz', 'low_gamma_30_55Hz']):
            spectral[name+'_log10_integrated_power'] = data['bands'][:, col]
        spectral.to_csv(dec / 'spectral_features.csv', index=False)
        for fold in range(5):
            z = arrays(dec / 'cache' / f'fold_{fold}.npz')
            frame = pd.concat([trials, pd.DataFrame(z['features'], columns=z['feature_names'])], axis=1)
            frame['held_out_in_this_feature_fit'] = data['folds'] == fold
            frame.to_csv(dec / f'geometry_features_fold_{fold}.csv', index=False)
        pred = pd.read_csv(dec / 'cache/predictions.csv')
        scores = pd.read_csv(dec / 'tables/recording_scores.csv')
        calculated = prediction_metrics(pred).set_index('method')
        assert set(calculated.index) == set(METHODS)
        assert np.allclose(calculated.loc[scores.method, 'macro_f1'], scores.macro_f1, atol=1e-12, rtol=0)
        assert (calculated.n_trials == 198).all()
        for method, group in pred.groupby('method'):
            expected = trials.set_index('raw_trial_index').loc[group.raw_trial_index]
            assert np.array_equal(group.true_direction, expected.direction)
            assert np.array_equal(group.fold, expected.heldout_fold_zero_based)
            cm = confusion_matrix(group.true_direction, group.predicted_direction, labels=range(1, 7), normalize='true')
            assert np.allclose(cm.sum(axis=1), 1)
            saved_cm = pd.read_csv(dec / 'tables/recording_confusions.csv')
            saved_cm = saved_cm[saved_cm.method == method].pivot(index='true_direction', columns='predicted_direction', values='fraction')
            assert np.allclose(cm, saved_cm.to_numpy())
        nulls = pd.read_csv(dec / 'cache/null_scores.csv')
        assert (nulls.groupby('method').size() == 200).all()
        selection = pd.read_csv(dec / 'tables/embedding_selection.csv')
        diag = pd.read_csv(dec / 'tables/fit_diagnostics.csv')
        held_diag = diag[diag.held_out]
        assert len(held_diag) == 198
        ar = pd.read_csv(ols / 'tables/selected_models.csv').query('family == "ar"').sort_values('outer_fold')
        ar_score = pd.read_csv(ols / 'tables/heldout_recording_scores.csv').query('family == "ar" and horizon_ms == 10').iloc[0]
        short_name = f"Monkey {info['monkey']} | {info['day']} | channel {info['lfp']}"
        overview = dict(recording=recording, monkey=info['monkey'], session=info['session'], day=info['day'],
                        channel=info['lfp'], source_trials=info['raw_trials'],
                        source_short_trials=int(manifest.loc[recording, 'short_delay_count']),
                        source_long_trials=int(manifest.loc[recording, 'long_delay_count']), selected_trials=198,
                        trials_per_direction=33, geometry_heldout_median_residual=held_diag.normalized_error.median(),
                        geometry_heldout_success_fraction=held_diag.success.mean(),
                        ar_orders=','.join(ar.p.astype(str)), ar_10ms_nmse=ar_score.nmse, ar_10ms_r2=ar_score.r2)
        overview.update({row.method: row.macro_f1 for row in scores.itertuples()})
        summary.append(overview)
        with PdfPages(folder / 'Recording_Report.pdf') as book:
            fig, ax = plt.subplots(figsize=(9.2, 7.5))
            fig.subplots_adjust(left=.31, right=.93, top=.75, bottom=.22)
            decoding_plot(ax, scores, nulls, 'Single-channel reach-direction decoding')
            fig.text(.06, .955, short_name, fontsize=17, weight='bold')
            fig.text(.06, .91, recording, fontsize=10)
            fig.text(.06, .845, f"198 short-delay trials = 33 per direction | final 500 ms before GO\n"
                     f"Five held-out folds: {', '.join(map(str, counts.sum(axis=1)))} test trials; 158-159 training trials.", fontsize=11)
            fig.text(.06, .045, 'Bars: macro-F1 from all held-out predictions. Gray dots/ranges: median and central 95%\n'
                     'of 200 saved label-shuffle scores, not confidence intervals. Geometry is the earlier K-mode\n'
                     'shape-matrix representation, not the 15-feature tube fit and not an OLS-derived decoder.', fontsize=9)
            save_figure(fig, figs, '01_decoding', 'Saved no-PCA pre-GO decoding. No new fitting or significance tests.', book)
            fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), layout='constrained')
            for ax, method, label in zip(axes.flat, METHODS, LABELS):
                g = pred[pred.method == method]
                cm = confusion_matrix(g.true_direction, g.predicted_direction, labels=range(1, 7), normalize='true')
                im = ax.imshow(cm, vmin=0, vmax=1, cmap='YlGnBu')
                for i in range(6):
                    for j in range(6):
                        ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                                color='white' if cm[i,j] > .55 else '#17282d')
                ax.set(title=label, xticks=range(6), xticklabels=range(1, 7), yticks=range(6),
                       yticklabels=range(1, 7), xlabel='Predicted direction', ylabel='True direction')
            axes.flat[-1].axis('off')
            axes.flat[-1].text(.03, .75, 'Each row contains 33 trials.\nCells are prediction fractions.\nDiagonal = class recall, not F1.\n\nSame 198 trials for every method.', va='top', fontsize=10)
            fig.colorbar(im, ax=axes.flat[-1], label='Prediction fraction', shrink=.75)
            fig.suptitle(short_name + ' | held-out confusion matrices', fontsize=15)
            save_figure(fig, figs, '02_confusion_matrices', 'All seven methods. Shared color scale 0-1; rows normalized.', book)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
            direction_scores = pd.read_csv(dec / 'tables/direction_scores.csv')
            for method, label, color in [('power_frequency', 'Power + frequency', COLORS[2]), ('ktorus', 'K-mode geometry', COLORS[4])]:
                g = direction_scores[direction_scores.method == method].sort_values('direction')
                axes[0].plot(g.direction, g.f1, 'o-', label=label, color=color)
            axes[0].set(xlabel='Reach direction', ylabel='Per-direction F1', xticks=range(1, 7), ylim=(0, .65))
            axes[0].legend(fontsize=8)
            axes[1].hist(held_diag.normalized_error, bins=np.linspace(0, 1, 21), color=COLORS[4], edgecolor='white')
            axes[1].axvline(held_diag.normalized_error.median(), color='black', linestyle='--',
                            label=f"Median {held_diag.normalized_error.median():.3f}")
            axes[1].set(xlabel='Geometry residual SSE/TSS (lower is better)', ylabel='Held-out trial fits', xlim=(0, 1))
            axes[1].legend(fontsize=9)
            fig.suptitle(short_name + ' | direction scores and fit quality', fontsize=14)
            save_figure(fig, figs, '03_direction_and_fit_quality', 'Per-class F1 from saved predictions; geometric residual is not decoding error.', book)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
            for fold in range(5):
                e = json.loads((dec / 'cache' / f'fold_{fold}.json').read_text())
                axes[0].plot(e['psd_frequencies'], e['residual_median'], label=f'Fold {fold+1}')
                for f in e['frequencies']:
                    axes[0].plot(f, np.interp(f, e['psd_frequencies'], e['residual_median']), 'o', color='#383838', ms=3)
                axes[1].plot(e['ami_lags'], e['ami'], label=f"Fold {fold+1}: tau={e['tau']} ms")
                axes[1].axvline(e['tau'], color='#999999', alpha=.35, linewidth=.8)
            axes[0].set(xlabel='Frequency (Hz)', ylabel='Log PSD residual above fitted background', xlim=(2, 55))
            axes[1].set(xlabel='Delay (ms)', ylabel='Average mutual information', xlim=(1, 100))
            axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
            fig.suptitle(short_name + ' | earlier decoder: training-only PSD and delay selection', fontsize=13)
            save_figure(fig, figs, '04_saved_embedding_selection', 'Frozen training-fold curves from the earlier no-PCA decoder, not OLS m=9/tau=1.', book)
            example_rows = []
            fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, layout='constrained')
            for direction, ax in zip(range(1, 7), axes.flat):
                candidates = np.flatnonzero((data['labels'] == direction) & (data['folds'] == 0))
                i = candidates[np.argmin(data['original_trial_number'][candidates])]
                example_rows.append(dict(direction=direction, row_index=int(i), original_trial_number=int(data['original_trial_number'][i])))
                ax.plot(np.arange(-500, 0), processed['segments'][i], color=COLORS[2], lw=.9)
                ax.set(title=f"Direction {direction} | trial {data['original_trial_number'][i]}", ylabel='Processed LFP (scaled)')
                ax.axvline(0, color='gray', linestyle='--', lw=.7)
            for ax in axes[-1]: ax.set_xlabel('Time before GO (ms)')
            fig.suptitle(short_name + ' | one held-out trial per direction', fontsize=14)
            save_figure(fig, figs, '05_trial_traces', 'Processed audit signals. Examples: first outer-test fold, lowest original trial ID within each direction, not performance-selected.', book)
            pd.DataFrame(example_rows).to_csv(inputs / 'example_trials.csv', index=False)
        curves = BASE / 'outputs/prego_ols_audit_v1/plots/recordings' / recording
        copy_tree(curves, ols / 'saved_figures', ledger)
        doc = fitz.open(folder / 'Recording_Report.pdf')
        with fitz.open(curves / 'order_sweep.pdf') as extra:
            doc.insert_pdf(extra)
        temp = folder / 'Recording_Report_complete.pdf'
        doc.save(temp); doc.close()
        temp.replace(folder / 'Recording_Report.pdf')
        presented = scores.set_index('method').loc[METHODS, ['macro_f1']].reset_index()
        presented['method'] = LABELS
        presented['macro_f1'] = presented.macro_f1.map(lambda v: f'{v:.6f}')
        selected_view = selection[['fold', 'K', 'dimension', 'tau_samples', 'feature_count', 'frequencies_hz']].copy()
        readme = f'''# {short_name}

Recording ID: `{recording}`. Selected because it has 198 trials, not because of its decoding performance.

## Start here

- [Recording_Report.pdf](Recording_Report.pdf): decoding, all confusion matrices, per-direction F1, geometric residuals, saved PSD/AMI selection, example traces, and the existing AR-order curve.
- `inputs/selected_trials.csv`: exactly 198 trials, 33 in each of six directions. Fold IDs are zero-based.
- `decoding_no_pca/cache/predictions.csv`: every saved held-out prediction.
- `decoding_no_pca/spectral_features.csv` and `geometry_features_fold_*.csv`: readable feature matrices.
- `ols_audit/tables/`: all recording-specific order sweeps, coefficients, poles, selections, and held-out signal-prediction scores.

## Trial accounting

The source recording contains {info['raw_trials']} trials: {overview['source_short_trials']} short-delay and {overview['source_long_trials']} long-delay.
The selected 198 are balanced short-delay trials only. Every trial contributes one 500-ms segment from this channel.
This is a single recording, not a pooled multi-channel decoder. Other channels can observe the same behavioral trials.

Test counts by fold and direction:

{md_table(counts.reset_index())}

Sampling rate: 1000 Hz. Audited GO is zero-based sample {info['go_sample_zero_based']}; use samples
[{info['start_sample']}:{info['end_sample_exclusive']}] for the pre-GO interval. Each short delay is {info['short_delay_ms']} ms.
IMPORTANT: `inputs/full_converted_recording.npz` preserves the old conversion exactly, including its legacy GO field.
For alignment, use `inputs/audited_alignment.json` or the already-correct `selected_raw_prego_epochs.npz`.

## Earlier no-PCA decoding

{md_table(presented)}

These are macro-F1 values pooled across the held-out predictions, not accuracy and not a mean across recordings.
The report shows empirical central 95% ranges of the 200 saved shuffle scores. These are not confidence intervals.
No new significance tests or claims of above-chance performance are introduced in this dossier.

Fold-specific geometry choices from THAT decoding run:

{md_table(selected_view)}

The geometric feature vector contains the upper-triangular mode-specific shape matrices, coordinate residual RMSEs,
and total normalized residual. It is NOT the original 15D R1/R2/r tube feature vector. The named CSV columns expose every feature.
For one mode in 3D, this is 6 shape terms + 3 coordinate errors + 1 total error = 10 features.
Feature dimensions may differ across folds; do not concatenate fold-specific features as if they were one fitted coordinate system.
Median held-out geometric residual SSE/TSS: {overview['geometry_heldout_median_residual']:.6f}; successful-fit fraction: {overview['geometry_heldout_success_fraction']:.6f}.
Numerical convergence does not establish good geometric recovery or a true torus topology.

Spectral peaks: {', '.join(bounds)}; power is log10 peak PSD and frequency is peak location in Hz.
The five band features are log10 integrated delta, theta, alpha, beta, and low-gamma power.
Classifier: training-fold median imputation, standardization, and shrinkage LDA. Only saved predictions are used here.
There is no serialized fitted LDA object in this package; the features and fold assignments needed to refit are included.

## Later OLS audit: NOT a new decoder

Selected AR orders in fold order 0-4: {overview['ar_orders']}.
Held-out 10-ms recursive AR NMSE: {ar_score.nmse:.6f}; R-squared: {ar_score.r2:.6f}.
These measure waveform prediction, not direction decoding. Coefficients, intercepts, numerical conditioning,
poles and complete validation candidates are retained in `ols_audit/`.
The auxiliary lag-readout m=9/tau=1 results remain archived there for completeness; they are not emphasized in this report
and were not used to update the geometric decoder. Preprocessing uses the whole pre-GO epoch, so prediction is offline, not causal online forecasting.

## Historical material

`historical_pca_predecessor/` preserves the earlier pre-GO tube-feature/PCA pipeline, including matched short/long analyses.
Those features, trial subsets and null controls must not be confused with the main no-PCA results above.
The underlying run configurations and source paths are in `../source_context/` and `../source_manifest.csv`.
Earlier movement-aligned experiments are not relabeled as these 198-trial pre-GO results; their project source remains at `{BASE}`.

## Verification

The selected trial identities, directions and folds match between the no-PCA decoding and OLS audit.
All seven macro-F1 scores were recomputed from saved predictions and checked against saved tables; all confusion rows sum to one.
No model was fit, no new decoding was run, and all packaged source files were checked for unchanged hashes.
'''
        (folder / 'README.md').write_text(readme)
        verification.append(dict(recording=recording, trials=198, classes=6, macro_f1_recomputed=True,
                                 confusion_rows_verified=True, folds_identical=True, source_copy_hashes_verified=True))
    pd.DataFrame(summary).to_csv(DEST / 'recording_summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.subplots_adjust(left=.19, right=.97, top=.88, bottom=.11, wspace=.65, hspace=.43)
    for ax, recording, info in zip(axes.flat, records, summary):
        d = DEST / recording / 'decoding_no_pca'
        decoding_plot(ax, pd.read_csv(d/'tables/recording_scores.csv'), pd.read_csv(d/'cache/null_scores.csv'),
                      f"{info['monkey']} | {info['day']} | channel {info['channel']}")
        ax.tick_params(axis='y', labelsize=8)
    fig.suptitle('Four recordings with 198 selected trials each', fontsize=19, weight='bold', y=.975)
    fig.text(.5, .925, '33 trials per direction | short delays | single-channel | final 500 ms before GO', ha='center', fontsize=12)
    fig.text(.04, .025, 'Saved no-PCA decoding. Gray dots/ranges: median and central 95% of 200 shuffled-label scores, not confidence intervals.\n'
             'K-mode geometry uses shape-matrix features, not the original tube-radii features. Channels in the same session share behavioral trials.', fontsize=10)
    fig.savefig(DEST / 'Overview.png', dpi=180)
    fig.savefig(DEST / 'Overview.pdf')
    plt.close(fig)
    links = '\n'.join(f"- [{r}](./{r}/Recording_Report.pdf) | [details](./{r}/README.md)" for r in records)
    readme = f'''# The four 198-trial recordings

These are four single-channel recordings from two sessions and two monkeys, not four independent sessions.
All have 198 balanced short-delay trials (33 per direction). No recording was selected by F1.
The first listed recording (M channel 1) is the walkthrough example solely by sorted recording ID.

## Reports

{links}

`Overview.pdf` / `Overview.png` compares their existing no-PCA decoding results. `recording_summary.csv` contains exact values.
Each folder includes full converted signal arrays, correctly aligned selected raw epochs, audit-processed epochs,
trial identities and folds, all saved spectral/geometric features, every held-out prediction, seven confusion matrices,
per-direction F1, 200 shuffle scores per method, PSD/AMI choices, geometric fits and diagnostics, AR coefficients/poles,
full candidate validation scores, the historical PCA predecessor, and standalone figures.

The no-PCA decoder is the earlier K-dependent shape-matrix model. The OLS audit did not replace it with an AR-derived decoder.
No serialized fitted LDA object was saved by the original pipeline. No new fitting, training, decoding, or significance testing occurs here.
Lag-readout diagnostics are archived only, not used to choose a geometric model.
Group-level day-clustered tests from the full cohort cannot be assigned to these individual recordings.

The complete multi-gigabyte raw animal MATLAB files are not duplicated; source locations/hashes are preserved in the copied run manifests.
The full per-recording converted arrays are included. Earlier movement-aligned experiments remain untouched in the project, not mixed into this package.

`source_manifest.csv` lists exact source paths and hashes. `verification.json` records internal checks.
`source/build_recording_198_dossiers.py` reproduces the package from existing local results only and refuses to overwrite it.
'''
    (DEST / 'README.md').write_text(readme)
    source = DEST / 'source'
    source.mkdir()
    shutil.copy2(__file__, source / Path(__file__).name)
    shutil.copy2('/tmp/test_recording_198_dossiers.py', source / 'test_recording_198_dossiers.py')
    for path, digest in ledger.items():
        if sha(Path(path)) != digest:
            raise ValueError('Source changed while packaging: '+path)
    pd.DataFrame([dict(source=p, sha256=h) for p, h in sorted(ledger.items())]).to_csv(DEST / 'source_manifest.csv', index=False)
    (DEST / 'verification.json').write_text(json.dumps(dict(recordings=verification, unchanged_source_files=len(ledger),
                              fitting_performed=False, decoding_performed=False), indent=2)+'\n')
    archive = DEST.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                z.write(path, Path(DEST.name) / path.relative_to(DEST))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print('DONE', DEST, 'ZIP_MB', archive.stat().st_size / 1e6, flush=True)


if __name__ == '__main__':
    main()
