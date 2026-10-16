"""Isolated Hilbert-envelope replication of corrected six-stage decoding."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

import run_stage_tau_sweep as sweep
import stage_ami_analysis as ami
from motor_lfp_utils import lag_embed
from plot_stage_normalization_geometry import (
    OUTPUT as EXAMPLES, RAW_SOURCE, envelope_window, normalization_checks,
    protected_hashes, trial_envelope,
)
from prego_geometric_fits import sha256, write_json
from run_stage_additive_spectral import METHODS, feature_sets, spectral_features
from stage_geometry_decoding import (
    COLUMNS, OUTPUT as SOURCE, STAGES, UNIT, decode, geometry_features,
    permutation, scores, validate_groups,
)
from prego_decoding import make_estimator

ROOT = UNIT / "outputs/stage_hilbert_envelope_tau_sweep_v1"
GRID = tuple(range(1, 26))
PILOT_TAUS = (1, 3, 13, 25)


def configuration():
    return dict(recording="monkeyT_session-y070316009-12_lfp-7", trials=198,
                windows=1188, fs_hz=1000, samples_per_window=300, m=3, K=1,
                taus_ms=list(GRID), variants=["native"], points="300-2*tau",
                anchors=["TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO"],
                normalization="window_bandpassed / (trial_envelope_window + 1e-6) * trial_median_envelope",
                envelope="full-trial linear detrend, median center, fourth-order 2-55 Hz zero-phase filter; abs(Hilbert); second-order 3 Hz zero-phase lowpass",
                precision="float64, matching frozen envelope examples", features=list(COLUMNS),
                fitter=sweep.config()["fitter"], decoder="saved five trial-grouped folds; training-only median imputation, standard scaling, LDA lsqr shrinkage auto",
                ami=ami.configuration(), ranking="pooled macro-F1 over 1-25 ms; exact ties favor smaller tau; exploratory",
                methods=list(METHODS), spectral_input="unchanged raw windows and existing Welch definition",
                bootstrap_count=2000, permutation_count=1000, permutation_seed=52000,
                interpretation="offline, full-trial temporal context; fixed-delay intervals and null tests do not correct for tau selection")


def source_manifest():
    names = ("run_stage_tau_sweep.py", "stage_ami_analysis.py",
             "select_trace_embedding_parameters.py", "plot_stage_normalization_geometry.py",
             "stage_geometry_decoding.py", "prego_geometric_fits.py",
             "prego_fixed_geometry_decoding.py", "prego_decoding.py", "motor_lfp_utils.py",
             "run_stage_additive_spectral.py", "plot_stage_hilbert_sweep.py",
             "run_stage_hilbert_sweep.py")
    paths = [UNIT / name for name in names]
    paths += [UNIT.parents[3] / "NeuralFieldManifold/fits/one_torus.py",
              EXAMPLES / "inputs_and_clouds.npz", EXAMPLES / "windows.csv",
              SOURCE / "inputs/event_samples.csv", SOURCE / "tables/summary.csv"]
    return {str(p.resolve()): sha256(p) for p in paths}


def initialize():
    protected, sources = protected_hashes(), source_manifest()
    if (ROOT / "config.json").exists():
        assert json.loads((ROOT / "config.json").read_text()) == configuration()
        assert json.loads((ROOT / "protected_hashes.json").read_text()) == protected
        assert json.loads((ROOT / "source_hashes.json").read_text()) == sources
    else:
        ROOT.mkdir(exist_ok=False)
        for name in ("inputs", "checkpoints", "tables", "predictions", "models", "selected", "ami"):
            (ROOT / name).mkdir()
        write_json(ROOT / "config.json", configuration())
        write_json(ROOT / "protected_hashes.json", protected)
        write_json(ROOT / "source_hashes.json", sources)
        write_json(ROOT / "environment.json", dict(python=platform.python_version(),
                   numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                   joblib=joblib.__version__, logical_cpus=os.cpu_count()))


def prepare():
    metadata, _ = sweep.load_inputs()
    normalization_checks()
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        raw = saved["raw"].copy()
    processed = np.full((1188, 300), np.nan)
    window_envelopes = np.full_like(processed, np.nan)
    failures, diagnostics, trials, filtered_trials, envelopes = [], [], [], [], []
    with np.load(RAW_SOURCE, allow_pickle=True) as data:
        assert int(data["sampling_rate_hz"]) == 1000
        for trial, rows in metadata.groupby("trial_index", sort=True):
            assert rows.raw_trial_index.nunique() == rows.original_trial_number.nunique() == 1
            index = int(rows.iloc[0].raw_trial_index)
            assert int(data["original_trial_number"][index]) == int(rows.iloc[0].original_trial_number)
            assert str(data["delay_label"][index]) == "short"
            assert int(data["direction"][index]) == int(rows.iloc[0].direction)
            full = data["lfp"][index].copy()
            trials.append(full)
            try:
                filtered, env, scale = trial_envelope(full)
            except ValueError as error:
                failures.append(dict(trial_index=int(trial), reason=str(error)))
                filtered = np.full(full.shape, np.nan)
                env = np.full(full.shape, np.nan)
                scale = np.nan
            filtered_trials.append(filtered)
            envelopes.append(env)
            for row in rows.itertuples():
                start, end = int(row.start_sample), int(row.end_sample_exclusive)
                np.testing.assert_array_equal(full[start:end], raw[row.window_index])
                assert end - start == 300
                if np.isfinite(scale):
                    processed[row.window_index] = envelope_window(raw[row.window_index], env[start:end], scale)
                    window_envelopes[row.window_index] = env[start:end]
                diagnostics.append(dict(window_index=row.window_index, trial_index=int(trial),
                    stage_name=row.stage_name, envelope_median=scale,
                    usable=bool(np.isfinite(processed[row.window_index]).all())))
    examples = pd.read_csv(EXAMPLES / "windows.csv")
    with np.load(EXAMPLES / "inputs_and_clouds.npz") as saved:
        np.testing.assert_array_equal(processed[examples.window_index], saved["envelope_processed"])
        np.testing.assert_array_equal(np.stack([lag_embed(x, 3, 3) for x in processed[examples.window_index]]),
                                      saved["envelope_clouds"])
    destination = ROOT / "inputs/windows.npz"
    if destination.exists():
        with np.load(destination) as saved:
            np.testing.assert_array_equal(processed, saved["processed"])
    else:
        np.savez_compressed(destination, raw=raw, processed=processed, window_envelopes=window_envelopes,
                            raw_trials=np.stack(trials), full_filtered=np.stack(filtered_trials),
                            full_envelopes=np.stack(envelopes))
        metadata.to_csv(ROOT / "inputs/window_metadata.csv", index=False)
        pd.read_csv(SOURCE / "inputs/event_samples.csv").to_csv(ROOT / "inputs/event_samples.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT / "tables/normalization_diagnostics.csv", index=False)
    write_json(ROOT / "normalization_validation.json", dict(trials=198, windows=1188,
               exact_three_trial_reproduction=True, failures=failures,
               finite_windows=int(np.isfinite(processed).all(axis=1).sum()), offline=True))
    print(f"Envelope preparation: {np.isfinite(processed).all(axis=1).sum()}/1188 finite windows", flush=True)
    return metadata, processed


def calculate_ami(metadata, processed):
    valid = np.isfinite(processed).all(axis=1)
    scopes = [("pooled", processed[valid], 48)]
    scopes += [(f"training_fold_{fold}", processed[valid & (metadata.heldout_fold_zero_based != fold)], 48)
               for fold in range(5)]
    scopes += [("pooled", processed[valid], bins) for bins in (32, 64)]
    curves, selections = [], []
    for name, segments, bins in scopes:
        checkpoint = ROOT / "ami" / f"{name}_bins{bins}.json"
        csv = checkpoint.with_suffix(".csv")
        if checkpoint.exists():
            record, curve = json.loads(checkpoint.read_text()), pd.read_csv(csv)
        else:
            if not len(segments):
                raise ValueError("No finite windows for AMI")
            curve, tau, low, high = ami.calculate_scope(segments, bins)
            record = dict(scope=name, bins=bins, n_windows=len(segments), selected_tau_ms=tau,
                          lower_edge=low, upper_edge=high)
            curve.to_csv(csv, index=False, float_format="%.17g")
            write_json(checkpoint, record)
        selected, smooth = ami.first_minimum(curve.ami_nats)
        assert selected == record["selected_tau_ms"]
        np.testing.assert_allclose(curve.smoothed_ami_nats, smooth, atol=1e-12)
        curves.append(curve.assign(scope=name, bins=bins))
        selections.append(record)
        print(f"AMI {name}, {bins} bins: tau={selected} ms", flush=True)
    pd.concat(curves).to_csv(ROOT / "tables/ami_curves.csv", index=False)
    pd.DataFrame(selections).to_csv(ROOT / "tables/ami_choices.csv", index=False)
    return selections[0]["selected_tau_ms"]


def configure_sweep(taus):
    # Reuse existing helpers only in this process, never their old output entrypoint.
    sweep.TAUS = tuple(sorted(set(taus)))
    sweep.VARIANTS = ("native",)
    sweep.PILOT_TAUS = PILOT_TAUS


def fit(metadata, processed, jobs, pilot=False, extra=None):
    configure_sweep(GRID if extra is None else (extra,))
    chosen = metadata[metadata.original_trial_number.isin((6, 9, 10))] if pilot else metadata
    taus = PILOT_TAUS if pilot else sweep.TAUS
    started = time.monotonic()
    workers = min(jobs, 6) if pilot else jobs
    for tau in taus:
        tasks = []
        for row in chosen.to_dict("records"):
            path = sweep.checkpoint(ROOT, "native", tau, row["window_index"])
            points = lag_embed(processed[row["window_index"]], 3, tau)
            if path.exists():
                sweep.checked_fit(path, row, points, tau)
            else:
                tasks.append((path, row, processed[row["window_index"]], tau))
        lap = time.monotonic()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as="generator_unordered")(
                delayed(fit_worker)(*task) for task in tasks)
            for n, (status, seconds) in enumerate(results, 1):
                if n % 200 == 0 or n == len(tasks):
                    print(f"native tau={tau}: {n}/{len(tasks)}; wall {time.monotonic()-lap:.1f}s", flush=True)
    if pilot:
        times = [json.loads(sweep.checkpoint(ROOT, "native", tau, int(i)).read_text())["elapsed_seconds"]
                 for tau in taus for i in chosen.window_index]
        record = dict(pilot_trials=[6,9,10], pilot_taus=list(taus), pilot_fits=len(times),
                      fit_cpu_seconds=sum(times), wall_seconds=time.monotonic()-started,
                      projected_fit_cpu_hours=float(np.mean(times))*29700/3600,
                      planned_fits=29700, jobs=jobs)
        record["projected_fit_wall_hours_ideal"] = record["projected_fit_cpu_hours"] / jobs
        write_json(ROOT / "pilot.json", record)
        print(json.dumps(record, indent=2), flush=True)


def fit_worker(path, row, signal, tau):
    # Spawned workers import modules afresh, so explicitly set their delay grid.
    configure_sweep(range(1, 101))
    return sweep.fit_task(path, row, signal, tau, "native")


def collect(metadata, processed, ami_tau):
    taus = GRID + (() if ami_tau is None or ami_tau in GRID else (ami_tau,))
    configure_sweep(taus)
    sweep.collect_and_decode(ROOT, metadata, processed)
    overall = pd.read_csv(ROOT / "tables/overall_f1.csv")
    eligible = overall[(overall.tau_ms.isin(GRID)) & (overall.decoder_status == "ok")]
    best = eligible.sort_values(["macro_f1", "tau_ms"], ascending=[False, True]).iloc[0]
    selection = dict(f1_tau_ms=int(best.tau_ms), f1=float(best.macro_f1), ami_tau_ms=ami_tau,
                     f1_selection="exploratory maximum over 1-25 ms", ami_selection="first pooled interior minimum, bins=48")
    write_json(ROOT / "selection.json", selection)
    pd.DataFrame([dict(criterion="F1", tau_ms=int(best.tau_ms), macro_f1=best.macro_f1),
                  dict(criterion="AMI", tau_ms=ami_tau)]).to_csv(ROOT / "tables/winners.csv", index=False)
    for tau in taus:
        path = ROOT / "inputs" / f"clouds_tau_{tau:02d}.npz"
        clouds = np.stack([sweep.cloud(x, tau, "native") for x in processed])
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(clouds, saved["clouds"])
        else:
            np.savez_compressed(path, clouds=clouds, window_index=metadata.window_index.to_numpy())
    return selection


def generic_decode(x, y, folds):
    predictions = np.full(len(y), -1, dtype=np.int8)
    models = {}
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if not np.isfinite(x[train]).any(axis=0).all():
            raise ValueError(f"All-missing training column in fold {fold}")
        model = make_estimator().fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        models[fold] = model
    return predictions, models


def null_task(path, index, x, y, folds, ids):
    if path.exists():
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}
    with threadpool_limits(limits=1):
        result = permutation(index, x, y, folds, ids)
    np.savez_compressed(path, **result)
    return result


def selected_reports(metadata, selection, jobs):
    y, folds, ids = (metadata[column].to_numpy() for column in ("stage", "heldout_fold_zero_based", "trial_index"))
    validate_groups(y, folds, ids)
    with np.load(ROOT / "inputs/windows.npz") as saved:
        bands, average = spectral_features(saved["raw"])
    with np.load(SOURCE / "bootstrap.npz") as saved:
        sampled_trials = saved["trial_indices"].copy()
    rows = np.stack([np.flatnonzero(ids == i) for i in range(198)])
    taus = sorted({t for t in (selection["f1_tau_ms"], selection["ami_tau_ms"]) if t is not None})
    for tau in taus:
        out = ROOT / "selected" / f"tau_{tau:02d}"
        out.mkdir(exist_ok=True)
        (out / "null_checkpoints").mkdir(exist_ok=True)
        x = pd.read_csv(ROOT / "tables" / f"features_native_tau_{tau:02d}.csv")[list(COLUMNS)].to_numpy()
        features = feature_sets(x, bands, average)
        with threadpool_limits(limits=1):
            decoded = [generic_decode(features[name], y, folds) for name in METHODS]
        predictions = np.column_stack([item[0] for item in decoded])
        with np.load(ROOT / "predictions" / f"native_tau_{tau:02d}.npz") as saved:
            np.testing.assert_array_equal(predictions[:, METHODS.index("geometry")], saved["predictions"])
        for method, (_, models) in zip(METHODS, decoded):
            for fold, model in models.items():
                joblib.dump(model, out / f"{method}_fold_{fold}.joblib")
        np.savez_compressed(out / "features.npz", **features, labels=y, folds=folds, trial_ids=ids)
        np.savez_compressed(out / "predictions.npz", predictions=predictions, labels=y, folds=folds, trial_ids=ids)
        boot_path = out / "bootstrap.npz"
        if boot_path.exists():
            with np.load(boot_path) as saved:
                boot, per_stage = saved["macro_f1"], saved["per_stage_f1"]
                np.testing.assert_array_equal(saved["trial_indices"], sampled_trials)
        else:
            boot = np.empty((2000, len(METHODS)))
            per_stage = np.empty((2000, len(METHODS), 6))
            for i, trial_sample in enumerate(sampled_trials):
                indices = rows[trial_sample].ravel()
                for j in range(len(METHODS)):
                    metric = scores(y[indices], predictions[indices, j])
                    boot[i, j], per_stage[i, j] = metric["macro_f1"], metric["per_stage_f1"]
            np.savez_compressed(boot_path, macro_f1=boot, per_stage_f1=per_stage, trial_indices=sampled_trials)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            null = Parallel(n_jobs=jobs)(delayed(null_task)(out / "null_checkpoints" / f"{i:04d}.npz",
                         i, x, y, folds, ids) for i in range(1000))
        null_scores = np.array([r["macro_f1"] for r in null])
        np.savez_compressed(out / "permutations.npz", macro_f1=null_scores,
                            labels=np.stack([r["labels"] for r in null]),
                            predictions=np.stack([r["predictions"] for r in null]))
        summary = []
        for j, name in enumerate(METHODS):
            metric = scores(y, predictions[:, j])
            low, high = np.quantile(boot[:, j], [.025, .975])
            summary.append(dict(method=name, dimensions=features[name].shape[1], macro_f1=metric["macro_f1"],
                                accuracy=metric["accuracy"], ci_low=low, ci_high=high))
            pd.DataFrame(metric["confusion_fraction"], index=STAGES, columns=STAGES).to_csv(out / f"recall_{name}.csv")
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, float_format="%.17g")
        g = METHODS.index("geometry")
        metric = scores(y, predictions[:, g])
        lower, upper = np.quantile(per_stage[:, g], [.025, .975], axis=0)
        pd.DataFrame(dict(stage=STAGES, f1=metric["per_stage_f1"], ci_low=lower, ci_high=upper)).to_csv(out / "per_stage_f1.csv", index=False)
        usable = np.isfinite(x).all(axis=1)
        failures = metadata.assign(usable=usable).groupby("stage_name").usable.agg(["count", "sum"])
        failures["unusable"] = failures["count"] - failures["sum"]
        failures.to_csv(out / "fit_accounting.csv")
        complete = np.all(usable.reshape(198, 6), axis=1)
        sensitivity = dict(complete_trials=int(complete.sum()))
        if not complete.all():
            mask = np.repeat(complete, 6)
            if all(np.any(mask & (folds == f)) for f in range(5)):
                with threadpool_limits(limits=1):
                    pred, _ = generic_decode(x[mask], y[mask], folds[mask])
                np.savez_compressed(out / "complete_trial_predictions.npz", labels=y[mask], predictions=pred, trial_ids=ids[mask], folds=folds[mask])
                sensitivity["macro_f1"] = scores(y[mask], pred)["macro_f1"]
            else:
                sensitivity["reason"] = "At least one original fold has no complete trials"
        paired = []
        for added, baseline in (("geometry_relevant_band", "relevant_band"), ("geometry_average_psd", "average_psd"), ("geometry_all_bands", "all_bands")):
            a, b = METHODS.index(added), METHODS.index(baseline)
            lo, hi = np.quantile(boot[:, a] - boot[:, b], [.025, .975])
            paired.append(dict(added=added, baseline=baseline, difference=summary[a]["macro_f1"] - summary[b]["macro_f1"], ci_low=lo, ci_high=hi))
        pd.DataFrame(paired).to_csv(out / "paired_differences.csv", index=False)
        write_json(out / "metrics.json", dict(tau_ms=tau, permutation_p=(1 + int((null_scores >= metric["macro_f1"]).sum())) / 1001,
                   null_median=float(np.median(null_scores)), sensitivity=sensitivity,
                   inference="fixed-delay descriptive; no correction for tau selection"))
        print(f"Completed selected delay {tau} ms: F1={metric['macro_f1']:.6f}", flush=True)


def verify(metadata, processed):
    assert protected_hashes() == json.loads((ROOT / "protected_hashes.json").read_text())
    assert source_manifest() == json.loads((ROOT / "source_hashes.json").read_text())
    overall = pd.read_csv(ROOT / "tables/overall_f1.csv")
    stages = pd.read_csv(ROOT / "tables/stage_f1.csv")
    y = metadata.stage.to_numpy()
    selection = json.loads((ROOT / "selection.json").read_text())
    for record in overall.itertuples():
        tau = int(record.tau_ms)
        with np.load(ROOT / "inputs" / f"clouds_tau_{tau:02d}.npz") as saved:
            points = saved["clouds"]
        assert points.shape == (1188, 300 - 2*tau, 3)
        np.testing.assert_array_equal(points, np.stack([lag_embed(x, 3, tau) for x in processed]))
        for row in metadata.to_dict("records"):
            sweep.checked_fit(sweep.checkpoint(ROOT, "native", tau, row["window_index"]), row, points[row["window_index"]], tau)
        if record.decoder_status != "ok":
            continue
        with np.load(ROOT / "predictions" / f"native_tau_{tau:02d}.npz") as saved:
            np.testing.assert_array_equal(saved["folds"], metadata.heldout_fold_zero_based)
            np.testing.assert_array_equal(saved["trial_ids"], metadata.trial_index)
            np.testing.assert_array_equal(saved["labels"], y)
            metric = scores(y, saved["predictions"])
        np.testing.assert_allclose(metric["macro_f1"], record.macro_f1, atol=1e-12)
        np.testing.assert_allclose(stages[stages.tau_ms == tau].set_index("stage_name").loc[list(STAGES), "f1"], metric["per_stage_f1"], atol=1e-12)
        np.testing.assert_allclose(metric["confusion_fraction"].sum(axis=1), 1.)
        if tau in {selection["f1_tau_ms"], selection["ami_tau_ms"]}:
            out = ROOT / "selected" / f"tau_{tau:02d}"
            with np.load(out / "features.npz") as saved:
                x = saved["geometry"]
                for method in METHODS:
                    for fold in range(5):
                        model = joblib.load(out / f"{method}_fold_{fold}.joblib")
                        train = metadata.heldout_fold_zero_based.to_numpy() != fold
                        np.testing.assert_allclose(model.named_steps["impute"].statistics_, np.nanmedian(saved[method][train], axis=0))
            with np.load(out / "permutations.npz") as saved:
                for index in (0, 17, 999):
                    with threadpool_limits(limits=1):
                        expected = permutation(index, x, y, metadata.heldout_fold_zero_based.to_numpy(), metadata.trial_index.to_numpy())
                    np.testing.assert_array_equal(expected["predictions"], saved["predictions"][index])
    expected_best = overall[overall.tau_ms.isin(GRID) & (overall.decoder_status == "ok")].sort_values(["macro_f1", "tau_ms"], ascending=[False, True]).iloc[0]
    assert int(expected_best.tau_ms) == selection["f1_tau_ms"]
    write_json(ROOT / "validation.json", dict(trials=198, windows=1188, delays=len(overall),
               all_clouds_and_checkpoints_verified=True, all_sweep_scores_reproduced=True,
               train_only_imputation_verified=True, selected_null_refits_verified=True,
               protected_results_unchanged=True, sources_unchanged=True))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("prepare", "pilot", "fit", "decode", "selected", "validate", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    start = time.monotonic()
    initialize()
    metadata, processed = prepare()
    ami_tau = calculate_ami(metadata, processed)
    if args.phase in ("pilot", "all"):
        fit(metadata, processed, args.jobs, pilot=True)
    if args.phase in ("fit", "all"):
        fit(metadata, processed, args.jobs)
        if ami_tau is not None and ami_tau not in GRID:
            fit(metadata, processed, args.jobs, extra=ami_tau)
    if args.phase in ("decode", "all"):
        collect(metadata, processed, ami_tau)
    if args.phase in ("selected", "all"):
        selected_reports(metadata, json.loads((ROOT / "selection.json").read_text()), args.jobs)
    if args.phase in ("validate", "all"):
        verify(metadata, processed)
    if args.phase == "all":
        from plot_stage_hilbert_sweep import render
        render()
    write_json(ROOT / f"runtime_{args.phase}.json", dict(phase=args.phase, wall_seconds=time.monotonic()-start, jobs=args.jobs))


if __name__ == "__main__":
    main()
