"""Resumable, fixed-cohort tau sweep for corrected six-stage macaque windows."""
from __future__ import annotations

import argparse
import json
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

from motor_lfp_utils import lag_embed
from prego_geometric_fits import array_hash, fit_one, sha256, write_json
from stage_geometry_decoding import (
    ANCHORS, COLUMNS, EVENT_CODES, OUTPUT as SOURCE, STAGES, UNIT,
    decode, geometry_features, scores, validate_groups,
)

ROOT = UNIT / "outputs/tau_sweep_v1"
VARIANTS = ("native", "matched260")
TAUS = tuple(range(1, 21))
PILOT_TRIALS = (6, 9, 10)
PILOT_TAUS = (1, 3, 10, 20)
N_WINDOWS = 1188
OLD_F1 = 0.2928413893498778


def config():
    return dict(recording="monkeyT_session-y070316009-12_lfp-7", trials=198,
                windows=N_WINDOWS, stages=list(STAGES), sampling_rate_hz=1000,
                window_samples=300, m=3, K=1, taus_ms=list(TAUS),
                variants=list(VARIANTS), matched_anchors="samples 40..299 inclusive",
                native_points="300-2*tau", matched_points=260,
                fitter=dict(name="one_torus", lam=.1, hole_ratio=.5,
                            regularization_harmonics=3, loss="huber", max_nfev=6000),
                features=list(COLUMNS), decoder="saved five trial-grouped folds; training-only "
                        "median imputation, standard scaling, shrinkage LDA",
                ranking="pooled six-stage macro-F1; exact ties choose smaller tau; exploratory")


def source_files():
    return [SOURCE / "inputs/windows.npz", SOURCE / "inputs/window_metadata.csv",
            SOURCE / "inputs/event_samples.csv", SOURCE / "tables/summary.csv",
            SOURCE / "heldout_predictions.npz", SOURCE / "features.npz",
            UNIT / "stage_geometry_decoding.py", UNIT / "prego_geometric_fits.py",
            UNIT / "prego_fixed_geometry_decoding.py", UNIT / "prego_decoding.py",
            UNIT / "motor_lfp_utils.py",
            Path(__file__).resolve(), UNIT / "plot_stage_tau_sweep.py",
            UNIT.parents[3] / "NeuralFieldManifold/fits/one_torus.py"]


def manifest():
    return {str(path.resolve()): sha256(path) for path in source_files()}


def initialize(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError("This experiment writes only to outputs/tau_sweep_v1")
    if root.exists():
        if json.loads((root / "config.json").read_text()) != config():
            raise ValueError("Existing sweep configuration differs")
        check_sources(root)
        return
    root.mkdir(parents=True)
    for name in ("checkpoints", "tables", "predictions", "models", "figures"):
        (root / name).mkdir()
    write_json(root / "config.json", config())
    write_json(root / "source_hashes.json", manifest())
    write_json(root / "environment.json", dict(python=platform.python_version(),
               numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
               joblib=joblib.__version__))


def check_sources(root):
    saved = json.loads((root / "source_hashes.json").read_text())
    actual = manifest()
    if actual != saved:
        changed = sorted(set(actual) ^ set(saved) | {p for p in actual.keys() & saved.keys()
                                                if actual[p] != saved[p]})
        raise ValueError(f"Frozen inputs or source changed: {changed}")


def load_inputs():
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv", keep_default_na=False)
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        processed = saved["processed"]
        old_clouds = saved["clouds"]
    if processed.shape != (N_WINDOWS, 300) or old_clouds.shape != (N_WINDOWS, 294, 3):
        raise ValueError("Frozen windows have unexpected shapes")
    if not np.isfinite(processed).all() or len(metadata) != N_WINDOWS:
        raise ValueError("Incomplete corrected windows")
    if not np.array_equal(metadata.window_index.to_numpy(), np.arange(N_WINDOWS)):
        raise ValueError("Window ordering changed")
    if set(metadata.stage_name) != set(STAGES) or not np.array_equal(
            metadata.groupby("stage_name").size().sort_index().to_numpy(), np.full(6, 198)):
        raise ValueError("Stage counts changed")
    if metadata.trial_index.nunique() != 198 or metadata.original_trial_number.nunique() != 198:
        raise ValueError("Trial cohort changed")
    if not (metadata.delay == "short").all():
        raise ValueError("Long-delay trial entered the cohort")
    for stage, anchor in zip(STAGES, ANCHORS):
        rows = metadata.loc[metadata.stage_name == stage]
        if not (rows.anchor_name == anchor).all() or not (rows.anchor_code == EVENT_CODES[anchor]).all():
            raise ValueError(f"Incorrect {stage} event anchor")
        if stage.startswith("post_") and not (rows.start_sample == rows.event_sample).all():
            raise ValueError(f"{stage} is not aligned to its cue offset/GO event")
        if stage.startswith("pre_") and not (rows.end_sample_exclusive == rows.event_sample).all():
            raise ValueError(f"{stage} does not end at its event")
    validate_groups(metadata.stage, metadata.heldout_fold_zero_based, metadata.trial_index)
    for i in range(N_WINDOWS):
        if not np.array_equal(cloud(processed[i], 3, "native"), old_clouds[i]):
            raise ValueError(f"Original tau=3 cloud differs at window {i}")
    return metadata, processed


def cloud(signal, tau, variant):
    if tau not in TAUS or variant not in VARIANTS or np.asarray(signal).shape != (300,):
        raise ValueError("Invalid delay, variant, or signal length")
    full = lag_embed(signal, dim=3, tau=tau)
    if full.shape != (300 - 2*tau, 3):
        raise ValueError("Wrong lag-embedding shape")
    return full if variant == "native" else full[-260:]


def canonical_variant(variant, tau):
    return "native" if variant == "matched260" and tau == 20 else variant


def checkpoint(root, variant, tau, index):
    variant = canonical_variant(variant, tau)
    return root / "checkpoints" / variant / f"tau_{tau:02d}" / f"window_{index:04d}.json"


def checked_fit(path, row, points, tau):
    result = json.loads(path.read_text())
    expected = dict(window_index=int(row["window_index"]),
                    original_trial_number=int(row["original_trial_number"]),
                    stage_name=row["stage_name"], tau_ms=tau,
                    cloud_sha256=array_hash(points))
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint identity/hash mismatch: {path}")
    return result


def fit_task(path, row, signal, tau, variant):
    points = cloud(signal, tau, variant)
    if path.exists():
        checked_fit(path, row, points, tau)
        return "reused", 0.
    with threadpool_limits(limits=1):
        result = fit_one(points, "one_torus")
    result.update(window_index=int(row["window_index"]),
                  original_trial_number=int(row["original_trial_number"]),
                  stage_name=row["stage_name"], tau_ms=tau,
                  point_count=len(points), cloud_sha256=array_hash(points))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    return result["status"], result["elapsed_seconds"]


def fit_grid(root, metadata, processed, jobs, pilot=False):
    chosen = metadata[metadata.original_trial_number.isin(PILOT_TRIALS)] if pilot else metadata
    taus = PILOT_TAUS if pilot else TAUS
    start = time.monotonic()
    fresh_times = []
    for variant in VARIANTS:
        for tau in taus:
            tasks = []
            for row in chosen.to_dict("records"):
                index = row["window_index"]
                path = checkpoint(root, variant, tau, index)
                if not path.exists():
                    tasks.append((path, row, processed[index], tau, variant))
                else:
                    checked_fit(path, row, cloud(processed[index], tau, variant), tau)
            if not tasks:
                print(f"{variant} tau={tau}: all {len(chosen)} fits already present", flush=True)
                continue
            started = time.monotonic()
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=jobs if not pilot else min(jobs, 6),
                                   return_as="generator_unordered", batch_size="auto")(
                    delayed(fit_task)(*task) for task in tasks)
                counts = {}
                for n, (status, seconds) in enumerate(results, 1):
                    counts[status] = counts.get(status, 0) + 1
                    fresh_times.append(seconds)
                    if n % 200 == 0 or n == len(tasks):
                        print(f"{variant} tau={tau}: {n}/{len(tasks)}; "
                              f"wall {time.monotonic()-started:.1f}s; statuses {counts}", flush=True)
    if pilot:
        average = float(np.mean(fresh_times)) if fresh_times else np.nan
        write_json(root / "pilot.json", dict(pilot_trials=list(PILOT_TRIALS),
                   pilot_taus=list(PILOT_TAUS), unique_new_fits=len(fresh_times),
                   fit_cpu_seconds=float(sum(fresh_times)), wall_seconds=time.monotonic()-start,
                   projected_total_fit_cpu_hours=average * 46332 / 3600))


def collect_and_decode(root, metadata, processed):
    y = metadata.stage.to_numpy()
    folds = metadata.heldout_fold_zero_based.to_numpy()
    ids = metadata.trial_index.to_numpy()
    summary, stage_rows, winners = [], [], []
    for variant in VARIANTS:
        for tau in TAUS:
            vectors, fit_rows = [], []
            for row in metadata.to_dict("records"):
                index = row["window_index"]
                points = cloud(processed[index], tau, variant)
                result = checked_fit(checkpoint(root, variant, tau, index), row, points, tau)
                vector, reason = geometry_features(result)
                vectors.append(vector)
                fit_rows.append(dict(window_index=index, original_trial_number=row["original_trial_number"],
                                     stage_name=row["stage_name"], point_count=len(points),
                                     status=result["status"], reason=result.get("reason", ""),
                                     usable=not bool(reason), unusable_reason=reason,
                                     elapsed_seconds=result["elapsed_seconds"],
                                     **result.get("summary", {})))
            x = np.asarray(vectors, dtype=float)
            if x.shape != (N_WINDOWS, 15) or np.isinf(x).any():
                raise ValueError("Invalid 15-feature table")
            stem = f"{variant}_tau_{tau:02d}"
            pd.DataFrame(x, columns=COLUMNS).assign(window_index=np.arange(N_WINDOWS)).to_csv(
                root / "tables" / f"features_{stem}.csv", index=False)
            pd.DataFrame(fit_rows).to_csv(root / "tables" / f"fits_{stem}.csv", index=False)
            try:
                predictions, models = decode(x, y, folds, ids, retain_models=True)
            except ValueError as error:
                write_json(root / "predictions" / f"{stem}_failure.json",
                           dict(variant=variant, tau_ms=tau, reason=str(error)))
                summary.append(dict(variant=variant, tau_ms=tau, macro_f1=np.nan,
                                    usable_fits=int(np.isfinite(x).all(axis=1).sum()),
                                    decoder_status="failed", decoder_reason=str(error)))
                continue
            model_dir = root / "models" / stem
            model_dir.mkdir(parents=True, exist_ok=True)
            for fold, model in models.items():
                joblib.dump(model, model_dir / f"fold_{fold}.joblib")
            np.savez_compressed(root / "predictions" / f"{stem}.npz", labels=y,
                                predictions=predictions, folds=folds, trial_ids=ids)
            result = scores(y, predictions)
            pd.DataFrame(result["confusion_counts"], index=STAGES, columns=STAGES).to_csv(
                root / "tables" / f"confusion_counts_{stem}.csv")
            pd.DataFrame(result["confusion_fraction"], index=STAGES, columns=STAGES).to_csv(
                root / "tables" / f"confusion_recall_{stem}.csv")
            summary.append(dict(variant=variant, tau_ms=tau, macro_f1=result["macro_f1"],
                                usable_fits=int(np.isfinite(x).all(axis=1).sum()),
                                decoder_status="ok", decoder_reason=""))
            stage_rows.extend(dict(variant=variant, tau_ms=tau, stage_name=stage,
                                   f1=float(result["per_stage_f1"][j]))
                              for j, stage in enumerate(STAGES))
            print(f"Decoded {stem}: macro-F1={result['macro_f1']:.6f}; "
                  f"usable={summary[-1]['usable_fits']}/{N_WINDOWS}", flush=True)
        subset = [row for row in summary if row["variant"] == variant and row["decoder_status"] == "ok"]
        if not subset:
            raise ValueError(f"No successful decoder for {variant}")
        best = sorted(subset, key=lambda row: (-row["macro_f1"], row["tau_ms"]))[0]
        winners.append(dict(variant=variant, tau_ms=best["tau_ms"], macro_f1=best["macro_f1"]))
    pd.DataFrame(summary).to_csv(root / "tables/overall_f1.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(root / "tables/stage_f1.csv", index=False)
    pd.DataFrame(winners).to_csv(root / "tables/winners.csv", index=False)
    return summary, winners


def validate(root, metadata, processed):
    check_sources(root)
    if not np.array_equal(cloud(processed[0], 20, "native"),
                          cloud(processed[0], 20, "matched260")):
        raise ValueError("Tau=20 variants should have the same cloud")
    overall = pd.read_csv(root / "tables/overall_f1.csv")
    by_stage = pd.read_csv(root / "tables/stage_f1.csv")
    winners = pd.read_csv(root / "tables/winners.csv")
    if len(overall) != 40:
        raise ValueError("Expected 40 sweep cells")
    y = metadata.stage.to_numpy()
    for row in overall.itertuples():
        stem = f"{row.variant}_tau_{row.tau_ms:02d}"
        if row.decoder_status != "ok":
            continue
        with np.load(root / "predictions" / f"{stem}.npz") as saved:
            if not np.array_equal(saved["labels"], y):
                raise ValueError("Saved labels changed")
            if not np.array_equal(saved["folds"], metadata.heldout_fold_zero_based):
                raise ValueError("Saved folds changed")
            if not np.array_equal(saved["trial_ids"], metadata.trial_index):
                raise ValueError("Saved trials changed")
            measured = scores(y, saved["predictions"])
        np.testing.assert_allclose(measured["macro_f1"], row.macro_f1, atol=1e-12)
        stage = by_stage[(by_stage.variant == row.variant) & (by_stage.tau_ms == row.tau_ms)]
        np.testing.assert_allclose(stage.set_index("stage_name").loc[list(STAGES), "f1"],
                                   measured["per_stage_f1"], atol=1e-12)
        np.testing.assert_allclose(measured["confusion_fraction"].sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(pd.read_csv(root / "tables" / f"confusion_recall_{stem}.csv",
                                            index_col=0).to_numpy(), measured["confusion_fraction"], atol=1e-12)
    tau3 = overall[(overall.variant == "native") & (overall.tau_ms == 3)]
    if len(tau3) != 1:
        raise ValueError("Native tau=3 absent")
    np.testing.assert_allclose(float(tau3.iloc[0].macro_f1), OLD_F1, atol=1e-12)
    for row in winners.itertuples():
        subset = overall[(overall.variant == row.variant) & (overall.decoder_status == "ok")]
        best = subset.sort_values(["macro_f1", "tau_ms"], ascending=[False, True]).iloc[0]
        if row.tau_ms != best.tau_ms or not np.isclose(row.macro_f1, best.macro_f1):
            raise ValueError("Winner selection changed")
    write_json(root / "validation.json", dict(cells=40, windows_per_cell=N_WINDOWS,
               native_tau3_macro_f1=float(tau3.iloc[0].macro_f1),
               old_tau3_macro_f1=OLD_F1, sources_unchanged=True,
               all_saved_predictions_reproduced=True, all_confusion_rows_normalized=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "fit", "decode", "validate", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    initialize(ROOT)
    metadata, processed = load_inputs()
    if args.phase in ("pilot", "all"):
        fit_grid(ROOT, metadata, processed, args.jobs, pilot=True)
        print((ROOT / "pilot.json").read_text(), flush=True)
    if args.phase in ("fit", "all"):
        fit_grid(ROOT, metadata, processed, args.jobs)
    if args.phase in ("decode", "all"):
        collect_and_decode(ROOT, metadata, processed)
    if args.phase in ("validate", "all"):
        validate(ROOT, metadata, processed)
    if args.phase == "all":
        from plot_stage_tau_sweep import render
        render(ROOT)


if __name__ == "__main__":
    main()
