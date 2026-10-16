#!/usr/bin/env python
"""Checkpointed pre-GO fitting and decoding. Plotting is a separate command."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from convert_motor_lfp import h5_cell_array, h5_cell_dataset, h5_string, trial_column
from motor_lfp_utils import CONVERTED_DIR, RAW_DIR, UNIT_DIR
from prego_decoding import (MAIN_METHODS, METHODS, TORUS_COLUMNS, balanced_indices,
    extract_prego, fit_geometry, fold_assignments, heldout_predictions, learn_embedding,
    permute_within_folds, recording_day, spectral_features, stable_seed)
from prego_statistics import prediction_metrics

DEFAULT_OUTPUT = UNIT_DIR / "outputs" / "prego_single_channel"
ANALYSES = ("short", "matched_short", "matched_long")


def write_json(path, value):
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_recording(path):
    with np.load(path, allow_pickle=True) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    monkey, session = str(data["monkey"]), str(data["session_id"])
    if len(np.unique(data["original_trial_number"])) != len(data["lfp"]) or np.any(data["original_trial_number"] < 0):
        raise ValueError("Duplicate or missing original behavioral trial IDs")
    row = int(data["source_pair_index"])
    go_events = []
    with h5py.File(RAW_DIR / f"Monkey{monkey}.mat") as raw:
        group = raw["Monkey"]
        if h5_string(raw, group, "Session_ID", row) != session:
            raise ValueError("Converted source row does not match raw session")
        codes_cells = h5_cell_dataset(raw, group, "TrialCodesCorr", row)
        time_cells = h5_cell_dataset(raw, group, "TrialTimesCorr", row)
        for condition in np.unique(data["source_condition_index"]):
            codes = h5_cell_array(raw, codes_cells, int(condition)-1)
            times = h5_cell_array(raw, time_cells, int(condition)-1)
            for trial in data["source_trial_index"][data["source_condition_index"] == condition]:
                cc, tt = trial_column(codes, int(trial)-1), trial_column(times, int(trial)-1)
                event = tt[cc == 207]
                if len(event) != 1 or not np.isfinite(event[0]):
                    raise ValueError("Missing/ambiguous GO event")
                go_events.append(int(event[0])-1)
    if len(set(go_events)) != 1:
        raise ValueError("Nonuniform GO timing: per-trial alignment required")
    go_sample = go_events[0]
    segments, good = extract_prego(data["lfp"], int(data["sampling_rate_hz"]), go_sample)
    metadata = dict(recording=path.stem, monkey=monkey, session=session,
                    day=recording_day(session), lfp=str(data["lfp_id"]),
                    raw_trials=len(data["lfp"]), valid_trials=len(good),
                    go_sample_zero_based=go_sample,
                    old_converted_go_sample=int(data["go_sample"]),
                    start_sample=go_sample-500, end_sample_exclusive=go_sample,
                    source_pair_index=row,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return data, segments, good, metadata


def selected_analyses(data, good, seed):
    labels = data["direction"][good].astype(int)
    delays = data["delay_label"][good].astype(str)
    candidates = {delay: np.flatnonzero(delays == delay) for delay in ("short", "long")}
    minimum = {delay: min(np.sum(labels[idx] == d) for d in range(1, 7)) for delay, idx in candidates.items()}
    selected = {}
    if minimum["short"] >= 15:
        idx = candidates["short"]
        selected["short"] = idx[balanced_indices(labels[idx], seed)]
    if min(minimum.values()) >= 15:
        for delay in ("short", "long"):
            idx = candidates[delay]
            selected["matched_"+delay] = idx[balanced_indices(labels[idx], seed, min(minimum.values()))]
    return selected, {k: int(v) for k, v in minimum.items()}


def fit_recording(path, output, config):
    started = time.monotonic()
    root = output / "cache" / path.stem
    root.mkdir(parents=True, exist_ok=True)
    complete = root / "fit_complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["source_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Raw converted source changed: {path.name}; use a new output folder")
        return previous
    data, segments, good, meta = load_recording(path)
    source_file = root / "source_metadata.json"
    if source_file.exists() and json.loads(source_file.read_text())["source_sha256"] != meta["source_sha256"]:
        raise ValueError("Source changed since partial fitting; refusing cached folds")
    write_json(source_file, meta)
    seed = stable_seed(path.stem, config["seed"])
    selections, minimum = selected_analyses(data, good, seed)
    meta["minimum_per_direction"] = minimum
    meta["status"] = "eligible" if "short" in selections else "fewer_than_15_short_trials_per_direction"
    meta["analyses"] = []
    fitted = {}
    previous_selection = {}
    for analysis, indices in selections.items():
        folder = root / analysis
        folder.mkdir(exist_ok=True)
        raw_indices = good[indices]
        x = segments[indices]
        y = data["direction"][raw_indices].astype(int)
        folds = fold_assignments(y, seed)
        power, frequency, bands = spectral_features(x, meta["monkey"])
        selection_hash = hashlib.sha256(raw_indices.tobytes()).hexdigest()
        torus = []
        for fold in range(5):
            cache = folder / f"fold_{fold}.npz"
            training_ids = raw_indices[folds != fold].tolist()
            if cache.exists():
                with np.load(cache) as saved:
                    features, success, nfev = saved["features"], saved["success"], saved["nfev"]
                    costs, errors = saved["cost"], saved["mse"]
                embedding = json.loads((folder / f"fold_{fold}.json").read_text())
                if embedding["training_trial_indices"] != training_ids:
                    raise ValueError("Stale fold cache")
            elif selection_hash in previous_selection:
                other = root / previous_selection[selection_hash]
                with np.load(other / f"fold_{fold}.npz") as saved:
                    features, success, nfev = saved["features"], saved["success"], saved["nfev"]
                    costs, errors = saved["cost"], saved["mse"]
                embedding = json.loads((other / f"fold_{fold}.json").read_text())
            else:
                embedding = learn_embedding(x[folds != fold], seed+fold, config["embedding_bootstraps"])
                embedding["training_trial_indices"] = training_ids
                coordinate_key = json.dumps({key: embedding[key] for key in ("dimension", "tau", "center", "components")}, sort_keys=True)
                features, info = [], []
                for segment, raw_index in zip(x, raw_indices):
                    key = (coordinate_key, int(raw_index))
                    if key not in fitted:
                        fitted[key] = fit_geometry(segment, embedding, stable_seed(int(raw_index), seed))
                    vector, details = fitted[key]
                    features.append(vector)
                    info.append(details)
                features = np.array(features)
                success = np.array([d["success"] for d in info])
                nfev = np.array([d["nfev"] for d in info])
                costs = np.array([d["cost"] for d in info])
                errors = np.array([d["mse"] for d in info])
            if not cache.exists():
                np.savez_compressed(cache, features=features, success=success, nfev=nfev,
                                    cost=costs, mse=errors, raw_trial_indices=raw_indices)
                write_json(folder / f"fold_{fold}.json", embedding)
            torus.append(features)
        torus = np.array(torus)
        np.savez_compressed(folder / "features.npz", power=power, frequency=frequency,
                            bands=bands, torus=torus, labels=y, folds=folds,
                            raw_trial_indices=raw_indices,
                            original_trial_number=data["original_trial_number"][raw_indices])
        spectral = pd.DataFrame({"raw_trial_index": raw_indices, "direction": y, "fold": folds,
                                  "original_trial_number": data["original_trial_number"][raw_indices]})
        for name, values in (("peak_power", power), ("peak_frequency", frequency), ("band", bands)):
            for j in range(values.shape[1]):
                spectral[f"{name}_{j+1}"] = values[:, j]
        spectral.to_csv(folder / "spectral_features.csv", index=False)
        frames = []
        for fold in range(5):
            frame = pd.DataFrame(torus[fold], columns=TORUS_COLUMNS)
            frame.insert(0, "raw_trial_index", raw_indices)
            frame.insert(1, "embedding_fold", fold)
            frame.insert(2, "held_out", folds == fold)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(folder / "geometry_features.csv", index=False)
        meta["analyses"].append(analysis)
        previous_selection[selection_hash] = analysis
    meta["fit_seconds"] = time.monotonic()-started
    write_json(complete, meta)
    return meta


def decode_recording(path, output, config):
    root = output / "cache" / path.stem
    meta = json.loads((root / "fit_complete.json").read_text())
    completed = root / "decode_complete.json"
    if completed.exists():
        return json.loads(completed.read_text())
    statuses = []
    for analysis in meta["analyses"]:
        folder = root / analysis
        with np.load(folder / "features.npz") as frozen:
            d = {key: frozen[key] for key in frozen.files}
        try:
            predictions = heldout_predictions(d["power"], d["frequency"], d["bands"], d["torus"], d["labels"], d["folds"])
        except ValueError as error:
            statuses.append(dict(analysis=analysis, status="excluded_geometry", reason=str(error)))
            continue
        rows = []
        for method, predicted in predictions.items():
            rows.append(pd.DataFrame(dict(recording=path.stem, monkey=meta["monkey"], day=meta["day"],
                analysis=analysis, method=method, raw_trial_index=d["raw_trial_indices"],
                original_trial_number=d["original_trial_number"], fold=d["folds"],
                true_direction=d["labels"], predicted_direction=predicted)))
        pd.concat(rows).to_csv(folder / "predictions.csv", index=False)
        if analysis == "short":
            null_file = folder / "null_scores.csv"
            if not null_file.exists():
                null_rows = []
                for permutation in range(config["null_permutations"]):
                    seed = stable_seed(f"{path.stem}-null-{permutation}", config["seed"])
                    labels = permute_within_folds(d["labels"], d["folds"], seed)
                    null = heldout_predictions(d["power"], d["frequency"], d["bands"], d["torus"], labels, d["folds"], MAIN_METHODS)
                    for method, pred in null.items():
                        score, _, _ = prediction_metrics(labels, pred)
                        null_rows.append(dict(recording=path.stem, monkey=meta["monkey"], day=meta["day"],
                                              method=method, permutation=permutation, macro_f1=score))
                pd.DataFrame(null_rows).to_csv(null_file, index=False)
        statuses.append(dict(analysis=analysis, status="included", n_trials=len(d["labels"])))
    result = dict(recording=path.stem, analyses=statuses)
    write_json(completed, result)
    return result


def run_one(path, output, config, stage, wait_for_fits=False):
    with threadpool_limits(limits=1):
        result = None
        if stage in ("all", "fit"):
            result = fit_recording(path, output, config)
        if stage in ("all", "decode"):
            started = time.monotonic()
            while wait_for_fits and not (output / "cache" / path.stem / "fit_complete.json").exists():
                if time.monotonic()-started > 7200:
                    raise TimeoutError(f"Fitting did not finish: {path.stem}")
                time.sleep(2)
            result = decode_recording(path, output, config)
        print(f"{stage}: {path.stem}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=["fit", "decode", "all"], default="all")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--monkeys", nargs="+", choices=["M", "T"], default=["M", "T"])
    parser.add_argument("--null-permutations", type=int, default=200)
    parser.add_argument("--embedding-bootstraps", type=int, default=120)
    parser.add_argument("--wait-for-fits", action="store_true", help="Decode as a separately running fit stage completes")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    core_hash = hashlib.sha256((UNIT_DIR / "prego_decoding.py").read_bytes()).hexdigest()
    config = dict(version=1, seed=42, core_sha256=core_hash, window_ms=[-500, 0],
                  null_permutations=args.null_permutations, embedding_bootstraps=args.embedding_bootstraps,
                  minimum_trials_per_direction=15, raw_event_alignment=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Configuration changed. Use a separate output folder; refusing stale caches.")
    write_json(config_path, config)
    paths = sorted(p for p in CONVERTED_DIR.glob("*.npz") if p.stem[6] in args.monkeys)
    if args.max_recordings is not None:
        paths = paths[:args.max_recordings]
    run_names = ("all", "fit", "decode") if args.stage == "all" else (args.stage,)
    for name in run_names:
        write_json(output / f"{name}_run.json", dict(status="running", recordings=[p.stem for p in paths]))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(run_one)(p, output, config, args.stage, args.wait_for_fits) for p in paths)
    for name in run_names:
        write_json(output / f"{name}_run.json", dict(status="complete", recordings=[p.stem for p in paths],
                                                   n_recordings=len(results)))
    print(f"Completed {args.stage}: {len(results)} audited recordings. {output}")


if __name__ == "__main__":
    main()
