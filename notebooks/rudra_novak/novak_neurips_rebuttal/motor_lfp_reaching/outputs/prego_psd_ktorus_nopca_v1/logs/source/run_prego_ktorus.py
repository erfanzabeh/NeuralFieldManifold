#!/usr/bin/env python
"""Isolated, resumable PSD/K-torus experiment on the audited pre-GO cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
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
import uuid

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from motor_lfp_utils import UNIT_DIR, CONVERTED_DIR, RAW_DIR
from prego_decoding import fold_assignments, permute_within_folds, spectral_features, stable_seed
from prego_ktorus import MAIN_METHODS, METHODS, decode, fit_trial, learn_geometry
from prego_statistics import prediction_metrics
from run_prego_single_channel import load_recording, selected_analyses

PREVIOUS = UNIT_DIR / "outputs" / "prego_single_channel"
DEFAULT_OUTPUT = UNIT_DIR / "outputs" / "prego_psd_ktorus_nopca_v1"
SOURCE_NAMES = ["prego_ktorus.py", "run_prego_ktorus.py", "prego_decoding.py",
                "prego_statistics.py", "run_prego_single_channel.py", "motor_lfp_utils.py",
                "select_trace_embedding_parameters.py", "convert_motor_lfp.py",
                "PREGO_PSD_KTORUS_NOPCA_SPEC.md"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.stem+"."+uuid.uuid4().hex+".npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_table(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validate_output(path):
    path, old = Path(path).resolve(), PREVIOUS.resolve()
    if path == old or old in path.parents or path in old.parents:
        raise ValueError("Output must be a separate experiment, never inside or above the predecessor")


def initialize_run(root, config, provenance, resume=False):
    root = Path(root)
    validate_output(root)
    if root.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {root}; use explicit --resume")
        if not (root / "manifest.json").exists():
            raise ValueError("Existing directory is not a registered run")
        if json.loads((root / "config.json").read_text()) != config:
            raise ValueError("Configuration changed; create a new experiment version")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["provenance"] != provenance:
            raise ValueError("Source, data, or predecessor changed; refusing stale checkpoints")
        return manifest
    if resume:
        raise ValueError("Cannot resume a nonexistent run")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("cache", "tables", "plots", "diagnostics", "logs"):
        (root / name).mkdir()
    manifest = dict(experiment_id=root.name, parent_experiment=PREVIOUS.name,
                    created_utc=utc(), state="initialized", provenance=provenance, commands=[])
    write_json(root / "config.json", config)
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_selection(current, previous):
    for key in ("raw_trial_indices", "folds", "labels"):
        if not np.array_equal(current[key], previous[key]):
            raise ValueError(f"Changed audited selection: {key}")


def old_output_hashes():
    paths = [PREVIOUS / "config.json"]
    for folder in ("tables", "panels"):
        paths.extend(sorted((PREVIOUS / folder).glob("*")))
    return {str(p.relative_to(PREVIOUS)): sha256(p) for p in paths if p.is_file()}


def provenance_for(paths):
    return dict(source={name: sha256(UNIT_DIR / name) for name in SOURCE_NAMES},
                converted={p.name: sha256(p) for p in paths},
                raw={p.name: sha256(p) for p in sorted(RAW_DIR.glob("Monkey*.mat"))},
                predecessor_trials={str(p.relative_to(PREVIOUS)): sha256(p)
                                    for p in sorted((PREVIOUS / "cache").glob("*/short/features.npz"))},
                predecessor=old_output_hashes())


def fit_recording(path, root, config):
    folder = root / "cache" / path.stem
    folder.mkdir(exist_ok=True)
    done = folder / "fit_complete.json"
    if done.exists():
        return json.loads(done.read_text())
    started = time.monotonic()
    data, segments, good, meta = load_recording(path)
    seed = stable_seed(path.stem, config["seed"])
    selection, minimum = selected_analyses(data, good, seed)
    meta.update(minimum_short=minimum["short"], minimum_long=minimum["long"],
                status="eligible" if "short" in selection else "insufficient_short_trials")
    if "short" not in selection:
        write_json(done, meta)
        return meta
    indices = selection["short"]
    raw_indices, x = good[indices], segments[indices]
    y = data["direction"][raw_indices].astype(int)
    folds = fold_assignments(y, seed)
    power, frequency, bands = spectral_features(x, meta["monkey"])
    current = dict(raw_trial_indices=raw_indices, labels=y, folds=folds)
    old_file = PREVIOUS / "cache" / path.stem / "short" / "features.npz"
    with np.load(old_file) as old:
        verify_selection(current, old)
        for key, values in (("power", power), ("frequency", frequency), ("bands", bands)):
            if not np.array_equal(values, old[key]):
                raise ValueError(f"Spectral baseline changed for {path.stem}: {key}")
    selection_hash = hashlib.sha256(raw_indices.tobytes()+folds.tobytes()+y.tobytes()).hexdigest()
    save_arrays(folder / "trials.npz", **current, power=power, frequency=frequency, bands=bands,
                original_trial_number=data["original_trial_number"][raw_indices])
    meta.update(n_trials=len(y), selection_hash=selection_hash, predecessor_features_sha256=sha256(old_file))
    unresolved, fit_cache = [], {}
    for fold in range(5):
        parameters_path = folder / f"fold_{fold}.json"
        feature_path = folder / f"fold_{fold}.npz"
        training_ids = raw_indices[folds != fold].tolist()
        if parameters_path.exists():
            embedding = json.loads(parameters_path.read_text())
            if embedding["selection_hash"] != selection_hash or embedding["training_trial_indices"] != training_ids:
                raise ValueError("Checkpoint has different training trials")
        else:
            embedding = learn_geometry(x[folds != fold], seed+fold, config["embedding_bootstraps"])
            embedding.update(training_trial_indices=training_ids, selection_hash=selection_hash)
            write_json(parameters_path, embedding)
        if embedding["status"] != "ok":
            unresolved.append(dict(fold=fold, reason=embedding["status"]))
            continue
        if feature_path.exists():
            with np.load(feature_path) as saved:
                if str(saved["selection_hash"]) != selection_hash:
                    raise ValueError("Stale fitted trial features")
            continue
        coordinate_key = json.dumps({k: embedding[k] for k in ("K", "dimension", "tau", "frequencies")}, sort_keys=True)
        features, info = [], []
        for segment, raw_id in zip(x, raw_indices):
            key = (coordinate_key, int(raw_id))
            if key not in fit_cache:
                fit_cache[key] = fit_trial(segment, embedding)
            vector, details = fit_cache[key]
            features.append(vector)
            info.append(details)
        features = np.array(features)
        k, m = embedding["K"], embedding["dimension"]
        save_arrays(feature_path, features=features, selection_hash=selection_hash,
                    raw_trial_indices=raw_indices, feature_names=np.array(embedding["feature_names"]),
                    success=np.array([r["success"] for r in info]),
                    coefficients=np.array([r.get("coefficients", np.full((1+2*k, m), np.nan)) for r in info]),
                    frequencies=np.array([r.get("frequencies", np.full(k, np.nan)) for r in info]))
        diagnostics = pd.DataFrame([{key: value for key, value in r.items()
                                     if key not in ("coefficients", "frequencies", "rmse_by_coordinate", "phase_largest_gaps")}
                                    for r in info])
        diagnostics.insert(0, "raw_trial_index", raw_indices)
        diagnostics.insert(1, "held_out", folds == fold)
        for j in range(k):
            diagnostics[f"phase_gap_mode_{j+1}"] = [r.get("phase_largest_gaps", [np.nan]*k)[j] for r in info]
        save_table(folder / f"fold_{fold}_diagnostics.csv", diagnostics)
    meta.update(unresolved_folds=unresolved, geometry_evaluable=not unresolved,
                fit_seconds=time.monotonic()-started)
    if not unresolved:
        for fold in range(5):
            with np.load(folder / f"fold_{fold}.npz") as saved:
                if not np.isfinite(saved["features"][folds != fold]).any(axis=0).all():
                    meta["geometry_evaluable"] = False
                    meta["unresolved_folds"].append(dict(fold=fold, reason="no_usable_training_geometry"))
    write_json(done, meta)
    return meta


def decode_recording(path, root, config):
    folder = root / "cache" / path.stem
    meta = json.loads((folder / "fit_complete.json").read_text())
    done = folder / "decode_complete.json"
    if done.exists():
        return json.loads(done.read_text())
    if meta["status"] != "eligible":
        write_json(done, meta)
        return meta
    started = time.monotonic()
    with np.load(folder / "trials.npz") as saved:
        d = {k: saved[k] for k in saved.files}
    usable = meta["geometry_evaluable"]
    geometry = []
    for fold in range(5):
        if usable:
            with np.load(folder / f"fold_{fold}.npz") as saved:
                geometry.append(saved["features"])
        else:
            geometry.append(np.zeros((len(d["labels"]), 1)))
    methods = METHODS if usable else MAIN_METHODS[:4]
    predicted = decode(d["power"], d["frequency"], d["bands"], geometry, d["labels"], d["folds"], methods)
    frames = []
    for method, values in predicted.items():
        frames.append(pd.DataFrame(dict(recording=path.stem, monkey=meta["monkey"], day=meta["day"],
                     method=method, geometry_evaluable=usable, raw_trial_index=d["raw_trial_indices"],
                     original_trial_number=d["original_trial_number"], fold=d["folds"],
                     true_direction=d["labels"], predicted_direction=values)))
    save_table(folder / "predictions.csv", pd.concat(frames, ignore_index=True))
    if usable:
        null_path = folder / "null_scores.csv"
        null_rows = pd.read_csv(null_path).to_dict("records") if null_path.exists() else []
        completed = len(null_rows)//len(MAIN_METHODS)
        for permutation in range(completed, config["null_permutations"]):
            seed = stable_seed(f"{path.stem}-null-{permutation}", config["seed"])
            labels = permute_within_folds(d["labels"], d["folds"], seed)
            null = decode(d["power"], d["frequency"], d["bands"], geometry, labels, d["folds"], MAIN_METHODS)
            for method, values in null.items():
                score, _, _ = prediction_metrics(labels, values)
                null_rows.append(dict(recording=path.stem, monkey=meta["monkey"], day=meta["day"],
                                      method=method, permutation=permutation, macro_f1=score))
            if (permutation+1) % 20 == 0:
                save_table(null_path, pd.DataFrame(null_rows))
        save_table(null_path, pd.DataFrame(null_rows))
    meta["decode_seconds"] = time.monotonic()-started
    write_json(done, meta)
    return meta


def run_one(path, root, config, stage):
    with threadpool_limits(limits=1):
        if stage in ("fit", "all"):
            fit_recording(path, root, config)
        if stage in ("decode", "all"):
            result = decode_recording(path, root, config)
        else:
            result = json.loads((root / "cache" / path.stem / "fit_complete.json").read_text())
        print(f"{stage}: {path.stem}: {result['status']}; geometry={result.get('geometry_evaluable', False)}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["fit", "decode", "all"], default="all")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--limit", type=int, help="Process a checkpointed subset; never declares the experiment complete")
    args = parser.parse_args()
    root = args.output.resolve()
    validate_output(root)
    if root.exists() and not args.resume:
        raise FileExistsError(f"Refusing existing output: {root}")
    paths = sorted(CONVERTED_DIR.glob("*.npz"))
    if len(paths) != 341:
        raise ValueError(f"Expected audited 341 source recordings, got {len(paths)}")
    config = dict(schema_version=1, seed=42, window_ms=[-500, 0], delay="short", directions=6,
                  embedding_bootstraps=120, peak_support=.5, dimension_rule="2*K+1",
                  tau_range_samples=[1, 100], minimum_embedding_points=300, maximum_condition=100.,
                  frequency_fit_half_width_hz=2., fit_max_nfev=120,
                  null_permutations=200, minimum_trials_per_direction=15,
                  geometry_model="affine_product_of_circles", pca=False, pinn=False,
                  feature_schema="per_mode_full_gram_upper_triangle_plus_coordinate_rmse_plus_normalized_error",
                  classifier="median_imputer_standard_scaler_shrinkage_lda", source_recordings=341)
    print("Hashing source code, raw data, converted recordings, and predecessor outputs...", flush=True)
    provenance = provenance_for(paths)
    manifest = initialize_run(root, config, provenance, args.resume)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest.update(state="running", pid=os.getpid(), updated_utc=utc())
        manifest["commands"].append(dict(utc=utc(), argv=sys.argv, stage=args.stage, limit=args.limit, jobs=args.jobs))
        manifest["environment"] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "joblib", "h5py"]})
        manifest["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UNIT_DIR, text=True).strip()
        snapshot = root / "logs" / "source"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in SOURCE_NAMES:
                shutil.copy2(UNIT_DIR / name, snapshot / name)
        write_json(root / "manifest.json", manifest)
        try:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                Parallel(n_jobs=args.jobs)(delayed(run_one)(p, root, config, args.stage) for p in paths[:args.limit])
            if old_output_hashes() != provenance["predecessor"]:
                raise RuntimeError("Predecessor outputs changed during execution")
            manifest.update(state="partial" if args.limit else "fits_complete" if args.stage == "fit" else "decoding_complete",
                            updated_utc=utc(), predecessor_unchanged=True)
        except BaseException as error:
            manifest.update(state="failed", updated_utc=utc(), error=repr(error))
            write_json(root / "manifest.json", manifest)
            raise
        write_json(root / "manifest.json", manifest)
        print(f"{manifest['state']}: {root}", flush=True)


if __name__ == "__main__":
    main()
