#!/usr/bin/env python
"""Run the isolated OLS audit. This entrypoint never renders or decodes."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from prego_ols import audit_fold, default_config

BASE = Path(__file__).resolve().parent
PREVIOUS = BASE / "outputs/prego_single_channel"
KPREVIOUS = BASE / "outputs/prego_psd_ktorus_nopca_v1"
DEFAULT_OUTPUT = BASE / "outputs/prego_ols_audit_v1"
TRIAL_KEYS = ("raw_trial_indices", "labels", "folds", "original_trial_number")
SOURCE_NAMES = ("prego_ols.py", "run_prego_ols_audit.py", "run_prego_single_channel.py",
                "motor_lfp_utils.py", "prego_decoding.py", "convert_motor_lfp.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def clean(value):
    if isinstance(value, (np.ndarray, pd.Series)):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def save_table(path, frame):
    temporary = path.with_name(path.name+".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_arrays(path, **arrays):
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def artifact_hashes(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name not in ("complete.json", "failure.json") and not p.name.endswith(".tmp")}


def verify_artifacts(folder, expected):
    for name, digest in expected.items():
        path = folder / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Changed or missing checkpoint: {path}")


def initialize(root, config, provenance, resume):
    environment = dict(python=sys.version, platform=platform.platform(), packages={
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "h5py", "joblib")})
    if root.exists():
        if not resume:
            raise FileExistsError(f"Refusing overwrite: {root}; explicit --resume required")
        manifest = json.loads((root / "manifest.json").read_text())
        if (json.loads((root / "config.json").read_text()) != config
                or manifest["provenance"] != provenance or manifest["environment"] != environment):
            raise ValueError("Configuration, code, inputs or environment changed; use a new run version")
        return manifest
    if resume:
        raise ValueError("Cannot resume a nonexistent run")
    root.mkdir(parents=True)
    for name in ("cache", "tables", "plots", "logs", "source"):
        (root / name).mkdir()
    manifest = dict(created_utc=utc(), state="initialized", experiment=root.name,
                    provenance=provenance, environment=environment, commands=[])
    write_json(root / "config.json", config)
    write_json(root / "manifest.json", manifest)
    return manifest


def cohort():
    paths = sorted((PREVIOUS / "cache").glob("*/short/features.npz"))
    rows = []
    for path in paths:
        with np.load(path) as saved:
            rows.append(dict(recording=path.parents[1].name, monkey=path.parents[1].name[6],
                             n_trials=len(saved["labels"])))
    result = pd.DataFrame(rows)
    if result.groupby("monkey").size().to_dict() != {"M": 115, "T": 95}:
        raise ValueError("Saved short-delay cohort is not the approved 115 M / 95 T recordings")
    return result


def provenance_for(records):
    inputs = [BASE / "lfp_converted_data" / (r+".npz") for r in records]
    inputs += sorted((BASE / "raw_data").glob("Monkey*.mat"))
    # Snapshot every existing result file, not only the files used by this audit.
    prior = [p for root in (PREVIOUS, KPREVIOUS) for p in sorted(root.rglob("*")) if p.is_file()]
    return dict(source={name: sha256(BASE / name) for name in SOURCE_NAMES},
                inputs={str(p): sha256(p) for p in inputs},
                previous_outputs={str(p): sha256(p) for p in prior})


def validate_trials(data, good, saved):
    indices = saved["raw_trial_indices"]
    if not np.isin(indices, good).all() or len(np.unique(indices)) != len(indices):
        raise ValueError("Saved trial identities are invalid or duplicated")
    if not np.array_equal(data["direction"][indices], saved["labels"]):
        raise ValueError("Saved direction labels changed")
    if not np.array_equal(data["original_trial_number"][indices], saved["original_trial_number"]):
        raise ValueError("Original trial numbers changed")
    if not np.all(data["delay_label"][indices].astype(str) == "short"):
        raise ValueError("Audit requires only short-delay trials")
    labels, counts = np.unique(saved["labels"], return_counts=True)
    if not np.array_equal(labels, np.arange(1, 7)) or len(set(counts)) != 1:
        raise ValueError("Saved trials must be balanced across six directions")
    if set(saved["folds"]) != set(range(5)):
        raise ValueError("Expected the five saved outer folds")


def process_recording(recording, root, config):
    folder = root / "cache" / recording
    folder.mkdir(exist_ok=True)
    completed = folder / "complete.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        verify_artifacts(folder, result["artifacts"])
        return result
    started = time.monotonic()
    try:
        # Reuse only the audited raw-data loader, never its fitting/decoding functions.
        from run_prego_single_channel import load_recording
        from motor_lfp_utils import detrend_zscore, bandpass_filter
        data, segments, good, meta = load_recording(BASE / "lfp_converted_data" / (recording+".npz"))
        with np.load(PREVIOUS / "cache" / recording / "short/features.npz") as old:
            saved = {key: old[key] for key in TRIAL_KEYS}
        with np.load(KPREVIOUS / "cache" / recording / "trials.npz") as prior:
            for key in TRIAL_KEYS:
                if not np.array_equal(saved[key], prior[key]):
                    raise ValueError(f"Predecessor selections disagree: {key}")
        validate_trials(data, good, saved)
        lookup = {int(raw): i for i, raw in enumerate(good)}
        selected = segments[[lookup[int(raw)] for raw in saved["raw_trial_indices"]]]
        # Same preprocessing as the previous run; unlike its helper, filter errors are fatal.
        x = np.array([bandpass_filter(detrend_zscore(s), fs=1000, low=2, high=55).astype(np.float32)
                      for s in selected])
        if x.shape != (len(saved["labels"]), 500) or not np.isfinite(x).all():
            raise ValueError("Invalid preprocessed epochs")
        if not (folder / "trials.npz").exists():
            save_arrays(folder / "trials.npz", segments=x, **saved)
        else:
            with np.load(folder / "trials.npz") as check:
                for key, value in dict(segments=x, **saved).items():
                    np.testing.assert_array_equal(check[key], value)
        meta.update(n_trials=len(x), status="ok", short_delay_ms=1000 if meta["monkey"] == "M" else 700)
        write_json(folder / "metadata.json", meta)
        seed = int(hashlib.sha256(recording.encode()).hexdigest()[:8], 16) % (2**31-10)
        for fold in range(5):
            destination = folder / f"fold_{fold}"
            destination.mkdir(exist_ok=True)
            checkpoint = destination / "complete.json"
            if checkpoint.exists():
                verify_artifacts(destination, json.loads(checkpoint.read_text())["artifacts"])
                continue
            with threadpool_limits(limits=1):
                result = audit_fold(x, saved["labels"], saved["folds"], fold, config, seed=seed)
            for key in ("validation", "inner_scores", "metrics"):
                table = result[key]
                table.insert(0, "outer_fold", fold)
                table.insert(0, "recording", recording)
                save_table(destination / (key+".csv"), table)
            write_json(destination / "models.json", {key: result[key] for key in ("selection", "models", "poles", "inner_splits")})
            save_arrays(destination / "predictions.npz", test_indices=result["test_indices"], **result["predictions"])
            write_json(checkpoint, dict(artifacts=artifact_hashes(destination)))
        meta.update(seconds=time.monotonic()-started, artifacts=artifact_hashes(folder))
        write_json(completed, meta)
        print(f"Complete {recording}: {meta['n_trials']} trials, {meta['seconds']:.1f} s", flush=True)
        return meta
    except Exception as error:
        failure = dict(recording=recording, monkey=recording[6], status="failed", reason=str(error),
                       traceback=traceback.format_exc(), seconds=time.monotonic()-started)
        write_json(folder / "failure.json", failure)
        print(f"FAILED {recording}: {error}", flush=True)
        return failure


def benchmark_records(frame):
    selected = []
    for _, group in frame.groupby("monkey"):
        median = group.assign(distance=(group.n_trials-group.n_trials.median()).abs()).sort_values(["distance", "recording"]).iloc[0]
        largest = group[group.recording != median.recording].sort_values(["n_trials", "recording"], ascending=[False, True]).iloc[0]
        selected.extend([median.recording, largest.recording])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.parent != DEFAULT_OUTPUT.parent or not root.name.startswith("prego_ols_audit_"):
        raise ValueError("Use a new sibling prego_ols_audit_* experiment, never a predecessor path")
    frame, config = cohort(), default_config()
    print(f"Hashing raw inputs, analysis sources and all predecessor artifacts; {len(frame)} recordings", flush=True)
    provenance = provenance_for(frame.recording)
    manifest = initialize(root, config, provenance, args.resume)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in SOURCE_NAMES:
            target = root / "source" / name
            if not target.exists():
                shutil.copyfile(BASE / name, target)
        save_table(root / "tables/cohort.csv", frame)
        selected = benchmark_records(frame) if args.benchmark else frame.recording.tolist()
        manifest.update(state="benchmark_running" if args.benchmark else "analysis_running")
        manifest["commands"].append(dict(utc=utc(), argv=sys.argv, recordings=len(selected), jobs=args.jobs))
        write_json(root / "manifest.json", manifest)
        started = time.monotonic()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=min(args.jobs, len(selected)))(delayed(process_recording)(r, root, config) for r in selected)
        elapsed = time.monotonic()-started
        status = pd.DataFrame([{k: v for k, v in r.items() if k not in ("artifacts", "traceback")} for r in results])
        save_table(root / "tables" / ("benchmark_recordings.csv" if args.benchmark else "recording_status.csv"), status)
        if args.benchmark:
            estimate = status.seconds.mean() * len(frame) / args.jobs / 60
            write_json(root / "benchmark.json", dict(wall_seconds=elapsed, recordings=selected, jobs=args.jobs,
                projected_compute_minutes=estimate, provisional_range_minutes=[estimate, estimate*2],
                selection="median and maximum trial count per animal; no model scores used"))
            print(f"Benchmark: {elapsed:.1f}s elapsed; projected {estimate:.1f}-{estimate*2:.1f} min at {args.jobs} workers", flush=True)
        manifest.update(state=("benchmark_complete" if args.benchmark else "analysis_complete")
                        if (status.status == "ok").all() else "completed_with_failures", updated_utc=utc())
        write_json(root / "manifest.json", manifest)
        print(f"State: {manifest['state']}; {len(status)} recordings", flush=True)


if __name__ == "__main__":
    main()
