"""Run only the approved fixed-delay geometric fits, with resumable checkpoints."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed, parallel_config

from prego_geometric_fits import (
    INPUT, OUTPUT, RECORDING, MODELS, UNIT, array_hash, fit_one, make_clouds,
    sha256, write_json,
)


def initialize(root, config, provenance, resume):
    root = Path(root)
    if root.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {root}; use --resume")
        if (json.loads((root/"config.json").read_text()) != config or
                json.loads((root/"provenance.json").read_text()) != provenance):
            raise ValueError("Changed configuration, inputs, or fitting source; use a new experiment")
    else:
        root.mkdir(parents=True)
        write_json(root/"config.json", config)
        write_json(root/"provenance.json", provenance)
    for sub in ("checkpoints", "tables", "source", "inputs", "logs"):
        (root/sub).mkdir(exist_ok=True)


def load_inputs(source):
    source = Path(source)
    with np.load(source/"selected_raw_prego_epochs.npz", allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    trials = pd.read_csv(source/"selected_trials.csv")
    alignment = json.loads((source/"audited_alignment.json").read_text())
    if (alignment["recording"] != RECORDING or alignment["sampling_rate_hz"] != 1000 or
            alignment["start_sample"] != 4000 or alignment["end_sample_exclusive"] != 4500 or
            alignment["status"] != "ok"):
        raise ValueError("Unexpected audited alignment")
    if data["segments"].shape != (198, 500) or len(trials) != 198:
        raise ValueError("Expected exactly 198 complete trial epochs")
    if not (trials.delay == "short").all() or trials.groupby("direction").size().to_dict() != dict.fromkeys(range(1, 7), 33):
        raise ValueError("Expected 33 short trials in each of six directions")
    for column, field in (("row_index", None), ("raw_trial_index", "raw_trial_indices"),
                          ("original_trial_number", "original_trial_number"),
                          ("direction", "labels"), ("heldout_fold_zero_based", "folds")):
        np.testing.assert_array_equal(trials[column], np.arange(198) if field is None else data[field])
    if trials.raw_trial_index.nunique() != 198 or trials.original_trial_number.nunique() != 198:
        raise ValueError("Repeated trial identities")
    np.testing.assert_array_equal(data["time_ms"], np.arange(-500, 0))
    # This trusted local conversion stores delay labels as an object array.
    with np.load(source/"full_converted_recording.npz", allow_pickle=True) as raw:
        ids = data["raw_trial_indices"]
        np.testing.assert_array_equal(data["segments"], raw["lfp"] [ids, 4000:4500])
        np.testing.assert_array_equal(data["labels"], raw["direction"][ids])
        np.testing.assert_array_equal(data["original_trial_number"], raw["original_trial_number"][ids])
        if not (raw["delay_label"][ids] == "short").all():
            raise ValueError("Source trials are not all short delay")
    return data, trials, alignment


def validate_checkpoint(path, row, model, points):
    result = json.loads(Path(path).read_text())
    if (result.get("row_index") != row or result.get("model") != model or
            result.get("cloud_sha256") != array_hash(points)):
        raise ValueError(f"Mismatched checkpoint: {path}")
    if result.get("status") not in ("returned", "failed"):
        raise ValueError(f"Incomplete checkpoint: {path}")
    return result


def worker(root, row, model, points, trial):
    result = fit_one(points, model)
    result.update(row_index=int(row), original_trial_number=int(trial["original_trial_number"]),
                  raw_trial_index=int(trial["raw_trial_index"]), direction=int(trial["direction"]))
    write_json(root/"checkpoints"/f"trial_{row:03d}_{model}.json", result)
    return row, model, result["status"], result["elapsed_seconds"]


def collect_tables(root, trials):
    rows = []
    for trial in trials.to_dict("records"):
        index = int(trial["row_index"])
        for model in MODELS:
            path = root/"checkpoints"/f"trial_{index:03d}_{model}.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            row = dict(trial, model=model, status=result["status"], reason=result["reason"],
                       elapsed_seconds=result["elapsed_seconds"], cloud_sha256=result["cloud_sha256"],
                       warnings=";".join(result["warnings"]))
            row.update(result.get("summary", {}))
            if result["status"] == "failed":
                row.update(optimizer_success=False, flags="fit_failed")
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(root/"tables/trial_fits.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-trials", type=int, default=None, help="Benchmark first N; resume later without this limit")
    args = parser.parse_args()
    data, trials, alignment = load_inputs(INPUT)
    config = dict(recording=RECORDING, n_trials=198, fs_hz=1000, epoch_samples=[4000, 4500],
                  dimension=3, tau_samples=16, tau_ms=16, points_per_trial=468,
                  models={"one_torus": dict(lam=.1, hole_ratio=.5, n_modes=3, max_nfev=6000),
                          "two_torus": dict(lam=.1, hole_ratio=.75, n_modes=3, max_nfev=6000)},
                  loss="huber", f_scale=1., decoding=False, pca_reduction=False,
                  preprocessing="existing per-trial linear detrend, median/MAD normalize, 4th-order zero-phase 2-55 Hz",
                  context_only_psd_peak_hz=22., context_only_candidate_K=1)
    source_files = [Path(__file__), UNIT/"prego_geometric_fits.py", UNIT/"motor_lfp_utils.py",
                    UNIT/"select_trace_embedding_parameters.py"]
    for model in MODELS:
        source_files.append(Path(importlib.import_module(f"NeuralFieldManifold.fits.{model}").__file__))
    inputs = [INPUT/k for k in ("selected_raw_prego_epochs.npz", "selected_trials.csv",
                                "audited_alignment.json", "full_converted_recording.npz")]
    provenance = {str(p.resolve()): sha256(p) for p in source_files+inputs}
    initialize(args.output, config, provenance, args.resume)
    root = args.output
    for p in source_files:
        dest = root/"source"/p.name
        if not dest.exists():
            shutil.copy2(p, dest)
    for p in inputs[:3]:
        dest = root/"inputs"/p.name
        if not dest.exists():
            shutil.copy2(p, dest)
    preserved = root/"prior_outputs_manifest.json"
    if not preserved.exists():
        print("Hashing prior result files (read-only)", flush=True)
        old_files = [p for p in (UNIT/"outputs").rglob("*") if p.is_file()
                     and root.resolve() not in p.resolve().parents and "__pycache__" not in p.parts]
        write_json(preserved, {str(p.resolve()): sha256(p) for p in old_files})
    processed, clouds = make_clouds(data["segments"])
    cloud_path = root/"clouds.npz"
    if cloud_path.exists():
        with np.load(cloud_path) as saved:
            np.testing.assert_array_equal(saved["clouds"], clouds)
            np.testing.assert_array_equal(saved["processed"], processed)
    else:
        np.savez_compressed(cloud_path, processed=processed, clouds=clouds,
                            original_trial_number=data["original_trial_number"], labels=data["labels"],
                            latest_sample_index=np.arange(32, 500), tau_samples=16, dimension=3)
    write_json(root/"logs/environment.json", dict(python=platform.python_version(), numpy=np.__version__,
                                                scipy=scipy.__version__, jobs=args.jobs, cpu_count=os.cpu_count()))
    todo = []
    selected = trials.iloc[:args.limit_trials] if args.limit_trials else trials
    for trial in selected.to_dict("records"):
        row = int(trial["row_index"])
        for model in MODELS:
            checkpoint = root/"checkpoints"/f"trial_{row:03d}_{model}.json"
            if checkpoint.exists():
                validate_checkpoint(checkpoint, row, model, clouds[row])
            else:
                todo.append((row, model, trial))
    started = time.monotonic()
    print(f"Fitting {len(todo)} remaining trial/model combinations; tau=16, m=3", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(worker)(root, row, model, clouds[row], trial) for row, model, trial in todo)
        for count, (row, model, status, elapsed) in enumerate(results, 1):
            if count <= 8 or count % 20 == 0 or count == len(todo):
                print(f"{count}/{len(todo)}: trial row {row}, {model}, {status}, {elapsed:.2f}s", flush=True)
    table = collect_tables(root, trials)
    changed = [p for p, digest in provenance.items() if sha256(p) != digest]
    if changed:
        raise ValueError(f"Input/source changed during fitting: {changed}")
    write_json(root/"run_status.json", dict(complete=len(table) == 396, n_results=len(table),
                                          elapsed_seconds=time.monotonic()-started,
                                          returned=int((table.status == "returned").sum()),
                                          failed=int((table.status == "failed").sum())))
    print(f"Saved {len(table)}/396 results to {root}", flush=True)


if __name__ == "__main__":
    main()
