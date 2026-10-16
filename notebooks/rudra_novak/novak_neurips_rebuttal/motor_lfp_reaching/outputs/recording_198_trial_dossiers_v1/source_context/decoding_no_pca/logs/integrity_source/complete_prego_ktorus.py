#!/usr/bin/env python
"""Audit frozen full-coordinate fits and complete all seven permutation nulls."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prego_decoding import permute_within_folds, stable_seed
from prego_ktorus import METHODS, decode, geometry_features, phase_design, trial_cloud
from prego_statistics import prediction_metrics
from run_prego_ktorus import (DEFAULT_OUTPUT, PREVIOUS, CONVERTED_DIR, UNIT_DIR,
    artifact_hashes, environment_info, fold_binding, old_output_hashes,
    save_table, sha256, utc, verify_artifacts, verify_selection, write_json)
from run_prego_single_channel import load_recording


def check_saved_trial(segment, embedding, coefficients, frequencies, features):
    points, times = trial_cloud(segment, embedding)
    design = phase_design(times, frequencies)
    prediction = design @ coefficients
    residual = points-prediction
    result = dict(coefficients=coefficients, rmse_by_coordinate=np.sqrt(np.mean(residual**2, axis=0)),
                  normalized_error=np.sum(residual**2)/np.sum((points-points.mean(axis=0))**2))
    np.testing.assert_allclose(features, geometry_features(result, points), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(design.T @ residual, 0., atol=1e-6*np.linalg.norm(points))


def pending_nulls(frame, count):
    if frame.duplicated(["method", "permutation"]).any():
        raise ValueError("Duplicate permutation checkpoints")
    done = set(zip(frame.permutation, frame.method))
    return {p: missing for p in range(count)
            if (missing := [method for method in METHODS if (p, method) not in done])}


def complete_one(folder, root):
    with threadpool_limits(limits=1):
        complete = folder / "integrity_complete.json"
        if complete.exists():
            result = json.loads(complete.read_text())
            verify_artifacts(folder, result["artifacts"])
            return result
        meta = json.loads((folder / "decode_complete.json").read_text())
        if meta["status"] != "eligible":
            return dict(recording=folder.name, audited_trials=0)
        data, segments, good, source = load_recording(CONVERTED_DIR / f"{folder.name}.npz")
        if source["source_sha256"] != meta["source_sha256"]:
            raise ValueError("Source changed before full-coordinate audit")
        with np.load(folder / "trials.npz") as saved:
            trials = {k: saved[k] for k in saved.files}
        with np.load(PREVIOUS / "cache" / folder.name / "short" / "features.npz") as old:
            verify_selection(trials, old)
        np.testing.assert_array_equal(trials["original_trial_number"], data["original_trial_number"][trials["raw_trial_indices"]])
        position = {int(raw): j for j, raw in enumerate(good)}
        geometry, bindings, checked = [], {}, 0
        for fold in range(5):
            embedding = json.loads((folder / f"fold_{fold}.json").read_text())
            np.testing.assert_array_equal(embedding["training_trial_indices"], trials["raw_trial_indices"][trials["folds"] != fold])
            if embedding["status"] != "ok":
                geometry.append(None)
                continue
            bindings[str(fold)] = fold_binding(embedding, fold)
            with np.load(folder / f"fold_{fold}.npz") as saved:
                fitted = {k: saved[k] for k in saved.files}
            np.testing.assert_array_equal(fitted["raw_trial_indices"], trials["raw_trial_indices"])
            np.testing.assert_array_equal(fitted["feature_names"], embedding["feature_names"])
            diag = pd.read_csv(folder / f"fold_{fold}_diagnostics.csv")
            np.testing.assert_array_equal(diag.raw_trial_index, trials["raw_trial_indices"])
            np.testing.assert_array_equal(diag.held_out, trials["folds"] == fold)
            np.testing.assert_array_equal(diag.success, fitted["success"])
            for index in np.flatnonzero(fitted["success"]):
                segment = segments[position[int(trials["raw_trial_indices"][index])]]
                check_saved_trial(segment, embedding, fitted["coefficients"][index],
                                  fitted["frequencies"][index], fitted["features"][index])
                np.testing.assert_allclose(diag.normalized_error.iloc[index], fitted["features"][index, -1], atol=1e-10)
                checked += 1
            geometry.append(fitted["features"])
        if meta["geometry_evaluable"]:
            null_path = folder / "null_scores.csv"
            frame = pd.read_csv(null_path)
            rows = frame.to_dict("records")
            for permutation, methods in pending_nulls(frame, 200).items():
                seed = stable_seed(f"{folder.name}-null-{permutation}", 42)
                labels = permute_within_folds(trials["labels"], trials["folds"], seed)
                predictions = decode(trials["power"], trials["frequency"], trials["bands"], geometry,
                                     labels, trials["folds"], methods)
                for method, values in predictions.items():
                    score, _, _ = prediction_metrics(labels, values)
                    rows.append(dict(recording=folder.name, monkey=meta["monkey"], day=meta["day"],
                                     method=method, permutation=permutation, macro_f1=score))
                if (permutation+1) % 20 == 0:
                    save_table(null_path, pd.DataFrame(rows))
            save_table(null_path, pd.DataFrame(rows))
            if pending_nulls(pd.DataFrame(rows), 200):
                raise ValueError("Permutation coverage incomplete")
        result = dict(recording=folder.name, audited_trials=checked, fold_bindings=bindings,
                      artifacts=artifact_hashes(folder), utc=utc())
        write_json(complete, result)
        print(f"audited: {folder.name}: {checked} full-coordinate trial fits", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    root = args.output.resolve()
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["state"] not in ("decoding_complete", "verified", "integrity_running"):
            raise ValueError("Finish the primary decoding run first")
        if manifest["environment"] != environment_info():
            raise ValueError("Environment changed since the primary run")
        for name, digest in manifest["provenance"]["source"].items():
            if sha256(root / "logs" / "source" / name) != digest:
                raise ValueError("Initial source snapshot changed")
            if name != "run_prego_ktorus.py" and sha256(UNIT_DIR / name) != digest:
                raise ValueError(f"Scientific source changed: {name}")
        snapshot = root / "logs" / "integrity_source"
        snapshot.mkdir(exist_ok=True)
        for name in ("complete_prego_ktorus.py", "run_prego_ktorus.py"):
            target = snapshot / name
            if target.exists() and sha256(target) != sha256(UNIT_DIR / name):
                raise ValueError("Integrity-stage source changed during resume")
            if not target.exists():
                shutil.copy2(UNIT_DIR / name, target)
        before = {p.parent.name: sha256(p) for p in (root / "cache").glob("*/predictions.csv")}
        manifest.update(state="integrity_running", completion_stage=dict(utc=utc(), argv=sys.argv,
            source={p.name: sha256(p) for p in snapshot.glob("*.py")},
            purpose="Full-coordinate checkpoint audit and seven-method permutation coverage; no geometry refit",
            scientific_model_unchanged=True))
        write_json(root / "manifest.json", manifest)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.jobs)(delayed(complete_one)(folder, root)
                       for folder in sorted((root / "cache").glob("*")))
        after = {p.parent.name: sha256(p) for p in (root / "cache").glob("*/predictions.csv")}
        assert before == after
        assert old_output_hashes() == manifest["provenance"]["predecessor"]
        manifest.update(state="decoding_complete", integrity_audited_trials=sum(r["audited_trials"] for r in results),
                        all_seven_nulls_complete=True, predictions_unchanged_by_completion=True)
        write_json(root / "manifest.json", manifest)
        print(f"Integrity audit and all seven null references complete: {manifest['integrity_audited_trials']} trial fits", flush=True)


if __name__ == "__main__":
    main()
