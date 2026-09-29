"""Independent persistent homology of frozen, cue-offset-aligned stage clouds."""
from __future__ import annotations

import argparse
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
import traceback

import numpy as np
import pandas as pd

UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
OUTPUT = UNIT / "outputs/macaque_task_six_stages_cue_offset_correct/BETTI_STAGE_v1"
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")
ANCHORS = ("TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO")
CODES = (202, 203, 204, 206, 207, 207)
PARAMETERS = dict(maxdim=2, coeff=2, thresh=float("inf"), n_perm=None,
                  metric="euclidean", distance_matrix=False, do_cocycles=False)
PILOT_TRIALS = (6, 9, 10)
TIMEOUT = 600


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(array):
    x = np.ascontiguousarray(array)
    return hashlib.sha256(str(x.shape).encode() + x.dtype.str.encode() + x.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    return dict(python=platform.python_version(), packages={
        p: importlib.metadata.version(p) for p in ("ripser", "numpy", "scipy", "pandas")})


def configuration():
    return dict(source=str(SOURCE), recording="monkeyT_session-y070316009-12_lfp-7",
        n_trials=198, n_windows=1188, n_points=294, fs_hz=1000, m=3, tau_ms=3,
        stages=list(STAGES), anchors=list(ANCHORS), event_codes=list(CODES),
        window_samples=300, post_SC_definition="spatial instruction ends at distractor onset (206)",
        input_transform="none; reuse exact frozen observed clouds",
        ripser_version="0.6.15", parameters=dict(PARAMETERS, thresh="infinity"),
        pilot_original_trials=list(PILOT_TRIALS), pilot_all_six_stages=True,
        sequential=True, timeout_seconds=TIMEOUT, grid_points=512,
        betti_interval="birth <= distance < death",
        normalized_distance="birth and death divided by centered RMS radius; no Ripser rerun",
        empty_dimension_grid="use shared finite H0 death maximum; 1.0 only if all clouds coincide",
        matched_sensitivity="trials with six successful PH calculations, only if failures occur",
        assumed_topology=None, fitter_calls=0, decoder_calls=0, significance_tests=False)


def load_inputs():
    config = json.loads((SOURCE / "config.json").read_text())
    assert (config["m"], config["tau_ms"], config["fs_hz"], config["samples_per_window"]) == (3, 3, 1000, 300)
    assert config["stage_anchors"] == list(ANCHORS)
    assert json.loads((SOURCE / "validation.json").read_text())["passed"]
    table = pd.read_csv(SOURCE / "inputs/window_metadata.csv", keep_default_na=False)
    events = pd.read_csv(SOURCE / "inputs/event_samples.csv").set_index("trial_index")
    trials = pd.read_csv(SOURCE / "inputs/selected_trials.csv")
    with np.load(SOURCE / "inputs/windows.npz", allow_pickle=False) as saved:
        clouds, processed = saved["clouds"], saved["processed"]
    assert clouds.shape == (1188, 294, 3) and processed.shape == (1188, 300)
    np.testing.assert_array_equal(clouds, np.stack((processed[:, 6:], processed[:, 3:-3], processed[:, :-6]), axis=2))
    np.testing.assert_array_equal(table.window_index, np.arange(1188))
    assert table.original_trial_number.nunique() == len(trials) == len(events) == 198
    assert set(table.original_trial_number) == set(trials.original_trial_number)
    assert (table.delay == "short").all()
    assert table.groupby("stage_name").size().eq(198).all()
    assert table.groupby(["stage_name", "direction"]).size().eq(33).all()
    assert table.groupby("trial_index").heldout_fold_zero_based.nunique().eq(1).all()
    for trial_index, rows in table.groupby("trial_index"):
        assert sorted(rows.stage) == list(range(6))
        event = events.loc[trial_index]
        assert (rows.original_trial_number == event.original_trial_number).all()
        ordered = rows.sort_values("stage")
        assert np.all(ordered.end_sample_exclusive.to_numpy()[:-1] <= ordered.start_sample.to_numpy()[1:])
        for row in rows.itertuples():
            assert row.stage_name == STAGES[row.stage] and row.anchor_name == ANCHORS[row.stage]
            assert row.anchor_code == CODES[row.stage]
            timestamp = int(event[ANCHORS[row.stage] + "_sample"])
            start = timestamp - 300 if row.stage % 2 == 0 else timestamp
            assert (row.start_sample, row.end_sample_exclusive, row.event_sample) == (start, start + 300, timestamp)
    return clouds, table


def protected_paths():
    paths = [p for p in (UNIT / "outputs").rglob("*") if p.is_file()
             and not p.is_relative_to(OUTPUT) and "__pycache__" not in p.parts]
    paths += [p for p in UNIT.glob("*.py") if p.name not in (
        "run_stage_persistent_homology.py", "plot_stage_persistent_homology.py")]
    paths += list((UNIT.parents[3] / "NeuralFieldManifold/fits").glob("*.py"))
    return sorted(set(paths))


def verify_manifest(path):
    entries = json.loads(Path(path).read_text())
    for name, expected in entries.items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f"Protected/source file changed: {name}")
    return len(entries)


def initialize(clouds, table, resume):
    if environment()["packages"]["ripser"] != "0.6.15":
        raise ValueError("Requires Ripser.py 0.6.15")
    if OUTPUT.exists():
        if not resume:
            raise FileExistsError("Use --resume for this frozen experiment")
        assert json.loads((OUTPUT / "config.json").read_text()) == configuration()
        assert json.loads((OUTPUT / "environment.json").read_text()) == environment()
        verify_manifest(OUTPUT / "provenance/input_source_hashes.json")
        return
    verify_manifest(SOURCE / "frozen_input_hashes.json")
    for name in ("provenance", "source", "inputs", "diagrams", "checkpoints", "tables", "logs", "validation", "captions"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "config.json", configuration())
    write_json(OUTPUT / "environment.json", environment())
    write_json(OUTPUT / "provenance/protected_hashes.json", {str(p): digest(p) for p in protected_paths()})
    inputs = [SOURCE / "inputs/windows.npz", SOURCE / "inputs/window_metadata.csv",
              SOURCE / "inputs/event_samples.csv", SOURCE / "inputs/selected_trials.csv",
              SOURCE / "config.json", SOURCE / "tables/fit_parameters_diagnostics.csv", Path(__file__).resolve()]
    write_json(OUTPUT / "provenance/input_source_hashes.json", {str(p): digest(p) for p in inputs})
    shutil.copy2(__file__, OUTPUT / "source" / Path(__file__).name)
    for name in ("window_metadata.csv", "event_samples.csv", "selected_trials.csv"):
        shutil.copy2(SOURCE / "inputs" / name, OUTPUT / "inputs" / name)
    write_json(OUTPUT / "provenance/cloud_hashes.json", [array_digest(c) for c in clouds])
    assert len(pilot_rows(table)) == 18


def centered_rms(cloud):
    cloud = np.asarray(cloud, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((cloud - cloud.mean(axis=0)) ** 2, axis=1))))


def compute_diagrams(cloud):
    import ripser
    if ripser.__version__ != "0.6.15":
        raise ValueError("Ripser version mismatch")
    cloud = np.asarray(cloud)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError("Expected finite observed coordinates, with no transformation")
    return ripser.ripser(cloud, **PARAMETERS)["dgms"]


def validate_diagrams(diagrams, n_points):
    assert len(diagrams) == 3
    for dimension, diagram in enumerate(diagrams):
        assert diagram.ndim == 2 and diagram.shape[1] == 2
        assert np.isfinite(diagram[:, 0]).all() and not np.isnan(diagram).any()
        assert (diagram[:, 0] >= 0).all() and (diagram[:, 1] > diagram[:, 0]).all()
        if dimension:
            assert np.isfinite(diagram).all(), "Full finite-cloud filtration must kill H1/H2"
    assert np.isinf(diagrams[0][:, 1]).sum() == 1
    assert 1 <= len(diagrams[0]) <= n_points


def betti_curve(diagram, grid):
    diagram = np.asarray(diagram)
    return ((diagram[:, :1] <= grid) & (grid < diagram[:, 1:2])).sum(axis=0)


def longest_lifetime(diagram):
    if len(diagram) == 0:
        return 0.0
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    return float(np.max(finite[:, 1] - finite[:, 0])) if len(finite) else float("nan")


def checkpoint_paths(index):
    stem = f"window_{index:04d}"
    return OUTPUT / "checkpoints" / f"{stem}.json", OUTPUT / "diagrams" / f"{stem}.npz"


def read_checkpoint(index, cloud, row):
    meta_path, data_path = checkpoint_paths(index)
    meta = json.loads(meta_path.read_text())
    for name in ("window_index", "original_trial_number", "trial_index", "stage_name"):
        assert meta[name] == row[name], name
    assert meta["cloud_sha256"] == array_digest(cloud)
    assert meta["config_sha256"] == digest(OUTPUT / "config.json")
    assert meta["status"] in ("success", "failed", "timeout")
    diagrams = None
    if meta["status"] == "success":
        assert digest(data_path) == meta["diagram_sha256"]
        with np.load(data_path, allow_pickle=False) as saved:
            diagrams = [saved[f"H{h}"] for h in range(3)]
        validate_diagrams(diagrams, len(cloud))
    return meta, diagrams


def worker(index):
    # Workers read frozen arrays directly; the parent audits all indexing once.
    with np.load(SOURCE / "inputs/windows.npz", allow_pickle=False) as data:
        cloud = data["clouds"][index]
    row = pd.read_csv(SOURCE / "inputs/window_metadata.csv").iloc[index]
    meta_path, data_path = checkpoint_paths(index)
    started = time.monotonic()
    meta = dict(window_index=index, original_trial_number=int(row.original_trial_number),
        trial_index=int(row.trial_index), stage_name=row.stage_name, n_points=len(cloud),
        cloud_sha256=array_digest(cloud), config_sha256=digest(OUTPUT / "config.json"))
    try:
        diagrams = compute_diagrams(cloud)
        validate_diagrams(diagrams, len(cloud))
        temporary = data_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **{f"H{h}": d for h, d in enumerate(diagrams)})
        temporary.replace(data_path)
        meta.update(status="success", reason="", diagram_sha256=digest(data_path))
    except Exception:
        meta.update(status="failed", reason=traceback.format_exc())
    meta["elapsed_seconds"] = time.monotonic() - started
    write_json(meta_path, meta)


def run_window(index, clouds, table):
    row, cloud = table.iloc[index], clouds[index]
    meta_path, _ = checkpoint_paths(index)
    if meta_path.exists():
        return read_checkpoint(index, cloud, row)[0], True
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    status, reason = "failed", ""
    with (OUTPUT / "logs" / f"window_{index:04d}.log").open("w") as log:
        try:
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                "--worker-index", str(index)], stdout=log, stderr=log, env=env, timeout=TIMEOUT, check=False)
            reason = f"Worker exit code {result.returncode}; see window log"
        except subprocess.TimeoutExpired:
            status, reason = "timeout", "Exceeded 600 seconds; no approximation"
    if not meta_path.exists():
        write_json(meta_path, dict(window_index=index, original_trial_number=int(row.original_trial_number),
            trial_index=int(row.trial_index), stage_name=row.stage_name, n_points=len(cloud),
            cloud_sha256=array_digest(cloud), config_sha256=digest(OUTPUT / "config.json"),
            status=status, reason=reason, elapsed_seconds=time.monotonic()-started))
    meta, _ = read_checkpoint(index, cloud, row)
    meta["wall_seconds"] = time.monotonic() - started
    write_json(meta_path, meta)
    return meta, False


def pilot_rows(table):
    return table.loc[table.original_trial_number.isin(PILOT_TRIALS)].sort_values(
        ["original_trial_number", "stage"]).window_index.astype(int).tolist()


def summarize_data(clouds, table):
    fits = pd.read_csv(SOURCE / "tables/fit_parameters_diagnostics.csv")
    np.testing.assert_array_equal(fits.window_index, table.window_index)
    records, interval_rows, diagrams = [], [], {}
    for index, row in table.iterrows():
        meta, d = read_checkpoint(index, clouds[index], row)
        rms = centered_rms(clouds[index])
        item = dict(window_index=index, trial_index=int(row.trial_index),
            original_trial_number=int(row.original_trial_number), stage=int(row.stage),
            stage_name=row.stage_name, direction=int(row.direction), status=meta["status"],
            failure_reason=meta["reason"], elapsed_seconds=meta["elapsed_seconds"],
            geometry_usable=bool(fits.iloc[index].usable), centered_rms=rms,
            normalization_valid=bool(np.isfinite(rms) and rms > 0),
            longest_H1_lifetime=np.nan, normalized_H1_lifetime=np.nan)
        if d is not None:
            diagrams[index] = d
            item["longest_H1_lifetime"] = longest_lifetime(d[1])
            if item["normalization_valid"]:
                item["normalized_H1_lifetime"] = item["longest_H1_lifetime"] / rms
            for h, bars in enumerate(d):
                item[f"H{h}_intervals"] = len(bars)
                item[f"H{h}_infinite"] = int(np.isinf(bars[:, 1]).sum())
                for j, (birth, death) in enumerate(bars):
                    interval_rows.append(dict(window_index=index, original_trial_number=int(row.original_trial_number),
                        stage_name=row.stage_name, homology_dimension=h, interval=j,
                        birth=birth, death=death, lifetime=death-birth))
        records.append(item)
    if not diagrams:
        raise ValueError("No successful PH calculations")
    measures = pd.DataFrame(records)
    complete_trials = measures.groupby("trial_index").status.apply(lambda s: (s == "success").all())
    matched = measures.trial_index.isin(complete_trials.index[complete_trials]).to_numpy()
    cohorts = {"available": np.ones(len(table), dtype=bool)}
    if not complete_trials.all():
        cohorts["matched_six_stages"] = matched
    arrays, grid_notes, summaries, wide = {}, {}, [], []
    for convention in ("raw", "rms_normalized"):
        converted = {i: d if convention == "raw" else [v/measures.iloc[i].centered_rms for v in d]
                     for i, d in diagrams.items() if convention == "raw" or measures.iloc[i].normalization_valid}
        for h in range(3):
            finite = np.concatenate([d[h][:, 1][np.isfinite(d[h][:, 1])] for d in converted.values()]) if converted else np.array([])
            fallback = not len(finite)
            if fallback:
                finite = np.concatenate([d[0][:, 1][np.isfinite(d[0][:, 1])] for d in converted.values()]) if converted else np.array([])
            extent = float(finite.max()) if len(finite) else 1.0
            grid = np.linspace(0, extent, 512)
            values = np.full((len(table), 512), np.nan)
            for i, d in converted.items():
                values[i] = betti_curve(d[h], grid)
            key = f"{convention}_H{h}"
            arrays[key] = values
            arrays[key + "_grid"] = grid
            grid_notes[key] = dict(max_distance=extent, fallback_to_H0=fallback,
                                   degenerate_unit_grid=not len(finite), points=512)
            frame = pd.DataFrame(values, columns=[f"grid_{j:03d}" for j in range(512)])
            frame.insert(0, "window_index", np.arange(len(table)))
            frame.insert(0, "homology_dimension", h)
            frame.insert(0, "distance_convention", convention)
            wide.append(frame)
            for cohort, include in cohorts.items():
                for stage in range(6):
                    selected = values[include & (table.stage.to_numpy() == stage) & np.isfinite(values).all(axis=1)]
                    if len(selected):
                        q1, med, q3 = np.quantile(selected, [.25, .5, .75], axis=0)
                        avg = selected.mean(axis=0)
                    else:
                        q1 = med = q3 = avg = np.full(512, np.nan)
                    summaries.extend(dict(cohort=cohort, distance_convention=convention,
                        homology_dimension=h, stage=stage, stage_name=STAGES[stage], grid_index=j,
                        distance=grid[j], n_windows=len(selected), mean=avg[j], q25=q1[j], median=med[j], q75=q3[j])
                        for j in range(512))
    distributions = []
    for cohort, include in cohorts.items():
        for stage in range(6):
            for column in ("longest_H1_lifetime", "normalized_H1_lifetime"):
                vals = measures.loc[include & (measures.stage == stage), column].dropna().to_numpy()
                q1, median, q3 = np.quantile(vals, [.25, .5, .75]) if len(vals) else (np.nan,)*3
                distributions.append(dict(cohort=cohort, stage=stage, stage_name=STAGES[stage],
                    measure=column, n_windows=len(vals), mean=vals.mean() if len(vals) else np.nan,
                    q25=q1, median=median, q75=q3))
    return dict(window_measurements=measures, persistence_intervals=pd.DataFrame(interval_rows),
        betti_summary=pd.DataFrame(summaries), lifetime_summary=pd.DataFrame(distributions),
        betti_curves_wide=pd.concat(wide, ignore_index=True)), arrays, grid_notes


def summarize(clouds, table):
    tables, arrays, grids = summarize_data(clouds, table)
    for name, frame in tables.items():
        suffix = ".csv.gz" if name in ("persistence_intervals", "betti_curves_wide") else ".csv"
        frame.to_csv(OUTPUT / "tables" / (name + suffix), index=False)
    np.savez_compressed(OUTPUT / "tables/betti_curves.npz", **arrays)
    write_json(OUTPUT / "tables/grid_definitions.json", grids)
    write_json(OUTPUT / "provenance/table_hashes.json", {str(p): digest(p) for p in (OUTPUT / "tables").iterdir() if p.is_file()})
    return tables["window_measurements"]


def verify_results(clouds, table):
    verify_manifest(OUTPUT / "provenance/input_source_hashes.json")
    verify_manifest(OUTPUT / "provenance/table_hashes.json")
    expected, arrays, grids = summarize_data(clouds, table)
    for name, frame in expected.items():
        suffix = ".csv.gz" if name in ("persistence_intervals", "betti_curves_wide") else ".csv"
        saved = pd.read_csv(OUTPUT / "tables" / (name + suffix), float_precision="round_trip")
        for col in frame.select_dtypes(include="object"):
            saved[col] = saved[col].fillna("")
            frame[col] = frame[col].fillna("")
        pd.testing.assert_frame_equal(saved, frame, check_dtype=False, atol=1e-12, rtol=1e-12)
    with np.load(OUTPUT / "tables/betti_curves.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    assert json.loads((OUTPUT / "tables/grid_definitions.json").read_text()) == grids
    assert json.loads((OUTPUT / "provenance/cloud_hashes.json").read_text()) == [array_digest(c) for c in clouds]
    protected = verify_manifest(OUTPUT / "provenance/protected_hashes.json")
    # Resume must reuse an existing checkpoint without recomputing or rewriting it.
    index = pilot_rows(table)[0]
    before = {str(p): digest(p) for p in checkpoint_paths(index)}
    _, reused = run_window(index, clouds, table)
    assert reused and before == {str(p): digest(p) for p in checkpoint_paths(index)}
    m = expected["window_measurements"]
    report = dict(passed=True, windows=len(m), trials=198,
        successes=int(m.status.eq("success").sum()), failures=int(m.status.ne("success").sum()),
        by_stage=m.groupby("stage_name").status.value_counts().unstack(fill_value=0).to_dict("index"),
        geometry_unusable_included=int((~m.geometry_usable & m.status.eq("success")).sum()),
        undefined_normalizations=int((~m.normalization_valid).sum()),
        saved_tables_reproduced=True, checkpoint_resume_verified=True, protected_files_unchanged=protected,
        inputs_match_frozen_delayed_samples=True, cue_offset_anchors_verified=True,
        all_points_used=True, geometric_fit_calls=0, decoder_calls=0)
    write_json(OUTPUT / "validation/numerical_checks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "all", "summarize", "verify"), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_index is not None:
        if not 0 <= args.worker_index < 1188:
            parser.error("Invalid window index")
        worker(args.worker_index)
        return
    clouds, table = load_inputs()
    initialize(clouds, table, args.resume)
    if args.phase == "verify":
        print(json.dumps(verify_results(clouds, table), indent=2), flush=True)
        return
    if args.phase == "summarize":
        summarize(clouds, table)
        print(json.dumps(verify_results(clouds, table), indent=2), flush=True)
        return
    pilot = pilot_rows(table)
    results, reused = [], 0
    for index in pilot:
        meta, old = run_window(index, clouds, table)
        results.append(meta)
        reused += int(old)
        print(f"Pilot trial {meta['original_trial_number']} {meta['stage_name']}: "
              f"{meta['status']}, {meta['elapsed_seconds']:.2f}s" + (" (saved)" if old else ""), flush=True)
        if meta["status"] != "success":
            write_json(OUTPUT / "run_status.json", dict(status="pilot_stopped", results=results))
            verify_manifest(OUTPUT / "provenance/protected_hashes.json")
            raise SystemExit("Pilot failed; stopped without approximation")
    timings = [r.get("wall_seconds", r["elapsed_seconds"]) for r in results]
    write_json(OUTPUT / "pilot.json", dict(windows=pilot, original_trials=list(PILOT_TRIALS),
        wall_seconds=timings, estimated_full_seconds=float(np.mean(timings)*1188), all_success=True))
    print(f"Pilot passed; projected sequential full run {np.mean(timings)*1188/60:.1f} minutes", flush=True)
    if args.phase == "pilot":
        verify_manifest(OUTPUT / "provenance/protected_hashes.json")
        return
    for count, index in enumerate([i for i in range(1188) if i not in pilot], 19):
        meta, old = run_window(index, clouds, table)
        reused += int(old)
        if count % 20 == 0 or count == 1188 or meta["status"] != "success":
            print(f"Accounted {count}/1188; window {index}: {meta['status']} "
                  f"({meta['elapsed_seconds']:.2f}s)", flush=True)
            write_json(OUTPUT / "run_status.json", dict(status="running", accounted=count, reused=reused, last_window=index))
    summarize(clouds, table)
    validated = verify_results(clouds, table)
    write_json(OUTPUT / "run_status.json", dict(status="complete", reused_checkpoints=reused, **validated))
    print(json.dumps(validated, indent=2), flush=True)


if __name__ == "__main__":
    main()
