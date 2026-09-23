"""Extract, fit, decode, and verify the isolated six-stage 15D experiment."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from prego_geometric_fits import array_hash, fit_one, recompute_distances, sha256, write_json
from stage_geometry_decoding import (
    UNIT, RECORDING, PREVIOUS, OUTPUT, STAGES, LABELS, COLUMNS, load_raw_windows,
    preprocess_window, embed_window, geometry_features, decode, scores, bootstrap,
    permutation, shuffle_stages, validate_groups,
)


def configuration():
    return dict(recording=RECORDING, n_trials=198, n_windows=1188, fs_hz=1000,
                stage_names=list(STAGES), event_codes=dict(TC=202, SC=204, GO=207),
                time_indexing="MATLAB one-based samples converted to zero-based; half-open windows",
                window_offsets_ms=[[-300, 0], [0, 300]], samples_per_window=300,
                m=3, tau_samples=3, tau_ms=3, points_per_cloud=294,
                preprocessing=dict(detrend="linear", normalization="per-window median/MAD; std fallback",
                                   filter="fourth-order Butterworth SOS, sosfiltfilt default odd padding",
                                   band_hz=[2, 55], dtype="float32 after filtering"),
                fitter=dict(name="one_torus_fit", K_assumed=1, lam=.1, hole_ratio=.5,
                            n_regularization_harmonics=3, loss="huber", max_nfev=6000,
                            initialization="native full-dimensional SVD; no coordinate reduction"),
                feature_columns=COLUMNS, n_features=15,
                half_width_definition="(R1_out-R1_in+R2_out-R2_in)/4",
                folds="saved trial assignments; all six stages grouped by trial",
                classifier=dict(imputation="training median", scale="training StandardScaler",
                                solver="lsqr", shrinkage="auto"),
                permutations=1000, permutation_seed=52000, permutation_scheme="stage labels within trial",
                bootstraps=2000, bootstrap_seed=42000,
                bootstrap_scheme="whole trials stratified by reach direction, all six windows retained",
                example_original_trials=[6, 9, 10], pca=False, betti=False, pinn=False,
                parameter_selection=False, spectral_features=False)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__)


def source_paths():
    native = Path(importlib.import_module("NeuralFieldManifold.fits.one_torus").__file__)
    return [Path(__file__).resolve(), native]+[UNIT/name for name in (
        "stage_geometry_decoding.py", "prego_geometric_fits.py", "prego_fixed_geometry_decoding.py",
        "prego_decoding.py", "motor_lfp_utils.py", "convert_motor_lfp.py",
        "select_trace_embedding_parameters.py")]


def check_manifest(path):
    hashes = json.loads(Path(path).read_text())
    changed = [p for p, h in hashes.items() if not Path(p).is_file() or sha256(p) != h]
    if changed:
        raise ValueError(f"Protected/source files changed: {changed[:10]}")
    return len(hashes)


def initialize(root, resume):
    if root.exists():
        if not resume:
            raise FileExistsError("Experiment exists; use --resume")
        if json.loads((root/"config.json").read_text()) != configuration():
            raise ValueError("Frozen configuration mismatch")
        if json.loads((root/"environment.json").read_text()) != environment():
            raise ValueError("Numerical environment mismatch")
        check_manifest(root/"source_input_hashes.json")
        return
    root.mkdir(parents=True)
    for name in ("inputs", "source", "checkpoints", "models", "null_checkpoints", "tables"):
        (root/name).mkdir()
    write_json(root/"config.json", configuration())
    write_json(root/"environment.json", environment())
    print("Hashing protected previous outputs and source inputs", flush=True)
    protected = {str(p.resolve()): sha256(p) for p in sorted((UNIT/"outputs").rglob("*"))
                 if p.is_file() and not p.is_relative_to(root)}
    write_json(root/"protected_prior_outputs.json", protected)
    inputs = [PREVIOUS/"inputs/selected_trials.csv", PREVIOUS/"inputs/alignment.json",
              UNIT/"lfp_converted_data"/f"{RECORDING}.npz", UNIT/"raw_data/MonkeyT.mat"]
    write_json(root/"source_input_hashes.json", {str(p.resolve()): sha256(p) for p in source_paths()+inputs})
    for p in source_paths():
        shutil.copy2(p, root/"source"/p.name)
    print(f"Protected {len(protected)} existing output files", flush=True)


def prepare(root):
    destination = root/"inputs/windows.npz"
    if destination.exists():
        check_manifest(root/"frozen_input_hashes.json")
        print("Reusing verified frozen windows", flush=True)
        return
    trials, metadata, events, raw = load_raw_windows()
    processed = np.full((1188, 300), np.nan, dtype=np.float32)
    clouds = np.full((1188, 294, 3), np.nan, dtype=np.float32)
    errors = []
    for i, x in enumerate(raw):
        try:
            processed[i] = preprocess_window(x)
            clouds[i] = embed_window(processed[i])
            errors.append("")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    metadata["preprocessing_error"] = errors
    metadata["cloud_sha256"] = [array_hash(c) for c in clouds]
    metadata.to_csv(root/"inputs/window_metadata.csv", index=False)
    events.to_csv(root/"inputs/event_samples.csv", index=False)
    trials.to_csv(root/"inputs/selected_trials.csv", index=False)
    np.savez_compressed(destination, raw=raw, processed=processed, clouds=clouds)
    files = sorted((root/"inputs").iterdir())
    write_json(root/"frozen_input_hashes.json", {str(p.resolve()): sha256(p) for p in files})
    print(f"Frozen {len(raw)} windows; preprocessing failures: {sum(bool(e) for e in errors)}", flush=True)


def inputs(root):
    metadata = pd.read_csv(root/"inputs/window_metadata.csv", keep_default_na=False)
    with np.load(root/"inputs/windows.npz") as z:
        data = {k: z[k] for k in z.files}
    return metadata, data


def checkpoint(root, index):
    return root/"checkpoints"/f"window_{index:04d}.json"


def read_fit(path, row, cloud):
    result = json.loads(path.read_text())
    assert result["window_index"] == row["window_index"]
    assert result["original_trial_number"] == row["original_trial_number"]
    assert result["stage_name"] == row["stage_name"]
    assert result["cloud_sha256"] == array_hash(cloud)
    return result


def fit_window(root, row, cloud):
    path = checkpoint(root, row["window_index"])
    if path.exists():
        result = read_fit(path, row, cloud)
        return row["window_index"], result["elapsed_seconds"], "reused"
    with threadpool_limits(limits=1):
        if row["preprocessing_error"]:
            result = dict(status="failed", reason="preprocessing: "+row["preprocessing_error"],
                          elapsed_seconds=0., cloud_sha256=array_hash(cloud), model="one_torus")
        else:
            result = fit_one(cloud, "one_torus")
    result.update({k: row[k] for k in ("window_index", "original_trial_number", "stage_name")})
    write_json(path, result)
    return row["window_index"], result["elapsed_seconds"], result["status"]


def fit_all(root, jobs, pilot=False):
    metadata, data = inputs(root)
    selected = metadata[metadata.original_trial_number == 6] if pilot else metadata
    started = time.monotonic()
    durations = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1 if pilot else jobs, return_as="generator_unordered")(
            delayed(fit_window)(root, row, data["clouds"][row["window_index"]])
            for row in selected.to_dict("records"))
        for n, (i, seconds, status) in enumerate(results, 1):
            durations.append(seconds)
            if pilot or n % 25 == 0 or n == len(selected):
                print(f"Fits {n}/{len(selected)}; window {i}, {status}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    if pilot:
        write_json(root/"pilot.json", dict(n=6, elapsed_seconds=time.monotonic()-started,
                   fit_seconds=durations, projected_worker_seconds=sum(durations)*198))


def collect_features(root):
    metadata, data = inputs(root)
    vectors, reasons, parameters = [], [], []
    for row in metadata.to_dict("records"):
        result = read_fit(checkpoint(root, row["window_index"]), row, data["clouds"][row["window_index"]])
        vector, reason = geometry_features(result)
        vectors.append(vector)
        reasons.append(reason)
        parameters.append(dict(window_index=row["window_index"], stage_name=row["stage_name"],
                               original_trial_number=row["original_trial_number"], fit_status=result["status"],
                               usable=not bool(reason), unusable_reason=reason,
                               elapsed_seconds=result["elapsed_seconds"], **result.get("summary", {})))
    x = np.asarray(vectors)
    features = pd.concat([metadata, pd.DataFrame(x, columns=COLUMNS)], axis=1)
    features["usable"] = np.isfinite(x).all(axis=1)
    features["unusable_reason"] = reasons
    features.to_csv(root/"tables/features.csv", index=False)
    np.savez_compressed(root/"features.npz", x=x, labels=metadata.stage.to_numpy(),
                        folds=metadata.heldout_fold_zero_based.to_numpy(), trial_ids=metadata.trial_index.to_numpy())
    params = pd.DataFrame(parameters)
    params.to_csv(root/"tables/fit_parameters_diagnostics.csv", index=False)
    accounting = features.groupby("stage_name", sort=False).agg(n_windows=("usable", "size"), usable=("usable", "sum"))
    accounting["unusable"] = accounting.n_windows-accounting.usable
    accounting.to_csv(root/"tables/fit_accounting.csv")
    distributions = []
    for stage, group in features.groupby("stage_name", sort=False):
        for feature in COLUMNS:
            values = group.loc[group.usable, feature]
            distributions.append(dict(stage_name=stage, feature=feature, n=len(values),
                                      mean=values.mean(), sd=values.std(ddof=1), median=values.median(),
                                      q25=values.quantile(.25), q75=values.quantile(.75)))
    pd.DataFrame(distributions).to_csv(root/"tables/feature_summaries.csv", index=False)
    return metadata, x


def null_worker(root, index, x, y, folds, ids):
    path = root/"null_checkpoints"/f"permutation_{index:04d}.npz"
    if path.exists():
        with np.load(path) as saved:
            result = {k: saved[k] for k in saved.files}
        np.testing.assert_array_equal(result["labels"], shuffle_stages(y, ids, index))
        return result
    with threadpool_limits(limits=1):
        result = permutation(index, x, y, folds, ids)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    return result


def evaluate(root, jobs):
    metadata, x = collect_features(root)
    y, folds, ids = [metadata[k].to_numpy() for k in ("stage", "heldout_fold_zero_based", "trial_index")]
    with threadpool_limits(limits=1):
        pred, models = decode(x, y, folds, ids, retain_models=True)
    for fold, model in models.items():
        joblib.dump(model, root/"models"/f"geometry_fold{fold}.joblib")
    np.savez_compressed(root/"heldout_predictions.npz", labels=y, predictions=pred, folds=folds, trial_ids=ids)
    pd.DataFrame(dict(window_index=metadata.window_index, original_trial_number=metadata.original_trial_number,
                      trial_index=ids, stage=y, stage_name=metadata.stage_name, heldout_fold=folds,
                      predicted_stage=pred, predicted_stage_name=np.asarray(STAGES)[pred])).to_csv(
                          root/"tables/heldout_predictions.csv", index=False)
    boot = bootstrap(y, pred, ids, metadata.direction.to_numpy())
    np.savez_compressed(root/"bootstrap.npz", **boot)
    nulls = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        iterator = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(null_worker)(root, i, x, y, folds, ids) for i in range(1000))
        for n, result in enumerate(iterator, 1):
            nulls.append(result)
            if n % 100 == 0:
                print(f"Permutation decoders {n}/1000", flush=True)
    nulls.sort(key=lambda r: int(r["index"]))
    null = {k: np.stack([r[k] for r in nulls]) for k in nulls[0]}
    np.savez_compressed(root/"permutations.npz", **null)
    s = scores(y, pred)
    result = dict(n_trials=198, n_windows=1188, features=15, usable_fits=int(np.isfinite(x).all(axis=1).sum()))
    for key in ("macro_f1", "accuracy"):
        lo, hi = np.quantile(boot[key], [.025, .975])
        result.update({key: s[key], key+"_ci_low": lo, key+"_ci_high": hi})
    result["permutation_p"] = (1+int((null["macro_f1"] >= s["macro_f1"]).sum()))/1001
    result["null_macro_f1_median"] = float(np.median(null["macro_f1"]))
    pd.DataFrame([result]).to_csv(root/"tables/summary.csv", index=False)
    lo, hi = np.quantile(boot["per_stage_f1"], [.025, .975], axis=0)
    pd.DataFrame(dict(stage_name=STAGES, f1=s["per_stage_f1"], ci_low=lo, ci_high=hi)).to_csv(
        root/"tables/per_stage_f1.csv", index=False)
    for key in ("confusion_counts", "confusion_fraction"):
        pd.DataFrame(s[key], index=STAGES, columns=STAGES).rename_axis("true_stage").to_csv(root/"tables"/f"{key}.csv")
    pd.DataFrame({k: null[k] for k in ("index", "macro_f1", "accuracy")}).to_csv(root/"tables/null_scores.csv", index=False)
    pd.DataFrame({k: boot[k] for k in ("macro_f1", "accuracy")}).to_csv(root/"tables/bootstrap_scores.csv", index=False)
    good_trials = [t for t in np.unique(ids) if np.isfinite(x[ids == t]).all()]
    sensitivity = dict(n_complete_trials=len(good_trials), status="not_needed")
    if len(good_trials) < 198:
        mask = np.isin(ids, good_trials)
        try:
            with threadpool_limits(limits=1):
                sp, _ = decode(x[mask], y[mask], folds[mask], ids[mask])
            ss = scores(y[mask], sp)
            sensitivity.update(status="completed", macro_f1=ss["macro_f1"], accuracy=ss["accuracy"])
            np.savez_compressed(root/"sensitivity_predictions.npz", row_indices=np.flatnonzero(mask), predictions=sp)
        except ValueError as error:
            sensitivity.update(status="unavailable", reason=str(error))
    write_json(root/"sensitivity.json", sensitivity)
    print(json.dumps(result, indent=2), flush=True)


def verify(root):
    if json.loads((root/"config.json").read_text()) != configuration():
        raise ValueError("Configuration changed")
    check_manifest(root/"source_input_hashes.json")
    check_manifest(root/"frozen_input_hashes.json")
    n_protected = check_manifest(root/"protected_prior_outputs.json")
    metadata, data = inputs(root)
    trials, raw_metadata, events, raw = load_raw_windows()
    np.testing.assert_array_equal(data["raw"], raw)
    pd.testing.assert_frame_equal(metadata[raw_metadata.columns], raw_metadata)
    pd.testing.assert_frame_equal(pd.read_csv(root/"inputs/event_samples.csv"), events)
    pd.testing.assert_frame_equal(pd.read_csv(root/"inputs/selected_trials.csv"), trials)
    assert len(metadata) == 1188 and data["clouds"].shape == (1188, 294, 3)
    features = pd.read_csv(root/"tables/features.csv", keep_default_na=False)
    x = features[COLUMNS].replace("", np.nan).to_numpy(dtype=float)
    np.testing.assert_allclose(x, np.load(root/"features.npz")["x"], equal_nan=True)
    for row in metadata.to_dict("records"):
        i = row["window_index"]
        if not row["preprocessing_error"]:
            np.testing.assert_array_equal(preprocess_window(raw[i]), data["processed"][i])
            np.testing.assert_array_equal(embed_window(data["processed"][i]), data["clouds"][i])
        result = read_fit(checkpoint(root, i), row, data["clouds"][i])
        np.testing.assert_allclose(geometry_features(result)[0], x[i], equal_nan=True)
        if result["status"] == "returned":
            distance = recompute_distances(data["clouds"][i], result["fit"], "one_torus")
            np.testing.assert_allclose(np.mean(distance**2), result["fit"]["mse"], rtol=1e-6, atol=1e-10)
    y, folds, ids = [metadata[k].to_numpy() for k in ("stage", "heldout_fold_zero_based", "trial_index")]
    validate_groups(y, folds, ids)
    with np.load(root/"heldout_predictions.npz") as z:
        pred = z["predictions"]
        for k, a in (("labels", y), ("folds", folds), ("trial_ids", ids)):
            np.testing.assert_array_equal(z[k], a)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        model = joblib.load(root/"models"/f"geometry_fold{fold}.joblib")
        assert list(model.named_steps) == ["impute", "scale", "lda"]
        assert model.n_features_in_ == 15
        np.testing.assert_allclose(model["impute"].statistics_, np.nanmedian(x[train], axis=0), atol=1e-13)
        np.testing.assert_allclose(model["scale"].mean_, model["impute"].transform(x[train]).mean(axis=0), atol=1e-13)
        np.testing.assert_array_equal(model.predict(x[test]), pred[test])
    s = scores(y, pred)
    np.testing.assert_allclose(s["confusion_fraction"].sum(axis=1), 1)
    for k in ("confusion_counts", "confusion_fraction"):
        np.testing.assert_allclose(pd.read_csv(root/"tables"/f"{k}.csv", index_col=0), s[k])
    null = np.load(root/"permutations.npz")
    assert len(null["index"]) == 1000
    for b in range(1000):
        np.testing.assert_array_equal(null["labels"][b], shuffle_stages(y, ids, b))
        ns = scores(null["labels"][b], null["predictions"][b])
        for k in ("macro_f1", "accuracy"):
            np.testing.assert_allclose(ns[k], null[k][b])
    # Rerun selected permutations to verify the saved null actually refits LDA.
    with threadpool_limits(limits=1):
        for b in (0, 499, 999):
            np.testing.assert_array_equal(permutation(b, x, y, folds, ids)["predictions"], null["predictions"][b])
    boot = np.load(root/"bootstrap.npz")
    expected = bootstrap(y, pred, ids, metadata.direction.to_numpy())
    for k in expected:
        np.testing.assert_allclose(boot[k], expected[k])
    summary = pd.read_csv(root/"tables/summary.csv").iloc[0]
    for k in ("macro_f1", "accuracy"):
        np.testing.assert_allclose(summary[k], s[k])
        np.testing.assert_allclose([summary[k+"_ci_low"], summary[k+"_ci_high"]], np.quantile(boot[k], [.025, .975]))
    assert np.isclose(summary.permutation_p, (1+(null["macro_f1"] >= s["macro_f1"]).sum())/1001)
    per_stage = pd.read_csv(root/"tables/per_stage_f1.csv")
    np.testing.assert_allclose(per_stage.f1, s["per_stage_f1"])
    np.testing.assert_allclose(per_stage[["ci_low", "ci_high"]].to_numpy().T,
                               np.quantile(boot["per_stage_f1"], [.025, .975], axis=0))
    sensitivity = json.loads((root/"sensitivity.json").read_text())
    if sensitivity["status"] == "completed":
        z = np.load(root/"sensitivity_predictions.npz")
        ix = z["row_indices"]
        with threadpool_limits(limits=1):
            expected_pred, _ = decode(x[ix], y[ix], folds[ix], ids[ix])
        np.testing.assert_array_equal(z["predictions"], expected_pred)
        np.testing.assert_allclose(sensitivity["macro_f1"], scores(y[ix], expected_pred)["macro_f1"])
    report = dict(passed=True, trials=198, windows=1188, points_per_cloud=294, features=15,
                  protected_files_unchanged=n_protected, fit_checkpoint_count=len(list((root/"checkpoints").glob("*.json"))),
                  usable_fits=int(np.isfinite(x).all(axis=1).sum()), permutation_count=1000, bootstrap_count=2000,
                  checks=["exact raw event slices", "direct observed delayed coordinates", "native half-width",
                          "saved folds grouped by trial", "training-only imputation and scaling",
                          "saved models reproduce predictions", "all null scores reproduce predictions",
                          "three null refits reproduced", "all clustered bootstrap scores reproduced",
                          "native 3D fit distances reproduced", "protected files and inputs unchanged"])
    write_json(root/"validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def write_report(root):
    s = pd.read_csv(root/"tables/summary.csv").iloc[0]
    accounting = pd.read_csv(root/"tables/fit_accounting.csv")
    params = pd.read_csv(root/"tables/fit_parameters_diagnostics.csv").fillna("")
    sensitivity = json.loads((root/"sensitivity.json").read_text())
    sensitivity_text = "All trials had six usable fits."
    if sensitivity["status"] == "completed":
        sensitivity_text = (f"Complete-fit sensitivity retained {sensitivity['n_complete_trials']} trials: "
                            f"macro-F1 {sensitivity['macro_f1']:.3f}, accuracy {sensitivity['accuracy']:.3f}.")
    elif sensitivity["status"] == "unavailable":
        sensitivity_text = f"Complete-fit sensitivity unavailable: {sensitivity['reason']}."
    failures = ", ".join(f"{r.stage_name}: {r.unusable}" for r in accounting.itertuples())
    circular_count = params["flags"].str.contains("nearly_circular_orientation").sum()
    text = f"""# Six-stage geometric decoding

Monkey T, session y070316009-12, channel 7: 198 short-delay trials, 33 per reach direction, yielded 1,188 windows (198 per stage). Each 300-ms window was independently detrended, median/MAD-normalized, and zero-phase filtered at 2-55 Hz. Direct embedding used m=3 and tau=3 ms, producing 294 observed points. No PCA reduction, AR signal replacement, PINN, or persistent homology was run.

The fixed annular-band fitter provided 15 features, including band half-width. {int(s.usable_fits)}/1,188 fits were usable; unusable counts: {failures}. Boundary flags were retained; {circular_count} fits had nearly circular, potentially unstable in-plane orientations. Missing feature rows were training-median imputed.

Five-fold LDA kept all six windows from each trial together. Held-out macro-F1 was **{s.macro_f1:.3f}** (95% conditional interval {s.macro_f1_ci_low:.3f}-{s.macro_f1_ci_high:.3f}); accuracy was **{s.accuracy:.3f}** ({s.accuracy_ci_low:.3f}-{s.accuracy_ci_high:.3f}). Intervals used 2,000 whole-trial bootstraps stratified by reach direction. The 1,000 within-trial stage permutations gave macro-F1 p={s.permutation_p:.4f}; null median {s.null_macro_f1_median:.3f}. {sensitivity_text}

Examples are original trials 6, 9, and 10, chosen by ID, not separation. Radii and half-width are normalized, not absolute voltage measures. Post-cue windows include delay activity; post-GO includes reaction and potentially movement. Short-window filtering, fit quality, temporal drift, and this previously inspected single recording limit interpretation. This is offline within-recording stage prediction, not cross-animal validation, causal decoding, or proof of distinct topologies. The selected annular model is a representation, not a verified topology for every stage.
"""
    (root/"report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "pilot", "fit", "decode", "verify", "all"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.parent != (UNIT/"outputs").resolve() or root.name != OUTPUT.name:
        raise ValueError("Write only to the named sibling experiment")
    initialize(root, args.resume)
    prepare(root)
    if args.phase == "pilot":
        fit_all(root, args.jobs, pilot=True)
    if args.phase in ("fit", "all"):
        fit_all(root, args.jobs)
        collect_features(root)
    if args.phase in ("decode", "all"):
        evaluate(root, args.jobs)
    if args.phase in ("verify", "all"):
        with threadpool_limits(limits=1):
            verify(root)
        write_report(root)


if __name__ == "__main__":
    main()
