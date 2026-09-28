"""Independent saved-artifact checks; no geometric fitting or analysis writes elsewhere."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from plot_stage_normalization_geometry import protected_hashes
from prego_geometric_fits import array_hash, sha256, write_json
from motor_lfp_utils import lag_embed
from run_stage_additive_spectral import METHODS, spectral_features
from run_stage_hilbert_sweep import ROOT, SOURCE, GRID, generic_decode
from stage_geometry_decoding import COLUMNS, geometry_features, shuffle_stages


def count_metrics(labels, predictions):
    counts = np.bincount(labels*6+predictions, minlength=36).reshape(6, 6)
    denominators = counts.sum(axis=0) + counts.sum(axis=1)
    f1 = np.divide(2*counts.diagonal(), denominators, out=np.zeros(6), where=denominators > 0)
    return f1.mean(), f1


def main():
    metadata = pd.read_csv(ROOT / "inputs/window_metadata.csv")
    selection = json.loads((ROOT / "selection.json").read_text())
    overall = pd.read_csv(ROOT / "tables/overall_f1.csv")
    y = metadata.stage.to_numpy()
    ids = metadata.trial_index.to_numpy()
    folds = metadata.heldout_fold_zero_based.to_numpy()
    events = pd.read_csv(ROOT / "inputs/event_samples.csv").set_index("trial_index")
    anchors = ("TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO")
    codes = (202, 203, 204, 206, 207, 207)
    assert len(overall[overall.tau_ms.isin(GRID)]) == 25
    all_delay_metrics = []
    for row in overall.itertuples():
        if row.decoder_status != "ok":
            all_delay_metrics.append(dict(tau_ms=row.tau_ms, status="failed", reason=row.decoder_reason))
            continue
        with np.load(ROOT / "predictions" / f"native_tau_{row.tau_ms:02d}.npz") as saved:
            predictions = saved["predictions"]
            np.testing.assert_array_equal(saved["labels"], y)
            np.testing.assert_array_equal(saved["folds"], folds)
            np.testing.assert_array_equal(saved["trial_ids"], ids)
            measured = f1_score(y, predictions, average="macro", labels=np.arange(6), zero_division=0)
            np.testing.assert_allclose(measured, row.macro_f1, atol=1e-12)
            all_delay_metrics.append(dict(tau_ms=row.tau_ms, status="ok", macro_f1=measured,
                accuracy=float(np.mean(y == predictions)), usable_fits=int(row.usable_fits),
                windows=len(y), points_per_cloud=300-2*int(row.tau_ms)))
    pd.DataFrame(all_delay_metrics).to_csv(ROOT / "tables/all_delay_scores.csv", index=False, float_format="%.17g")
    with np.load(SOURCE / "bootstrap.npz") as saved:
        samples = saved["trial_indices"].copy()
    trial_rows = np.stack([np.flatnonzero(ids == i) for i in range(198)])
    for trial in range(198):
        rows = metadata[metadata.trial_index == trial].sort_values("stage")
        assert rows.heldout_fold_zero_based.nunique() == 1
        assert rows.stage.tolist() == list(range(6))
        assert (rows.start_sample.to_numpy()[1:] >= rows.end_sample_exclusive.to_numpy()[:-1]).all()
        assert int(events.loc[trial, "original_trial_number"]) == int(rows.iloc[0].original_trial_number)
        for row in rows.itertuples():
            anchor, code = anchors[row.stage], codes[row.stage]
            actual_event = int(events.loc[trial, anchor + "_sample"])
            assert row.anchor_name == anchor and row.anchor_code == code
            assert row.event_sample == actual_event
            assert (row.start_sample if row.stage % 2 else row.end_sample_exclusive) == actual_event
            assert row.end_sample_exclusive - row.start_sample == 300
    with np.load(ROOT / "inputs/windows.npz") as saved:
        raw = saved["raw"].copy()
        bands, average = spectral_features(raw)
        processed = saved["processed"].copy()
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        np.testing.assert_array_equal(raw, saved["raw"])
        for i, cloud in enumerate(saved["clouds"]):
            assert array_hash(cloud) == metadata.iloc[i].cloud_sha256
    envelope_metadata = metadata.rename(columns={"cloud_sha256": "source_mad_tau3_cloud_sha256"}).copy()
    envelope_metadata["envelope_signal_sha256"] = [array_hash(x) for x in processed]
    envelope_metadata["envelope_tau3_cloud_sha256"] = [array_hash(lag_embed(x, 3, 3)) for x in processed]
    envelope_metadata["normalization_usable"] = np.isfinite(processed).all(axis=1)
    envelope_metadata.to_csv(ROOT / "inputs/envelope_window_metadata.csv", index=False)
    checked = []
    for tau in sorted({t for t in (selection["f1_tau_ms"], selection["ami_tau_ms"]) if t is not None}):
        out = ROOT / "selected" / f"tau_{tau:02d}"
        with np.load(out / "features.npz") as saved:
            features = {m: saved[m].copy() for m in METHODS}
        with np.load(out / "predictions.npz") as saved:
            predictions = saved["predictions"].copy()
            np.testing.assert_array_equal(saved["labels"], y)
            np.testing.assert_array_equal(saved["folds"], folds)
            np.testing.assert_array_equal(saved["trial_ids"], ids)
        np.testing.assert_allclose(features["all_bands"], bands, rtol=0, atol=0)
        np.testing.assert_allclose(features["average_psd"], average, rtol=0, atol=0)
        for i in range(1188):
            result = json.loads((ROOT / "checkpoints/native" / f"tau_{tau:02d}" / f"window_{i:04d}.json").read_text())
            expected, reason = geometry_features(result)
            np.testing.assert_allclose(features["geometry"][i], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        summary = pd.read_csv(out / "summary.csv").set_index("method")
        with np.load(out / "bootstrap.npz") as saved:
            boot, per_stage = saved["macro_f1"].copy(), saved["per_stage_f1"].copy()
            np.testing.assert_array_equal(saved["trial_indices"], samples)
        for j, method in enumerate(METHODS):
            value = f1_score(y, predictions[:,j], average="macro", labels=np.arange(6), zero_division=0)
            np.testing.assert_allclose(value, summary.loc[method, "macro_f1"], atol=1e-12)
            counts = confusion_matrix(y, predictions[:,j], labels=np.arange(6))
            recall = pd.read_csv(out / f"recall_{method}.csv", index_col=0).to_numpy()
            np.testing.assert_allclose(recall, counts/counts.sum(axis=1, keepdims=True), atol=1e-12)
            for fold in range(5):
                model = joblib.load(out / f"{method}_fold_{fold}.joblib")
                train, test = folds != fold, folds == fold
                np.testing.assert_allclose(model.named_steps["impute"].statistics_, np.nanmedian(features[method][train], axis=0), atol=1e-12)
                imputed = model.named_steps["impute"].transform(features[method][train])
                np.testing.assert_allclose(model.named_steps["scale"].mean_, imputed.mean(axis=0), atol=1e-12)
                np.testing.assert_array_equal(model.predict(features[method][test]), predictions[test,j])
            for i, sample in enumerate(samples):
                indices = trial_rows[sample].ravel()
                macro, f1 = count_metrics(y[indices], predictions[indices,j])
                np.testing.assert_allclose(boot[i,j], macro, atol=1e-12)
                np.testing.assert_allclose(per_stage[i,j], f1, atol=1e-12)
        paired = pd.read_csv(out / "paired_differences.csv")
        for row in paired.itertuples():
            a, b = METHODS.index(row.added), METHODS.index(row.baseline)
            np.testing.assert_allclose([row.ci_low, row.ci_high], np.quantile(boot[:,a]-boot[:,b], [.025,.975]), atol=1e-12)
            np.testing.assert_allclose(row.difference, summary.loc[row.added,"macro_f1"] - summary.loc[row.baseline,"macro_f1"], atol=1e-12)
        with np.load(out / "permutations.npz") as saved:
            null_scores = saved["macro_f1"].copy()
            assert saved["labels"].shape == saved["predictions"].shape == (1000, 1188)
            for i in range(1000):
                np.testing.assert_array_equal(saved["labels"][i], shuffle_stages(y, ids, i))
                np.testing.assert_allclose(saved["macro_f1"][i], count_metrics(saved["labels"][i], saved["predictions"][i])[0], atol=1e-12)
        measured = summary.loc["geometry", "macro_f1"]
        metrics = json.loads((out / "metrics.json").read_text())
        np.testing.assert_allclose(metrics["permutation_p"], (1+np.count_nonzero(null_scores >= measured))/1001)
        if (out / "complete_trial_predictions.npz").exists():
            with np.load(out / "complete_trial_predictions.npz") as saved:
                complete_y, complete_pred = saved["labels"], saved["predictions"]
                np.testing.assert_allclose(f1_score(complete_y, complete_pred, average="macro", labels=np.arange(6)), metrics["sensitivity"]["macro_f1"], atol=1e-12)
        checked.append(tau)
    comparison = []
    for criterion, tau in (("F1 exploratory winner", selection["f1_tau_ms"]),
                           ("AMI first pooled minimum", selection["ami_tau_ms"])):
        if tau is None:
            comparison.append(dict(criterion=criterion, status="no_AMI_interior_minimum"))
            continue
        out = ROOT / "selected" / f"tau_{tau:02d}"
        summary = pd.read_csv(out / "summary.csv").set_index("method").loc["geometry"]
        metrics = json.loads((out / "metrics.json").read_text())
        unusable = int(pd.read_csv(out / "fit_accounting.csv").unusable.sum())
        comparison.append(dict(criterion=criterion, tau_ms=tau, status="ok",
            macro_f1=float(summary.macro_f1), ci_low=float(summary.ci_low), ci_high=float(summary.ci_high),
            accuracy=float(summary.accuracy), fixed_delay_nominal_permutation_p=metrics["permutation_p"],
            usable_fits=1188-unusable, unusable_fits=unusable,
            complete_trials=metrics["sensitivity"]["complete_trials"],
            complete_trial_macro_f1=metrics["sensitivity"].get("macro_f1"),
            inference="exploratory; not corrected for delay selection"))
    pd.DataFrame(comparison).to_csv(ROOT / "tables/selected_delay_comparison.csv", index=False, float_format="%.17g")
    assert protected_hashes() == json.loads((ROOT / "protected_hashes.json").read_text())
    write_json(ROOT / "independent_validation.json", dict(selected_delays=checked,
               feature_definitions_verified=True, all_selected_scores_verified_with_sklearn=True,
               all_bootstraps_reproduced=True, paired_intervals_reproduced=True,
               all_null_labels_and_scores_reproduced=True, train_only_scaling_verified=True,
               all_event_anchors_compared_to_timestamp_table=True,
               source_and_envelope_hashes_disambiguated=True,
               protected_results_unchanged=True, validator_sha256=sha256(Path(__file__))))
    print("Independent validation passed", flush=True)


if __name__ == "__main__":
    main()
