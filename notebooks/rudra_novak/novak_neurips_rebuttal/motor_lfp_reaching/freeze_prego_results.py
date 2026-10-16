#!/usr/bin/env python
"""Freeze audited pre-GO predictions, paired statistics, and illustrative features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from prego_decoding import MAIN_METHODS, METHODS, TORUS_COLUMNS, stable_seed
from prego_statistics import cluster_interval, holm, prediction_metrics, select_example

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "prego_single_channel"
COMPARISONS = [("D", "torus", "power_frequency"), ("D", "torus", "all_bands"),
               ("E", "torus_power_frequency", "power_frequency"), ("E", "torus_all_bands", "all_bands")]


def freeze(output):
    cache, tables = output / "cache", output / "tables"
    tables.mkdir(exist_ok=True)
    fit_run = json.loads((output / "fit_run.json").read_text())
    decode_run = json.loads((output / "decode_run.json").read_text())
    if fit_run["status"] != "complete" or decode_run["status"] != "complete":
        raise ValueError("Cannot freeze an unfinished run")
    if set(fit_run["recordings"]) != set(decode_run["recordings"]):
        raise ValueError("Fit and decode inventories differ")
    predictions, audits, failures, nulls, failed_fits = [], [], [], [], []
    embeddings = []
    for recording in fit_run["recordings"]:
        root = cache / recording
        metadata = json.loads((root / "fit_complete.json").read_text())
        decoded = json.loads((root / "decode_complete.json").read_text())
        status = {d["analysis"]: d for d in decoded["analyses"]}
        audit = {k: v for k, v in metadata.items() if k not in ("analyses", "minimum_per_direction")}
        audit.update({"minimum_"+k: v for k, v in metadata["minimum_per_direction"].items()})
        main = status.get("short", {}).get("status") == "included"
        matched = main and all(status.get(a, {}).get("status") == "included" for a in ("matched_short", "matched_long"))
        audit.update(main_included=main, matched_included=matched, fit_error=np.nan)
        for analysis, item in status.items():
            if item["status"] != "included":
                failures.append(dict(recording=recording, **item))
                continue
            if not main or (analysis != "short" and not matched):
                continue
            folder = root / analysis
            pred = pd.read_csv(folder / "predictions.csv")
            if set(pred.method) != set(METHODS):
                raise ValueError("Incomplete method set")
            if pred.groupby("method").raw_trial_index.apply(tuple).nunique() != 1:
                raise ValueError("Methods do not share trials")
            predictions.append(pred)
            success_counts, errors = [], []
            for fold in range(5):
                model = json.loads((folder / f"fold_{fold}.json").read_text())
                embeddings.append(dict(recording=recording, monkey=metadata["monkey"], analysis=analysis,
                    fold=fold, dimension=model["dimension"], tau_samples=model["tau"],
                    n_training_trials=model["n_training_trials"], tau_method=model["tau_method"],
                    robust_peaks_hz=json.dumps(model["robust_peaks"]), adjustment=model["adjustment"]))
                with np.load(folder / f"fold_{fold}.npz") as saved:
                    success_counts.extend(saved["success"].tolist())
                    errors.extend(saved["mse"][saved["success"]].tolist())
                    for index in np.flatnonzero(~saved["success"]):
                        failed_fits.append(dict(recording=recording, monkey=metadata["monkey"], analysis=analysis,
                            embedding_fold=fold, raw_trial_index=int(saved["raw_trial_indices"][index]),
                            nfev=int(saved["nfev"][index]), cost=float(saved["cost"][index]),
                            mse=float(saved["mse"][index]), action="training_median_imputation"))
            audit[f"{analysis}_successful_fit_fraction"] = float(np.mean(success_counts))
            audit[f"{analysis}_trials"] = item["n_trials"]
            if analysis == "short":
                audit["fit_error"] = float(np.median(errors))
                nulls.append(pd.read_csv(folder / "null_scores.csv"))
        audits.append(audit)
    audit = pd.DataFrame(audits)
    audit.to_csv(tables / "recording_audit.csv", index=False)
    pd.DataFrame(embeddings).to_csv(tables / "embedding_selection.csv", index=False)
    pd.DataFrame(failed_fits, columns=["recording", "monkey", "analysis", "embedding_fold", "raw_trial_index",
                                      "nfev", "cost", "mse", "action"]).to_csv(tables / "fit_failures.csv", index=False)
    pd.DataFrame(failures, columns=["recording", "analysis", "status", "reason"]).to_csv(tables / "exclusions_geometry.csv", index=False)
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_csv(tables / "heldout_predictions.csv", index=False)
    score_rows, direction_rows, confusion_rows = [], [], []
    keys = ["recording", "monkey", "day", "analysis", "method"]
    for values, group in prediction.groupby(keys, sort=True):
        info = dict(zip(keys, values))
        score, classes, matrix = prediction_metrics(group.true_direction, group.predicted_direction)
        if group.raw_trial_index.duplicated().any() or set(group.fold) != set(range(5)):
            raise ValueError("Invalid held-out trial inventory")
        score_rows.append(dict(**info, macro_f1=score, n_trials=len(group)))
        direction_rows.extend(dict(**info, direction=i+1, f1=float(value)) for i, value in enumerate(classes))
        confusion_rows.extend(dict(**info, true_direction=i+1, predicted_direction=j+1, fraction=matrix[i, j])
                              for i in range(6) for j in range(6))
    scores = pd.DataFrame(score_rows)
    directions = pd.DataFrame(direction_rows)
    confusion = pd.DataFrame(confusion_rows)
    scores.to_csv(tables / "recording_scores.csv", index=False)
    directions.to_csv(tables / "direction_scores.csv", index=False)
    confusion.to_csv(tables / "recording_confusions.csv", index=False)
    summary = scores.groupby(["analysis", "monkey", "method"]).agg(
        mean=("macro_f1", "mean"), sd=("macro_f1", "std"), n_lfps=("recording", "nunique"),
        n_days=("day", "nunique"), n_trials=("n_trials", "sum")).reset_index()
    summary["n_animals"] = 1
    summary.to_csv(tables / "summary.csv", index=False)
    directions.groupby(["analysis", "monkey", "method", "direction"]).f1.agg(["mean", "std", "count"]).reset_index().to_csv(tables / "direction_summary.csv", index=False)
    confusion.groupby(["analysis", "monkey", "method", "true_direction", "predicted_direction"]).fraction.mean().reset_index().to_csv(tables / "mean_confusions.csv", index=False)
    null = pd.concat(nulls, ignore_index=True)
    null.to_csv(tables / "null_recording_scores.csv", index=False)
    null_group = null.groupby(["monkey", "method", "permutation"]).macro_f1.mean().reset_index()
    null_summary = null_group.groupby(["monkey", "method"]).macro_f1.agg(
        mean="mean", low=lambda x: x.quantile(.025), high=lambda x: x.quantile(.975), n_permutations="size").reset_index()
    null_summary.to_csv(tables / "null_summary.csv", index=False)
    all_tests, pairs, days = [], [], []
    for monkey in ("M", "T"):
        primary = scores[(scores.analysis == "short") & (scores.monkey == monkey)]
        wide = primary.pivot(index=["recording", "day"], columns="method", values="macro_f1")
        for panel, enhanced, baseline in COMPARISONS:
            difference = (wide[enhanced] - wide[baseline]).rename("difference").reset_index()
            difference["monkey"], difference["enhanced"], difference["baseline"] = monkey, enhanced, baseline
            day = difference.groupby("day").difference.mean().reset_index()
            day["difference"] = day.difference.round(12)
            day["monkey"], day["enhanced"], day["baseline"] = monkey, enhanced, baseline
            result = wilcoxon(day.difference.to_numpy(), alternative="two-sided", zero_method="wilcox", method="auto") if np.any(day.difference != 0) else None
            mean, low, high = cluster_interval(difference, stable_seed(monkey+enhanced+baseline))
            all_tests.append(dict(panel=panel, monkey=monkey, enhanced=enhanced, baseline=baseline,
                n_lfps=len(difference), n_days=len(day), n_animals=1, mean_difference=mean,
                ci_low=low, ci_high=high, day_mean_difference=float(day.difference.mean()),
                statistic=float(result.statistic) if result else 0., p_raw=float(result.pvalue) if result else 1.))
            pairs.append(difference)
            days.append(day)
    tests = pd.DataFrame(all_tests)
    tests["p_holm"] = holm(tests.p_raw)
    tests.to_csv(tables / "planned_comparisons.csv", index=False)
    pd.concat(pairs).to_csv(tables / "paired_differences.csv", index=False)
    pd.concat(days).to_csv(tables / "paired_day_means.csv", index=False)
    examples, example_features = [], []
    for monkey in ("M", "T"):
        eligible = audit[audit.main_included & (audit.monkey == monkey)]
        recording = select_example(eligible)
        row = eligible[eligible.recording == recording].iloc[0]
        examples.append(dict(monkey=monkey, recording=recording, fit_error=row.fit_error,
                             animal_median_fit_error=float(eligible.fit_error.median())))
        folder = cache / recording / "short"
        with np.load(folder / "features.npz") as saved:
            frame = pd.DataFrame(saved["torus"][0], columns=TORUS_COLUMNS)
            frame["direction"] = saved["labels"]
            frame["fold"] = saved["folds"]
            frame["raw_trial_index"] = saved["raw_trial_indices"]
        frame["monkey"], frame["recording"], frame["reference_fold"] = monkey, recording, 0
        example_features.append(frame)
    pd.DataFrame(examples).to_csv(tables / "example_selection.csv", index=False)
    pd.concat(example_features).to_csv(tables / "example_geometry.csv", index=False)
    manifest = dict(status="frozen", config=json.loads((output / "config.json").read_text()),
                    table_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tables.glob("*.csv"))})
    (output / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    freeze(parser.parse_args().output)
