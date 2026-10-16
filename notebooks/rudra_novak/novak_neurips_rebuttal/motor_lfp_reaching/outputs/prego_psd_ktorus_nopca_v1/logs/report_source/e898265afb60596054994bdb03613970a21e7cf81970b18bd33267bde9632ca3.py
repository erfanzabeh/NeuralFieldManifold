#!/usr/bin/env python
"""Freeze, verify and render completed no-PCA decoding. Never fits geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from prego_decoding import stable_seed
from prego_ktorus import MAIN_METHODS, METHODS, decode, phase_design, trial_cloud
from prego_statistics import cluster_interval, holm, prediction_metrics
from run_prego_ktorus import (DEFAULT_OUTPUT, PREVIOUS, CONVERTED_DIR, old_output_hashes,
                             save_table, sha256, utc, write_json, validate_output,
                             verify_artifacts)
from run_prego_single_channel import load_recording

LABELS = dict(peak_power="Peak\npower", peak_frequency="Peak\nfrequency",
              power_frequency="Power +\nfrequency", all_bands="All band\npowers",
              ktorus="K-torus\ngeometry", ktorus_power_frequency="Geometry + power/frequency",
              ktorus_all_bands="Geometry + all bands")
COLORS = dict(peak_power="#6c899b", peak_frequency="#b7c7cf", power_frequency="#36697f",
              all_bands="#8d9292", ktorus="#b53a32", ktorus_power_frequency="#a02e29",
              ktorus_all_bands="#a02e29")
COMPARISONS = [("standalone", "ktorus", "power_frequency"),
               ("standalone", "ktorus", "all_bands"),
               ("additive", "ktorus_power_frequency", "power_frequency"),
               ("additive", "ktorus_all_bands", "all_bands")]


def manual_f1(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    values = []
    for label in range(1, 7):
        tp = np.sum((truth == label) & (predicted == label))
        denominator = np.sum(truth == label)+np.sum(predicted == label)
        values.append(float(2*tp/denominator) if denominator else 0.)
    return float(np.mean(values)), np.array(values)


def summarize_predictions(predictions):
    scores, directions, confusions = [], [], []
    keys = ["recording", "monkey", "day", "method", "geometry_evaluable"]
    for values, group in predictions.groupby(keys):
        meta = dict(zip(keys, values))
        if group.raw_trial_index.duplicated().any() or set(group.fold) != set(range(5)):
            raise ValueError("Duplicate or incomplete held-out predictions")
        score, classes, matrix = prediction_metrics(group.true_direction, group.predicted_direction)
        independent_score, independent_classes = manual_f1(group.true_direction, group.predicted_direction)
        np.testing.assert_allclose(score, independent_score, atol=1e-12)
        np.testing.assert_allclose(classes, independent_classes, atol=1e-12)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.)
        scores.append(dict(**meta, macro_f1=score, n_trials=len(group)))
        directions.extend(dict(**meta, direction=i+1, f1=value) for i, value in enumerate(classes))
        confusions.extend(dict(**meta, true_direction=i+1, predicted_direction=j+1, fraction=matrix[i, j])
                          for i in range(6) for j in range(6))
    return pd.DataFrame(scores), pd.DataFrame(directions), pd.DataFrame(confusions)


def planned_statistics(scores):
    tests, pairs, days = [], [], []
    for monkey in ("M", "T"):
        wide = scores[scores.monkey == monkey].pivot(index=["recording", "day"], columns="method", values="macro_f1")
        for family, enhanced, baseline in COMPARISONS:
            difference = (wide[enhanced]-wide[baseline]).rename("difference").reset_index()
            difference["monkey"], difference["enhanced"], difference["baseline"] = monkey, enhanced, baseline
            daily = difference.groupby("day").difference.mean().round(12).reset_index()
            daily["monkey"], daily["enhanced"], daily["baseline"] = monkey, enhanced, baseline
            result = wilcoxon(daily.difference, alternative="two-sided", zero_method="wilcox", method="auto") if np.any(daily.difference != 0) else None
            mean, low, high = cluster_interval(difference, stable_seed(monkey+enhanced+baseline))
            tests.append(dict(monkey=monkey, family=family, enhanced=enhanced, baseline=baseline,
                              n_lfps=len(difference), n_days=len(daily), n_animals=1,
                              mean_difference=mean, ci_low=low, ci_high=high,
                              day_mean_difference=daily.difference.mean(),
                              statistic=float(result.statistic) if result else 0.,
                              p_raw=float(result.pvalue) if result else 1.))
            pairs.append(difference)
            days.append(daily)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_raw)
    return tests, pd.concat(pairs, ignore_index=True), pd.concat(days, ignore_index=True)


def freeze(root):
    validate_output(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["state"] not in ("decoding_complete", "verified"):
        raise ValueError("Cannot report an incomplete experiment")
    if not manifest.get("all_seven_nulls_complete"):
        raise ValueError("Run the full-coordinate integrity/completion stage first")
    folders = sorted((root / "cache").glob("*"))
    if len(folders) != 341:
        raise ValueError("Source recording inventory is incomplete")
    audits, predictions, embeddings, diagnostics, nulls = [], [], [], [], []
    for folder in folders:
        meta = json.loads((folder / "decode_complete.json").read_text())
        meta["geometry_evaluable"] = meta.get("geometry_evaluable", False)
        audits.append({k: v for k, v in meta.items() if k != "unresolved_folds"})
        if meta["status"] != "eligible":
            continue
        integrity = json.loads((folder / "integrity_complete.json").read_text())
        verify_artifacts(folder, integrity["artifacts"])
        prediction = pd.read_csv(folder / "predictions.csv")
        expected_methods = METHODS if meta["geometry_evaluable"] else MAIN_METHODS[:4]
        if set(prediction.method) != set(expected_methods):
            raise ValueError("Incomplete method set")
        with np.load(folder / "trials.npz") as saved:
            trials = {k: saved[k] for k in saved.files}
        geometry = []
        for fold in range(5):
            params = json.loads((folder / f"fold_{fold}.json").read_text())
            np.testing.assert_array_equal(params["training_trial_indices"], trials["raw_trial_indices"][trials["folds"] != fold])
            embeddings.append(dict(recording=folder.name, monkey=meta["monkey"], fold=fold,
                status=params["status"], K=params["K"], dimension=params["dimension"], tau_samples=params["tau"],
                condition=params.get("condition"), feature_count=len(params["feature_names"]),
                n_training_trials=params["n_training_trials"], frequencies_hz=json.dumps(params["frequencies"]),
                training_trials_per_feature=params["n_training_trials"]/len(params["feature_names"]) if params["feature_names"] else np.nan,
                harmonic_ambiguities=json.dumps(params["harmonic_ambiguities"]),
                geometry_evaluable=meta["geometry_evaluable"]))
            if params["status"] == "ok":
                with np.load(folder / f"fold_{fold}.npz") as saved:
                    geometry.append(saved["features"] if meta["geometry_evaluable"] else np.zeros((len(trials["labels"]), 1)))
                    assert saved["features"].shape[1] == len(params["feature_names"])
                diag = pd.read_csv(folder / f"fold_{fold}_diagnostics.csv")
                diag["recording"], diag["monkey"], diag["fold"] = folder.name, meta["monkey"], fold
                diagnostics.append(diag)
            else:
                geometry.append(np.zeros((len(trials["labels"]), 1)))
        reproduced = decode(trials["power"], trials["frequency"], trials["bands"], geometry,
                            trials["labels"], trials["folds"], expected_methods)
        for method, values in reproduced.items():
            group = prediction[prediction.method == method]
            np.testing.assert_array_equal(group.raw_trial_index, trials["raw_trial_indices"])
            np.testing.assert_array_equal(group.true_direction, trials["labels"])
            np.testing.assert_array_equal(group.predicted_direction, values)
        # The four unchanged spectral decoders must reproduce their predecessor.
        previous = pd.read_csv(PREVIOUS / "cache" / folder.name / "short" / "predictions.csv")
        for method in MAIN_METHODS[:4]:
            a = prediction[prediction.method == method].sort_values("raw_trial_index")
            b = previous[previous.method == method].sort_values("raw_trial_index")
            np.testing.assert_array_equal(a.predicted_direction, b.predicted_direction)
        predictions.append(prediction)
        if meta["geometry_evaluable"]:
            null = pd.read_csv(folder / "null_scores.csv")
            assert len(null) == 200*len(METHODS)
            assert not null.duplicated(["method", "permutation"]).any()
            assert set(null.method) == set(METHODS)
            assert set(null.permutation) == set(range(200))
            nulls.append(null)
    audit, embedding = pd.DataFrame(audits), pd.DataFrame(embeddings)
    predicted = pd.concat(predictions, ignore_index=True)
    diag = pd.concat(diagnostics, ignore_index=True)
    scores, directions, confusions = summarize_predictions(predicted)
    main = scores[scores.geometry_evaluable].copy()
    tables = root / "tables"
    for name, frame in [("recording_audit", audit), ("embedding_selection", embedding),
                        ("fit_diagnostics", diag), ("heldout_predictions", predicted),
                        ("recording_scores", scores), ("direction_scores", directions),
                        ("recording_confusions", confusions)]:
        save_table(tables / f"{name}.csv", frame)
    summary = main.groupby(["monkey", "method"]).agg(mean=("macro_f1", "mean"), sd=("macro_f1", "std"),
        n_lfps=("recording", "nunique"), n_days=("day", "nunique"), n_trials=("n_trials", "sum")).reset_index()
    save_table(tables / "summary.csv", summary)
    save_table(tables / "full_cohort_spectral_summary.csv", scores[scores.method.isin(MAIN_METHODS[:4])].groupby(
        ["monkey", "method"]).macro_f1.agg(["mean", "std", "count"]).reset_index())
    save_table(tables / "direction_summary.csv", directions[directions.geometry_evaluable].groupby(
        ["monkey", "method", "direction"]).f1.agg(["mean", "std", "count"]).reset_index())
    save_table(tables / "mean_confusions.csv", confusions[confusions.geometry_evaluable].groupby(
        ["monkey", "method", "true_direction", "predicted_direction"]).fraction.mean().reset_index())
    null = pd.concat(nulls, ignore_index=True)
    save_table(tables / "null_recording_scores.csv", null)
    null_mean = null.groupby(["monkey", "method", "permutation"]).macro_f1.mean().reset_index()
    save_table(tables / "null_summary.csv", null_mean.groupby(["monkey", "method"]).macro_f1.agg(
        mean="mean", low=lambda x: x.quantile(.025), high=lambda x: x.quantile(.975)).reset_index())
    tests, pairs, days = planned_statistics(main)
    save_table(tables / "planned_comparisons.csv", tests)
    save_table(tables / "paired_differences.csv", pairs)
    save_table(tables / "paired_day_means.csv", days)
    old = pd.read_csv(PREVIOUS / "tables" / "recording_scores.csv")
    old = old[(old.analysis == "short") & (old.method == "torus")][["recording", "macro_f1"]].rename(columns={"macro_f1": "old_pca_tube_f1"})
    comparison = main[main.method == "ktorus"].merge(old, on="recording", validate="one_to_one")
    comparison["difference"] = comparison.macro_f1-comparison.old_pca_tube_f1
    save_table(tables / "exploratory_predecessor_comparison.csv", comparison)
    assert old_output_hashes() == manifest["provenance"]["predecessor"]
    write_json(root / "validation.json", dict(utc=utc(), recordings_audited=len(audit),
        eligible_recordings=int((audit.status == "eligible").sum()),
        geometry_evaluable_recordings=int(audit.geometry_evaluable.sum()),
        heldout_predictions_reproduced=len(predicted), spectral_predictions_unchanged=True,
        manual_f1_verified=True, confusion_rows_verified=True, null_permutations_verified=200,
        predecessor_unchanged=True, primary_tests=len(tests),
        reporter_sha256=sha256(Path(__file__)), table_sha256={p.name: sha256(p) for p in tables.glob("*.csv")}))
    return audit, embedding, diag


def save_figure(root, name, figure, data, caption):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(root / "plots" / f"{name}.{extension}", dpi=600, facecolor="white")
    save_table(root / "plots" / f"{name}.csv", data)
    (root / "plots" / f"{name}.txt").write_text(caption+"\n")
    plt.close(figure)


def verify_frozen_tables(root):
    validation = json.loads((root / "validation.json").read_text())
    for name, digest in validation["table_sha256"].items():
        path = root / "tables" / name
        if not path.exists() or sha256(path) != digest:
            raise ValueError(f"Frozen table changed: {name}")


def verify_example_inputs(folder):
    integrity = json.loads((folder / "integrity_complete.json").read_text())
    verify_artifacts(folder, integrity["artifacts"])
    meta = json.loads((folder / "decode_complete.json").read_text())
    if sha256(CONVERTED_DIR / f"{folder.name}.npz") != meta["source_sha256"]:
        raise ValueError(f"Example source changed: {folder.name}")


def archive_report(root):
    digest = sha256(Path(__file__))
    snapshot = root / "logs" / "report_source" / f"{digest}.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and sha256(snapshot) != digest:
        raise ValueError("Reporting source archive changed")
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    validation = json.loads((root / "validation.json").read_text())
    validation.update(reporter_sha256=digest, reporter_argv=sys.argv,
        report_source=str(snapshot.relative_to(root)),
        table_sha256={p.name: sha256(p) for p in (root / "tables").glob("*.csv")},
        plot_sha256={p.name: sha256(p) for p in (root / "plots").iterdir() if p.is_file()})
    write_json(root / "validation.json", validation)
    return digest


def coordinate_views(observed, fitted, tau, title):
    m = observed.shape[1]
    columns = 2 if m-1 == 4 else min(3, m-1)
    rows = int(np.ceil((m-1)/columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.35*columns, 2.4*rows+.3),
                                squeeze=False, layout="constrained")
    for j, axis in enumerate(axes.flat, start=1):
        if j >= m:
            figure.delaxes(axis)
            continue
        axis.plot(observed[:, 0], observed[:, j], color="#617884", lw=.65, alpha=.8, label="Observed")
        axis.plot(fitted[:, 0], fitted[:, j], color="#b53a32", lw=.8, label="K-mode fit")
        axis.set(xlabel="x(t)", ylabel=f"x(t - {j*tau} ms)")
        axis.set_aspect("equal", adjustable="datalim")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False, ncol=2, fontsize=8)
    figure.suptitle(title+"\nDisplay only; full-coordinate fit", fontsize=10)
    return figure


def render(root):
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    tables = root / "tables"
    summary = pd.read_csv(tables / "summary.csv")
    null = pd.read_csv(tables / "null_summary.csv")
    tests = pd.read_csv(tables / "planned_comparisons.csv")
    pairs = pd.read_csv(tables / "paired_differences.csv")
    confusion = pd.read_csv(tables / "mean_confusions.csv")
    direction = pd.read_csv(tables / "direction_summary.csv")
    embedding = pd.read_csv(tables / "embedding_selection.csv")
    ylim = max(.30, np.ceil((summary[summary.method.isin(MAIN_METHODS)].eval("mean+sd").max()+.08)*20)/20)
    delta_max = max(.05, np.ceil(abs(pairs[pairs.enhanced != "ktorus"].difference).max()*20)/20+.05)
    for monkey in ("M", "T"):
        group = summary[summary.monkey == monkey].set_index("method").loc[MAIN_METHODS]
        references = null[null.monkey == monkey].set_index("method").loc[MAIN_METHODS]
        fig, ax = plt.subplots(figsize=(5.7, 3.7), layout="constrained")
        x = np.arange(5)
        ax.bar(x, group["mean"], yerr=group.sd, color=[COLORS[k] for k in MAIN_METHODS], capsize=3, width=.67)
        ax.errorbar(x+.28, references["mean"], yerr=np.vstack([references["mean"]-references.low, references.high-references["mean"]]),
                    fmt="_", color="#555555", capsize=2, markersize=7, label="Shuffled labels: 95% null interval")
        for i, row in enumerate(group.itertuples()):
            ax.text(i, row.mean+row.sd+.007, f"{row.mean:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set(xticks=x, xticklabels=[LABELS[k] for k in MAIN_METHODS], ylabel="Macro-F1", ylim=(0, ylim),
               title=f"Monkey {monkey} | {int(group.n_lfps.iloc[0])} LFPs, {int(group.n_days.iloc[0])} days")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        planned = tests[(tests.monkey == monkey) & (tests.family == "standalone")]
        ax.text(.98, .98, "\n".join(f"Geometry vs {'power/frequency' if r.baseline == 'power_frequency' else 'all bands'}: p = {r.p_holm:.3g}" for r in planned.itertuples()),
                transform=ax.transAxes, va="top", ha="right", fontsize=8)
        save_figure(root, f"decoding_{monkey}", fig, group.reset_index(),
            "Short-delay, final 500 ms pre-GO; balanced five-fold single-channel LDA. Bars: recording mean +/- SD. "
            "Null markers: 200 label permutations. K-torus geometry retains all selected lag coordinates, with no PCA. "
            "P-values: paired recording-day Wilcoxon, Holm-adjusted across eight planned comparisons. "
            + ("M peak-power/frequency bands: 12-25 and 25-40 Hz, 2D each and 4D combined. " if monkey == "M" else
               "T peak-power/frequency band: 12-40 Hz, 1D each and 2D combined. ")
            + "All band powers: 5D. Geometry: K*m*(m+1)/2 + m + 1 features, varying by training fold; see embedding_selection.csv.")
        fig, ax = plt.subplots(figsize=(4.5, 3.8), layout="constrained")
        rng = np.random.default_rng(42)
        enhanced = ["ktorus_power_frequency", "ktorus_all_bands"]
        for index, method in enumerate(enhanced):
            points = pairs[(pairs.monkey == monkey) & (pairs.enhanced == method)]
            row = tests[(tests.monkey == monkey) & (tests.enhanced == method)].iloc[0]
            ax.scatter(index+rng.uniform(-.17, .17, len(points)), points.difference, s=12, color="#7f929b", alpha=.55)
            ax.errorbar(index, row.mean_difference, yerr=[[row.mean_difference-row.ci_low], [row.ci_high-row.mean_difference]],
                        fmt="o", color="#ad332e", capsize=5, markersize=6)
            ax.text(index, delta_max*.87, f"p = {row.p_holm:.3g}", ha="center", fontsize=9)
        ax.axhline(0, color="#333333", lw=.8)
        ax.set(xticks=[0, 1], xticklabels=["Add geometry to\npower + frequency", "Add geometry to\nall band powers"],
               ylabel="Change in macro-F1", ylim=(-delta_max, delta_max), title=f"Monkey {monkey}")
        save_figure(root, f"added_value_{monkey}", fig, pairs[(pairs.monkey == monkey) & pairs.enhanced.isin(enhanced)],
            "Each gray point is a within-recording matched difference. Red: recording mean and recording-day cluster-bootstrap 95% CI. "
            "Positive favors adding geometry. P-values are Holm-adjusted across all eight planned tests, using paired day summaries.")
        for method in ("power_frequency", "ktorus"):
            data = confusion[(confusion.monkey == monkey) & (confusion.method == method)]
            matrix = data.pivot(index="true_direction", columns="predicted_direction", values="fraction").to_numpy()
            fig, ax = plt.subplots(figsize=(3.7, 3.45), layout="constrained")
            artist = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if matrix[i,j] > .55 else "black")
            ax.set(xticks=range(6), yticks=range(6), xticklabels=range(1, 7), yticklabels=range(1, 7),
                   xlabel="Predicted direction", ylabel="True direction", title=f"Monkey {monkey}: {LABELS[method].replace(chr(10), ' ')}")
            fig.colorbar(artist, ax=ax, label="Prediction fraction", shrink=.8)
            save_figure(root, f"confusion_{monkey}_{method}", fig, data,
                "Held-out predictions; each recording's matrix is row-normalized before equal-weight averaging. Diagonal entries are class recall, not F1.")
        fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
        selected = direction[(direction.monkey == monkey) & direction.method.isin(["power_frequency", "ktorus"])]
        for index, method in enumerate(["power_frequency", "ktorus"]):
            data = selected[selected.method == method].sort_values("direction")
            ax.bar(np.arange(6)+(index-.5)*.35, data["mean"], yerr=data["std"], width=.34,
                   capsize=2, color=COLORS[method], label=LABELS[method].replace("\n", " "))
        ax.set(xticks=range(6), xticklabels=range(1, 7), xlabel="Reach direction", ylabel="Per-direction F1",
               ylim=(0, .65), title=f"Monkey {monkey}")
        ax.legend(frameon=False, fontsize=8)
        save_figure(root, f"direction_f1_{monkey}", fig, selected, "Mean per-direction F1 +/- SD across matched recordings. Descriptive; no per-direction significance tests.")
        selected = embedding[(embedding.monkey == monkey) & embedding.geometry_evaluable]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
        for ax, column, label in zip(axes, ["K", "dimension", "tau_samples"],
                                     ["Candidate mode count K", "Delay coordinates m", "Delay (ms)"]):
            counts = selected[column].value_counts().sort_index()
            ax.bar(counts.index, counts.values, color="#36697f", width=.75)
            ax.set(xlabel=label, ylabel="Training folds")
        fig.suptitle(f"Monkey {monkey}: full-coordinate embedding selection", fontsize=10)
        save_figure(root, f"embedding_parameters_{monkey}", fig, selected,
            "Selection counts across outer training folds of geometry-evaluable recordings. Folds are not independent biological replicates. "
            "K is PSD-derived, m=2K+1, and delay is the first AMI local minimum passing sample-count and delay-map conditioning checks. No PCA.")


def examples(root, audit, embedding, diagnostics):
    rows = []
    for monkey in ("M", "T"):
        ids = set(audit[audit.geometry_evaluable & (audit.monkey == monkey)].recording)
        candidates = diagnostics[(diagnostics.recording.isin(ids)) & (diagnostics.fold == 0) & ~diagnostics.held_out & diagnostics.success]
        errors = candidates.groupby("recording").normalized_error.median()
        selected = (errors-errors.median()).abs().sort_values(kind="stable").index[0]
        folder = root / "cache" / selected
        verify_example_inputs(folder)
        params = json.loads((folder / "fold_0.json").read_text())
        with np.load(folder / "trials.npz") as saved:
            trials = {k: saved[k] for k in saved.files}
        with np.load(folder / "fold_0.npz") as saved:
            fits = {k: saved[k] for k in saved.files}
        choices = np.flatnonzero((trials["folds"] == 0) & fits["success"])
        index = choices[np.argmin(trials["original_trial_number"][choices])]
        data, segments, good, _ = load_recording(CONVERTED_DIR / f"{selected}.npz")
        raw_id = trials["raw_trial_indices"][index]
        observed, times = trial_cloud(segments[np.flatnonzero(good == raw_id)[0]], params)
        predicted = phase_design(times, fits["frequencies"][index]) @ fits["coefficients"][index]
        frame = pd.DataFrame({"time_ms": times*1000-500})
        for j in range(params["dimension"]):
            frame[f"observed_lag_{j}"] = observed[:, j]
            frame[f"fitted_lag_{j}"] = predicted[:, j]
        save_table(root / "diagnostics" / f"example_{monkey}_coordinates.csv", frame)
        rows.append(dict(monkey=monkey, recording=selected, reference_fold=0, K=params["K"], dimension=params["dimension"],
                         tau_samples=params["tau"], raw_trial_index=int(raw_id), original_trial_number=int(trials["original_trial_number"][index]),
                         direction=int(trials["labels"][index]), training_median_error=errors[selected]))
        m = params["dimension"]
        fig, axes = plt.subplots(m, 1, figsize=(6.4, 1.+.85*m), sharex=True, layout="constrained")
        for j, ax in enumerate(np.atleast_1d(axes)):
            ax.plot(frame.time_ms, observed[:, j], color="#334a58", lw=.8, label="Observed")
            ax.plot(frame.time_ms, predicted[:, j], color="#b53a32", lw=.9, label="K-mode fit")
            ax.set_ylabel(f"Lag {j}", fontsize=8)
        axes[0].set_title(f"Monkey {monkey}: K={params['K']}, m={m}, delay={params['tau']} ms")
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        axes[-1].set_xlabel("Reference time before GO (ms)")
        save_figure(root, f"full_coordinate_example_{monkey}", fig, frame,
            f"Descriptive held-out trial, {selected}. Recording selected by median training-only normalized fit error, not decoding. "
            "Every retained delay coordinate is shown; no PCA. Fits are read from cache, not recomputed. Coordinate j is x(t-j*tau).")
        fig = coordinate_views(observed, predicted, params["tau"],
                               f"Monkey {monkey}: K={params['K']}, m={m}, delay={params['tau']} ms")
        save_figure(root, f"coordinate_views_{monkey}", fig, frame,
            f"Same label-blind held-out example, {selected}. Pairwise displays of the original delay coordinates, "
            "not PCA scores and not a reduced fitting space. The complete m-dimensional cloud is used in fitting. "
            "Lines follow sample time; this short trajectory does not establish full torus phase coverage.")
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout="constrained")
        axes[0].plot(params["psd_frequencies"], params["residual_median"], color="#8096a2", label="Median residual PSD")
        axes[0].plot(params["psd_frequencies"], params["psd_smoothed"], color="#263e4c", label="Smoothed")
        for frequency in params["frequencies"]:
            axes[0].axvline(frequency, color="#b53a32", ls="--", lw=.8)
        axes[0].set(xlabel="Frequency (Hz)", ylabel="Log PSD above fitted background", title=f"Monkey {monkey}: candidate K={params['K']}")
        axes[0].legend(loc="lower left", frameon=False, fontsize=8)
        axes[1].plot(params["ami_lags"], params["ami"], color="#263e4c")
        axes[1].axvline(params["tau"], color="#b53a32", ls="--", lw=.8)
        axes[1].set(xlabel="Delay (ms)", ylabel="Average mutual information", title=f"m={m}; selected delay={params['tau']} ms")
        save_figure(root, f"selection_example_{monkey}", fig, pd.DataFrame([rows[-1]]),
            "Parameters selected using training trials only. Peaks imply candidate modes, not independently proven toroidal topology. Red: selected peaks/delay. Full curves and peak-support values are saved in fold_0.json.")
    save_table(root / "tables" / "example_selection.csv", pd.DataFrame(rows))


def write_report(root, audit, embedding, diagnostics):
    summary = pd.read_csv(root / "tables" / "summary.csv")
    tests = pd.read_csv(root / "tables" / "planned_comparisons.csv")
    lines = ["# PSD-selected K-torus decoding, without PCA", "", "## Scope", "",
             "Single LFP channels, six reach directions, short-delay trials, final 500 ms before GO. "
             "K, m=2K+1 and AMI delay are training-fold-specific. No PCA or PINN. "
             "All original delay coordinates enter an affine product-of-circles fit; this is not the former 15D tube model.",
             "", "## Cohort", "", "| Monkey | Eligible LFPs | Geometry-evaluable LFPs | Recording days |", "|---|---:|---:|---:|"]
    for monkey in ("M", "T"):
        eligible = audit[(audit.monkey == monkey) & (audit.status == "eligible")]
        good = eligible[eligible.geometry_evaluable]
        lines.append(f"| {monkey} | {len(eligible)} | {len(good)} | {good.day.nunique()} |")
    trial_counts = audit[audit.geometry_evaluable].groupby("monkey").n_trials.sum()
    lines += ["", "All 341 source LFP recordings were audited (214 M, 127 T). "
              "Trial-count eligibility required at least 15 valid short-delay trials per direction before balancing. "
              f"The matched geometry cohort contains {int(trial_counts['M']):,} trials in M and {int(trial_counts['T']):,} in T; "
              "these are recording-trial observations, not distinct behavioral trials pooled across simultaneous channels."]
    lines += ["", "## Main results", "", "Matched geometry-evaluable cohort; mean +/- recording SD. Statistical unit is recording day for paired tests, not independent animal.",
              "", "| Representation | Monkey M macro-F1 | Monkey T macro-F1 |", "|---|---:|---:|"]
    for method in METHODS:
        values = []
        for monkey in ("M", "T"):
            row = summary[(summary.monkey == monkey) & (summary.method == method)].iloc[0]
            values.append(f"{row['mean']:.4f} +/- {row.sd:.4f}")
        lines.append(f"| {LABELS[method].replace(chr(10), ' ')} | {' | '.join(values)} |")
    lines += ["", "## Planned comparisons", "", "Mean paired differences with recording-day-clustered 95% CIs. "
              "Two-sided Wilcoxon on paired day means; Holm correction across eight tests.", "",
              "| Monkey | Comparison | Change in F1 [95% CI] | Holm p |", "|---|---|---:|---:|"]
    for row in tests.itertuples():
        lines.append(f"| {row.monkey} | {row.enhanced} - {row.baseline} | {row.mean_difference:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}] | {row.p_holm:.4g} |")
    lines += ["", "## Geometry checks", ""]
    for monkey in ("M", "T"):
        chosen = embedding[(embedding.monkey == monkey) & embedding.geometry_evaluable]
        ids = set(audit[audit.geometry_evaluable & (audit.monkey == monkey)].recording)
        diag = diagnostics[diagnostics.recording.isin(ids) & diagnostics.held_out]
        good = diag[diag.success]
        lines.append(f"- Monkey {monkey}: candidate K counts across included folds {chosen.K.value_counts().sort_index().to_dict()}; "
                     f"m range {chosen.dimension.min()}-{chosen.dimension.max()}; delay median {chosen.tau_samples.median():g} ms; "
                     f"feature count range {chosen.feature_count.min()}-{chosen.feature_count.max()}. "
                     f"Training trials per feature: median {chosen.training_trials_per_feature.median():.2f}, "
                     f"range {chosen.training_trials_per_feature.min():.2f}-{chosen.training_trials_per_feature.max():.2f}. "
                     f"Held-out successful-fit fraction {diag.success.mean():.3f}; median normalized residual {good.normalized_error.median():.3f}; "
                     f"frequency-bound fraction {good.frequency_at_bound.mean():.3f}.")
    excluded = embedding[embedding.status != "ok"]
    lines += [f"- Unresolved fold reasons: {excluded.status.value_counts().to_dict()}.", "",
              "## Interpretation limits", "",
              "- PSD-derived K is a hypothesis about oscillatory modes, not proof of independent phases or topology. Harmonic ambiguities are logged, not silently resolved.",
              "- The fitter assumes approximately constant modal frequencies within each 500-ms trial. Boundary hits and large residuals limit geometric interpretation.",
              "- Geometry features and the older tube features are different representations. The predecessor comparison is exploratory and cannot isolate PCA as the sole cause of any change.",
              "- Feature counts vary with K and m; no dimension-matched PCA control is used. Shrinkage LDA and permutation references do not eliminate feature-capacity confounds.",
              "- This experiment follows inspection of the predecessor. It is not an independent preregistered confirmation or cross-animal transfer experiment.", "",
              "## Verification", "", "All stored held-out predictions were reproduced from frozen features. F1 was independently recomputed from class counts; "
              "confusion rows, unchanged spectral predictions, 200 null permutations, and predecessor output hashes were checked. "
              "See validation.json and the exact source snapshot under logs/source.", "",
              "## Reproduction", "", "Use the Python executable recorded in manifest.json. The source snapshot and config identify this exact run. "
              "The initial primary run used the exact files in logs/source. The documented completion stage used "
              "logs/integrity_source to audit every saved fit and append missing fused-method nulls, without changing "
              "geometry or held-out predictions. Use report_prego_ktorus.py --render-only to regenerate figures from "
              "verified frozen tables/cached fits without training any classifier or geometry model."]
    (root / "RESULTS.md").write_text("\n".join(lines)+"\n")
    (root / "README.md").write_text("# prego_psd_ktorus_nopca_v1\n\nStart with RESULTS.md. plots/ contains individual PDF, SVG, and 600-dpi PNG panels, "
        "each with CSV data and a caption. tables/ contains scores, predictions, selections, diagnostics and tests. "
        "cache/ contains fold-specific full-dimensional fits. manifest.json and config.json identify the run. "
        "The predecessor prego_single_channel is unchanged.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-only", action="store_true",
                        help="Verify frozen tables and render without fitting geometry or classifiers")
    args = parser.parse_args()
    root = args.output.resolve()
    validate_output(root)
    if args.render_only:
        verify_frozen_tables(root)
        audit, embedding, diagnostics = [pd.read_csv(root / "tables" / f"{name}.csv")
            for name in ("recording_audit", "embedding_selection", "fit_diagnostics")]
    else:
        print("Reproducing predictions and freezing tables...", flush=True)
        audit, embedding, diagnostics = freeze(root)
    print("Rendering standalone figures and cached-fit diagnostics...", flush=True)
    render(root)
    examples(root, audit, embedding, diagnostics)
    write_report(root, audit, embedding, diagnostics)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(state="verified", completed_utc=utc(), reporter_sha256=archive_report(root))
    write_json(root / "manifest.json", manifest)
    print((root / "RESULTS.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
