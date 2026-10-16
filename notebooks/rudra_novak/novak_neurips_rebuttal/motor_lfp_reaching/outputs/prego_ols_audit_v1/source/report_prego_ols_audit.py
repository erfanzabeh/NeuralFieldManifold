#!/usr/bin/env python
"""Verify, tabulate and render frozen OLS results. Never fits a model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from run_prego_ols_audit import (BASE, DEFAULT_OUTPUT, PREVIOUS, TRIAL_KEYS,
    sha256, verify_artifacts, write_json, save_table)

METRICS = ["mse", "nmse", "r2", "pearson", "nmse_persistence", "r2_persistence",
           "pearson_persistence", "nmse_improvement", "persistence_skill"]
COLORS = dict(ar="#247a83", embedding="#b34637", persistence="#727272")
CONTEXT = ("Balanced short-delay trials; final 500 ms before GO; single LFP channels. "
           "Offline prediction of detrended, robustly normalized, 2-55 Hz zero-phase-filtered signals. "
           "Origins are epoch samples 200-489; no target reaches GO. "
           "AR: recursive 10-ms prediction with no true-signal refresh. Embedding: direct readout. ")


def mode_and_agreement(values):
    unique, counts = np.unique(values, return_counts=True)
    best = int(unique[np.flatnonzero(counts == counts.max())[0]])
    return best, float(counts.max()/counts.sum())


def recording_scores(metrics):
    columns = [c for c in METRICS if c in metrics]
    keys = ["recording", "monkey", "day", "family", "horizon_ms"]
    # Each held-out trial appears once; do not give small outer folds extra weight.
    groups = metrics.groupby(keys)
    result = groups[columns].agg(lambda x: np.mean(x.to_numpy())).reset_index()
    result["n_trials"] = groups.size().to_numpy()
    return result


def independent_metrics(y, prediction, persistence):
    y, prediction, persistence = [np.asarray(a, dtype=float) for a in (y, prediction, persistence)]
    variance = np.var(y, axis=1)
    mse = np.mean(np.square(prediction-y), axis=1)
    base_mse = np.mean(np.square(persistence-y), axis=1)
    def corr(a, b):
        a = a-a.mean(axis=1, keepdims=True)
        b = b-b.mean(axis=1, keepdims=True)
        return np.mean(a*b, axis=1)/(np.std(a, axis=1)*np.std(b, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        nmse = np.where(variance > 0, mse/variance, np.nan)
        base = np.where(variance > 0, base_mse/variance, np.nan)
        return dict(mse=mse, nmse=nmse, r2=1-nmse, pearson=corr(y, prediction),
                    nmse_persistence=base, r2_persistence=1-base,
                    pearson_persistence=corr(y, persistence), nmse_improvement=base-nmse,
                    persistence_skill=np.where(base > 0, 1-nmse/base, np.nan))


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    for name, digest in manifest["provenance"]["source"].items():
        assert sha256(BASE / name) == digest, f"Analysis source changed: {name}"
        assert sha256(root / "source" / name) == digest
    for category in ("inputs", "previous_outputs"):
        for name, digest in manifest["provenance"][category].items():
            assert sha256(Path(name)) == digest, f"Source/predecessor changed: {name}"
    cohort = pd.read_csv(root / "tables/cohort.csv")
    origins, total_trials, tested_folds, failed = np.arange(200, 490), 0, 0, []
    arithmetic_checks = []
    assert cohort.groupby("monkey").size().to_dict() == {"M": 115, "T": 95}
    for recording in cohort.recording:
        folder = root / "cache" / recording
        if not (folder / "complete.json").exists():
            failure = json.loads((folder / "failure.json").read_text())
            failed.append(failure)
            continue
        completed = json.loads((folder / "complete.json").read_text())
        verify_artifacts(folder, completed["artifacts"])
        with np.load(folder / "trials.npz") as trial_file:
            trials = {key: trial_file[key] for key in trial_file.files}
        with np.load(PREVIOUS / "cache" / recording / "short/features.npz") as old:
            for key in TRIAL_KEYS:
                np.testing.assert_array_equal(trials[key], old[key])
        x, folds = trials["segments"].astype(float), trials["folds"]
        assert x.shape[1] == 500 and np.isfinite(x).all()
        assert completed["end_sample_exclusive"] == completed["go_sample_zero_based"]
        assert completed["end_sample_exclusive"]-completed["start_sample"] == 500
        total_trials += len(x)
        for fold in range(5):
            checkpoint = folder / f"fold_{fold}"
            info = json.loads((checkpoint / "models.json").read_text())
            validation = pd.read_csv(checkpoint / "validation.csv")
            inner = pd.read_csv(checkpoint / "inner_scores.csv")
            scores = pd.read_csv(checkpoint / "metrics.csv")
            expected_test = np.flatnonzero(folds == fold)
            outer_train = set(np.flatnonzero(folds != fold))
            validation_indices = []
            for split in info["inner_splits"]:
                a, b = set(split["train"]), set(split["validation"])
                assert a.isdisjoint(b) and a | b == outer_train
                validation_indices.extend(split["validation"])
            assert sorted(validation_indices) == sorted(outer_train)
            for row in validation.itertuples():
                part = inner[(inner.family == row.family) & (inner.p == row.p) & (inner.m == row.m)
                             & (inner.tau == row.tau) & (inner.horizon_ms == row.horizon_ms)]
                assert len(part) == 3
                np.testing.assert_allclose(row.mean_nmse, part.nmse.mean(), rtol=1e-7, atol=1e-14)
                np.testing.assert_allclose(row.se_nmse, part.nmse.std(ddof=1)/np.sqrt(3), rtol=1e-6, atol=1e-14)
            with np.load(checkpoint / "predictions.npz") as saved:
                np.testing.assert_array_equal(saved["test_indices"], expected_test)
                np.testing.assert_allclose(saved["truth"], x[expected_test][:, origins[:, None]+[1, 10]], rtol=0, atol=0)
                np.testing.assert_allclose(saved["persistence"], x[expected_test][:, origins], rtol=0, atol=0)
                for family in ("ar", "embedding"):
                    choice = info["selection"][family]
                    table = validation[(validation.family == family) & (validation.horizon_ms == 10)
                                       & validation.valid].sort_values(["mean_nmse", "p", "m", "tau"])
                    if table.empty:
                        assert choice["status"] == "no_valid_candidate"
                        continue
                    threshold = table.iloc[0].mean_nmse+table.iloc[0].se_nmse
                    near = table[table.mean_nmse <= threshold+1e-14]
                    selected = near.sort_values(["p" if family == "ar" else "m", "mean_nmse", "tau"]).iloc[0]
                    assert (choice["p"], choice["m"], choice["tau"]) == (selected.p, selected.m, selected.tau)
                    c = np.array(info["models"][family]["coefficients"])
                    delays = np.arange(choice["p"]) if family == "ar" else np.arange(choice["m"])*choice["tau"]
                    assert delays.max() <= 200
                    history = x[expected_test][:, origins[:, None]-delays]
                    training = x[folds != fold][:, origins[:, None]-delays].reshape(-1, len(delays))
                    np.testing.assert_allclose(training.mean(axis=0), info["models"][family]["column_mean"], atol=1e-12)
                    if family == "ar":
                        # Extended-precision sequential recurrence independently checks the
                        # saved float64 affine-composition implementation under cancellation.
                        history, c = history.astype(np.longdouble), c.astype(np.longdouble)
                        outputs = []
                        for step in range(1, 11):
                            new = c[0, 0]+np.sum(history*c[1:, 0], axis=2)
                            history = np.concatenate((new[..., None], history[..., :-1]), axis=2)
                            if step in (1, 10):
                                outputs.append(new)
                        reconstructed = np.stack(outputs, axis=-1)
                    else:
                        reconstructed = c[0]+history@c[1:]
                    difference = np.asarray(saved[family]-reconstructed, dtype=float)
                    arithmetic_checks.append(dict(recording=recording, outer_fold=fold, family=family,
                        max_absolute_difference=float(np.max(np.abs(difference))),
                        rms_difference=float(np.sqrt(np.mean(difference**2)))))
                    # Fixed absolute limit in robust-normalized signal units, not a
                    # percentage tolerance that would disguise errors near zero.
                    np.testing.assert_allclose(saved[family], reconstructed, atol=1e-6, rtol=0)
                    for j, h in enumerate((1, 10)):
                        calculated = independent_metrics(saved["truth"][:, :, j], saved[family][:, :, j], saved["persistence"])
                        part = scores[(scores.family == family) & (scores.horizon_ms == h)].sort_values("trial_index")
                        np.testing.assert_array_equal(part.trial_index, expected_test)
                        for key, values in calculated.items():
                            np.testing.assert_allclose(part[key], values, rtol=1e-7, atol=1e-12, equal_nan=True)
            tested_folds += 1
        if tested_folds % 100 == 0:
            print(f"Verified {tested_folds} outer folds", flush=True)
    result = dict(recordings=len(cohort), successful_recordings=len(cohort)-len(failed), failures=failed,
                  trials=total_trials, verified_outer_folds=tested_folds,
                  previous_outputs_unchanged=True, input_and_source_hashes_verified=True,
                  saved_trials_and_folds_identical=True, prediction_and_metric_reconstruction=True,
                  inner_selection_reproduced=True, all_targets_before_GO=True,
                  model_scope="OLS only; no torus, shape features, PCA, PINN, or decoding",
                  independent_forecast_absolute_tolerance=1e-6,
                  max_independent_forecast_difference=max(r["max_absolute_difference"] for r in arithmetic_checks),
                  status="passed" if not failed else "passed_with_explicit_recording_failures")
    save_table(root / "tables/numerical_reconstruction_checks.csv", pd.DataFrame(arithmetic_checks))
    write_json(root / "verification.json", result)
    return result


def collect(root):
    validations, inner_scores, trial_scores, choices, coefficients, poles, records = [], [], [], [], [], [], []
    for recording in pd.read_csv(root / "tables/cohort.csv").recording:
        folder = root / "cache" / recording
        if not (folder / "complete.json").exists():
            records.append(json.loads((folder / "failure.json").read_text()))
            continue
        meta = json.loads((folder / "metadata.json").read_text())
        records.append(meta)
        with np.load(folder / "trials.npz") as saved:
            trial_ids = saved["raw_trial_indices"]
            original_ids = saved["original_trial_number"]
        for fold in range(5):
            checkpoint = folder / f"fold_{fold}"
            info = json.loads((checkpoint / "models.json").read_text())
            for name, destination in (("validation", validations), ("inner_scores", inner_scores), ("metrics", trial_scores)):
                frame = pd.read_csv(checkpoint / (name+".csv"))
                frame["monkey"], frame["day"] = meta["monkey"], meta["day"]
                if name == "metrics":
                    frame["raw_trial_index"] = trial_ids[frame.trial_index]
                    frame["original_trial_number"] = original_ids[frame.trial_index]
                destination.append(frame)
            keys = dict(recording=recording, monkey=meta["monkey"], day=meta["day"], outer_fold=fold)
            for family, selection in info["selection"].items():
                model = info["models"].get(family, {})
                choices.append(dict(**keys, family=family, **selection,
                    rank=model.get("rank"), condition=model.get("condition"),
                    condition_original=model.get("condition_original"), fit_status=model.get("status")))
                if not model:
                    continue
                c = np.array(model["coefficients"])
                for col in range(c.shape[1]):
                    for j, value in enumerate(c[:, col]):
                        coefficients.append(dict(**keys, family=family, horizon_ms=(1, 10)[col],
                            coefficient_index=j, lag_ms=-1 if j == 0 else (j-1)*(1 if family == "ar" else selection["tau"]),
                            value=value, kind="intercept" if j == 0 else "lag_weight"))
            for j, pole in enumerate(info["poles"]):
                poles.append(dict(**keys, pole_index=j, **pole))
    tables = dict(validation_scores=pd.concat(validations, ignore_index=True),
                  inner_validation_scores=pd.concat(inner_scores, ignore_index=True),
                  heldout_trial_scores=pd.concat(trial_scores, ignore_index=True),
                  selected_models=pd.DataFrame(choices), coefficients=pd.DataFrame(coefficients),
                  selected_ar_poles=pd.DataFrame(poles), recording_audit=pd.DataFrame(records))
    tables["heldout_recording_scores"] = recording_scores(tables["heldout_trial_scores"])
    val = tables["validation_scores"]
    keys = ["recording", "monkey", "family", "p", "m", "tau", "horizon_ms"]
    tables["recording_validation_scores"] = val.groupby(keys).agg(
        mean_nmse=("mean_nmse", "mean"), outer_fold_sd=("mean_nmse", "std"),
        mean_persistence=("mean_persistence", "mean"), valid_folds=("valid", "sum"),
        max_condition=("max_condition", "max")).reset_index()
    tables["candidate_failures"] = val[~val.valid].copy()
    summaries = []
    for (recording, family), group in tables["selected_models"].groupby(["recording", "family"]):
        good = group[group.status == "ok"]
        row = dict(recording=recording, family=family, monkey=group.iloc[0].monkey, n_valid_folds=len(good))
        if len(good):
            for key in ("p", "m", "tau"):
                row["modal_"+key], row[key+"_agreement"] = mode_and_agreement(good[key])
            row["boundary_folds"] = int(good.boundary.sum())
        summaries.append(row)
    tables["recording_selections"] = pd.DataFrame(summaries)
    keys = ["monkey", "family", "horizon_ms"]
    pop = []
    for group_key, group in tables["heldout_recording_scores"].groupby(keys):
        for metric in METRICS:
            values = group[metric].to_numpy()
            pop.append(dict(zip(keys, group_key), metric=metric, n_recordings=len(group),
                            n_days=group.day.nunique(), mean=float(np.mean(values)),
                            sd=float(np.std(values, ddof=1)), median=float(np.median(values)),
                            n_undefined=int((~np.isfinite(values)).sum())))
    tables["population_scores"] = pd.DataFrame(pop)
    for name, frame in tables.items():
        save_table(root / "tables" / (name+".csv"), frame)
    # Every inner AR fit is retained too, not only the chosen outer models.
    inner = tables["inner_validation_scores"]
    sweep_poles = []
    for row in inner[(inner.family == "ar") & (inner.horizon_ms == 1)].itertuples():
        c = np.array(json.loads(row.coefficients))[:, 0]
        for j, pole in enumerate(np.roots(np.r_[1., -c[1:]])):
            sweep_poles.append(dict(recording=row.recording, outer_fold=row.outer_fold,
                inner_fold=row.inner_fold, p=row.p, pole_index=j, real=pole.real,
                imaginary=pole.imag, magnitude=abs(pole), frequency_hz=np.angle(pole)*1000/(2*np.pi)))
    save_table(root / "tables/inner_ar_poles.csv", pd.DataFrame(sweep_poles))
    return tables


def style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white"})


def export(fig, path, data, caption):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix("."+suffix), dpi=300, bbox_inches="tight", pad_inches=.10)
    data.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".caption.txt").write_text(CONTEXT+caption+"\n")
    plt.close(fig)


def order_plot(frame, path, title, population=False):
    frame = frame[(frame.family == "ar")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    for ax, horizon in zip(axes, (1, 10)):
        sub = frame[frame.horizon_ms == horizon]
        grouped = sub.groupby("p").agg(mean=("mean_nmse", "mean"), sd=("mean_nmse", "std"),
            persistence=("mean_persistence", "mean")).reset_index()
        if not population:
            grouped["sd"] = sub.groupby("p").outer_fold_sd.first().to_numpy()
        ax.plot(grouped.p, grouped["mean"], color=COLORS["ar"], marker=".", label="AR")
        lower = (grouped["mean"]-grouped.sd.fillna(0)).where(lambda values: values > 0, np.nan)
        ax.fill_between(grouped.p, lower, grouped["mean"]+grouped.sd.fillna(0), color=COLORS["ar"], alpha=.15)
        ax.plot(grouped.p, grouped.persistence, "--", color=COLORS["persistence"], label="Persistence")
        ax.set(xlabel="AR order p", ylabel="Validation NMSE (log scale)", yscale="log",
               title="1-ms one-step" if horizon == 1 else "10-ms recursive", xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.15)
    axes[0].legend()
    fig.suptitle(title)
    export(fig, path, frame, "Order sweep. Shading is SD across " +
           ("recording-level means" if population else "five outer-training validation summaries") +
           ". SD bands crossing zero are omitted on the log axis. Equal trial weighting within inner folds; no test data select p. "
           "All candidate scores are retained, including numerically ineligible ones.")


def embedding_plot(frame, path, title, config):
    frame = frame[frame.family == "embedding"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), layout="constrained")
    for ax, horizon in zip(axes, (1, 10)):
        sub = frame[frame.horizon_ms == horizon]
        pivot = sub.groupby(["m", "tau"]).mean_nmse.mean().unstack().reindex(
            index=config["dimensions"], columns=config["delays"])
        finite = pivot.to_numpy()[np.isfinite(pivot.to_numpy()) & (pivot.to_numpy() > 0)]
        vmin, vmax = finite.min(), finite.max()
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d8d8d8")
        im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy()), origin="lower", aspect="auto", cmap=cmap,
                       norm=LogNorm(vmin=max(vmin, 1e-16), vmax=max(vmax, vmin*1.01)))
        ax.set(xticks=np.arange(len(config["delays"])), xticklabels=config["delays"],
               yticks=np.arange(len(config["dimensions"])), yticklabels=config["dimensions"],
               xlabel="Delay tau (ms)", ylabel="Lag dimension m", title=f"{horizon}-ms direct prediction")
        ax.tick_params(axis="x", rotation=60)
        fig.colorbar(im, ax=ax, shrink=.85, label="Validation NMSE (log)")
    fig.suptitle(title)
    export(fig, path, frame, "Complete dimension-by-delay sweep. Gray cells are unavailable (span >200 ms). "
           "Each map uses its full, explicitly labeled log color range; color ranges differ between maps. "
           "These are predictive errors, not torus errors. Each recording contributes equally to population means.")


def distribution_plot(selected, summaries, root):
    data = selected[selected.status == "ok"].copy()
    config = json.loads((root / "config.json").read_text())
    grids = dict(p=config["orders"], m=config["dimensions"], tau=config["delays"])
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.9), layout="constrained", sharey="col")
    for i, monkey in enumerate(("M", "T")):
        for j, (family, key, label) in enumerate((("ar", "p", "Selected AR order p"),
                ("embedding", "m", "Selected lag dimension m"), ("embedding", "tau", "Selected delay tau (ms)"))):
            sub = data[(data.monkey == monkey) & (data.family == family)]
            counts = sub[key].astype(int).value_counts().reindex(grids[key], fill_value=0)
            positions = np.arange(len(counts))
            axes[i, j].bar(positions, counts, color=COLORS[family], width=.75)
            axes[i, j].set(xlabel=label, ylabel="Outer-fold selections", title=f"Monkey {monkey}: {sub.recording.nunique()} LFPs")
            ticks = [0, 4, 9, 14, 19] if key == "p" else positions
            axes[i, j].set_xticks(ticks, [counts.index[t] for t in ticks])
            if key == "tau":
                axes[i, j].tick_params(axis="x", rotation=60)
    export(fig, root / "plots/summary/selected_parameters", data,
           "Five training-only selections per recording. Fold counts are repeated choices, not independent recording/animal counts. "
           "A boundary selection does not establish an optimum beyond the grid.")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), layout="constrained")
    modes = summaries[summaries.family == "ar"]
    for ax, monkey in zip(axes, ("M", "T")):
        sub = modes[modes.monkey == monkey]
        counts = sub.modal_p.value_counts().sort_index()
        ax.bar(counts.index, counts, color=COLORS["ar"])
        ax.set(xlabel="Modal AR order p", ylabel="LFP recordings", title=f"Monkey {monkey}", xticks=counts.index)
    export(fig, root / "plots/summary/modal_ar_order", modes,
           "Recording-level modal p; ties favor smaller p. This descriptive summary never replaces fold-specific selected models.")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), layout="constrained")
    rows = []
    for ax, (family, key) in zip(axes, (("ar", "p"), ("embedding", "m"), ("embedding", "tau"))):
        for offset, monkey in zip((-.035, .035), ("M", "T")):
            part = summaries[(summaries.family == family) & (summaries.monkey == monkey)]
            counts = part[key+"_agreement"].value_counts().reindex([.2, .4, .6, .8, 1.], fill_value=0)
            ax.bar(counts.index+offset, counts, width=.065, label=monkey,
                   color=COLORS["ar"] if monkey == "M" else COLORS["embedding"])
            rows.extend(dict(monkey=monkey, parameter=key, agreement=float(a), recordings=int(n)) for a, n in counts.items())
        ax.set(xlabel="Fraction of folds agreeing", ylabel="LFP recordings", title=key,
               xticks=[.2, .4, .6, .8, 1.])
    axes[0].legend(title="Monkey")
    export(fig, root / "plots/summary/selection_agreement", pd.DataFrame(rows),
           "Within-recording selection agreement: fraction of the five folds selecting the modal parameter.")


def heldout_plot(scores, selected, root):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), layout="constrained", sharey=True)
    plot_rows = []
    for ax, monkey in zip(axes, ("M", "T")):
        sub = scores[(scores.monkey == monkey) & (scores.horizon_ms == 10)]
        for j, (family, label) in enumerate((("ar", "AR recursive"), ("embedding", "Lag readout"), ("persistence", "Persistence"))):
            values = sub[sub.family == (family if family != "persistence" else "ar")]["nmse" if family != "persistence" else "nmse_persistence"]
            jitter = np.linspace(-.10, .10, len(values))
            ax.scatter(j+jitter, values, color=COLORS[family], s=7, alpha=.35, rasterized=False)
            ax.errorbar(j, values.mean(), yerr=values.std(ddof=1), color="black", marker="_", markersize=15, capsize=4)
            plot_rows.append(dict(monkey=monkey, method=family, mean=values.mean(), sd=values.std(ddof=1), n=len(values)))
        ax.set(xticks=[0, 1, 2], xticklabels=["AR\nrecursive", "Lag\nreadout", "Persistence"],
               ylabel="Held-out 10-ms NMSE (log)", title=f"Monkey {monkey}", yscale="log")
        ax.grid(axis="y", alpha=.15)
    export(fig, root / "plots/summary/heldout_prediction", pd.DataFrame(plot_rows),
           "Each point is one recording's equal-trial held-out NMSE, on a shared log axis. Black markers show recording-level mean +/- SD, not SEM. "
           "Every selected configuration is evaluated only on its saved outer test fold.")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3), layout="constrained")
    for ax, family in zip(axes, ("ar", "embedding")):
        for monkey, color in (("M", COLORS["ar"]), ("T", COLORS["embedding"])):
            sub = selected[(selected.family == family) & (selected.monkey == monkey)]
            ax.hist(np.log10(sub.condition), bins=20, histtype="step", linewidth=1.5, label=monkey, color=color)
        ax.set(xlabel="log10(condition number)", ylabel="Outer-fold models", title="AR" if family == "ar" else "Lag readout")
    axes[0].legend(title="Monkey")
    export(fig, root / "plots/summary/selected_conditioning", selected,
           "Condition numbers of the training-column-standardized design including intercept. "
           "Large values indicate collinearity/sensitivity even when numerical rank is full; no regularization was substituted.")


def example_recording(selected, monkey):
    choices = selected[(selected.monkey == monkey) & (selected.family == "ar")
                       & (selected.status == "ok") & (selected.outer_fold == 0)]
    scores = choices.set_index("recording").mean_nmse
    return (scores-scores.median()).abs().sort_index().sort_values(kind="stable").index[0]


def examples(root, selected):
    config = json.loads((root / "config.json").read_text())
    origins = np.arange(200, 490)
    example_rows = []
    for monkey in ("M", "T"):
        recording = example_recording(selected, monkey)
        folder = root / "cache" / recording
        info = json.loads((folder / "fold_0/models.json").read_text())
        with np.load(folder / "trials.npz") as saved:
            trials = {k: saved[k] for k in saved.files}
        with np.load(folder / "fold_0/predictions.npz") as saved:
            predictions = {k: saved[k] for k in saved.files}
        ids = predictions["test_indices"]
        index = int(np.argmin(trials["original_trial_number"][ids]))
        trial = int(ids[index])
        original = int(trials["original_trial_number"][trial])
        ar = info["selection"]["ar"]
        embedding = info["selection"]["embedding"]
        example_rows.append(dict(monkey=monkey, recording=recording, outer_fold=0, trial_index=trial,
            original_trial_number=original, p=ar["p"], m=embedding["m"], tau_ms=embedding["tau"],
            selection_rule="fold-0 inner-validation AR NMSE nearest animal median; lowest fold-0 test original trial number"))
        title = f"Monkey {monkey} | p={ar['p']}, m={embedding['m']}, tau={embedding['tau']} ms"
        rows = []
        fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.1), layout="constrained")
        for ax, j, h in zip(axes, (0, 1), (1, 10)):
            time_ms = origins+h-500
            truth = predictions["truth"][index, :, j]
            ax.plot(time_ms, truth, color="#202020", lw=1.5, label="Observed")
            for family, label in (("ar", "AR"), ("embedding", "Lag readout")):
                pred = predictions[family][index, :, j]
                ax.plot(time_ms, pred, color=COLORS[family], lw=.9, label=label)
                rows.extend(dict(horizon_ms=h, time_to_GO_ms=int(t), observed=y, predicted=v, family=family,
                                 residual=y-v) for t, y, v in zip(time_ms, truth, pred))
            ax.set(xlabel="Target time relative to GO (ms)", ylabel="Preprocessed signal", title=f"{h}-ms horizon")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="lower center",
                   bbox_to_anchor=(.5, -.045), frameon=False)
        fig.suptitle(title)
        export(fig, root / f"plots/examples/monkey_{monkey}_predictions", pd.DataFrame(rows),
               f"Illustrative recording {recording}, original trial {original}, outer fold 0. "
               "Recording chosen by distance to median fold-0 inner-validation AR error, without behavior labels or any fold-0 test scores; "
               "trial chosen by original trial number. Curves are rolling-origin forecasts, not one 500-ms free-running rollout.")
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
        residual_data = pd.DataFrame(rows)
        for family, label in (("ar", "AR"), ("embedding", "Lag readout")):
            sub = residual_data[(residual_data.family == family) & (residual_data.horizon_ms == 10)]
            axes[0].plot(sub.time_to_GO_ms, sub.residual, color=COLORS[family], label=label)
            axes[1].hist(sub.residual, bins=25, histtype="step", density=True, color=COLORS[family], label=label)
        axes[0].axhline(0, color="gray", lw=.6)
        axes[0].set(xlabel="Target time relative to GO (ms)", ylabel="Observed - predicted")
        axes[1].set(xlabel="10-ms residual", ylabel="Density")
        axes[1].legend()
        fig.suptitle(title)
        export(fig, root / f"plots/examples/monkey_{monkey}_residuals", residual_data,
               "Residuals from the same preselected example; serially correlated samples are not independent observations.")
        m, tau = embedding["m"], embedding["tau"]
        x = trials["segments"][trial]
        coords = x[origins[:, None]-np.arange(m)*tau]
        subset = sorted(set([0, m//2, m-1]))
        fig = plt.figure(figsize=(7.1, 3.7))
        fig.text(.045, .91, f"Monkey {monkey} | m={m}, tau={tau} ms", fontsize=10)
        grid = fig.add_gridspec(1, 2, left=.025, right=.985, bottom=.18, top=.72, wspace=.48)
        ax = fig.add_subplot(grid[0], projection="3d" if len(subset) == 3 else None)
        if len(subset) == 3:
            ax.plot(*coords[:, subset].T, color=COLORS["embedding"], lw=.8)
            ax.set_zlabel(f"x(t-{subset[2]*tau} ms)", labelpad=1, fontsize=8)
            ax.tick_params(labelsize=8)
        else:
            ax.plot(*coords[:, subset].T, color=COLORS["embedding"], lw=.8)
        ax.set_xlabel("x(t)", labelpad=1, fontsize=8)
        ax.set_ylabel(f"x(t-{subset[1]*tau} ms)", labelpad=1, fontsize=8)
        ax.set_title("Coordinate subset (no PCA)", fontsize=9)
        other = fig.add_subplot(grid[1])
        for j in range(m):
            other.plot(origins-500, coords[:, j], linewidth=.7, label=f"{j*tau}")
        other.set(xlabel="Origin time relative to GO (ms)", ylabel="Lag-coordinate value", title=f"All {m} coordinates")
        other.legend(title="Lag (ms)", ncol=5, fontsize=8, title_fontsize=8,
                     loc="upper center", bbox_to_anchor=(.5, 1.42), frameon=False,
                     columnspacing=.7, handlelength=1.3)
        coordinate_table = pd.DataFrame(coords, columns=[f"x_lag_{j*tau}_ms" for j in range(m)])
        coordinate_table.insert(0, "origin_ms", origins-500)
        export(fig, root / f"plots/examples/monkey_{monkey}_coordinates", coordinate_table,
               f"Same example recording/trial. Selected m={m}, tau={tau} ms from outer fold 0 training only. "
               f"Left displays coordinate indices {subset} (zero-based), not a dimensionality reduction. "
               "The full-coordinate time traces are at right. No torus surface is fitted.")
    save_table(root / "tables/examples.csv", pd.DataFrame(example_rows))


def render(root, include_recordings):
    style()
    config = json.loads((root / "config.json").read_text())
    val = pd.read_csv(root / "tables/recording_validation_scores.csv")
    selected = pd.read_csv(root / "tables/selected_models.csv")
    summaries = pd.read_csv(root / "tables/recording_selections.csv")
    scores = pd.read_csv(root / "tables/heldout_recording_scores.csv")
    for monkey, sub in val.groupby("monkey"):
        order_plot(sub, root / f"plots/summary/monkey_{monkey}_order_sweep", f"Monkey {monkey}: {sub.recording.nunique()} LFP recordings", True)
        embedding_plot(sub, root / f"plots/summary/monkey_{monkey}_embedding_sweep", f"Monkey {monkey}: equal-recording averages", config)
    distribution_plot(selected, summaries, root)
    heldout_plot(scores, selected, root)
    examples(root, selected)
    if include_recordings:
        for i, (recording, sub) in enumerate(val.groupby("recording")):
            title = recording.replace("monkey", "Monkey ").replace("_session-", " | ").replace("_lfp-", " | LFP ")
            order_plot(sub, root / "plots/recordings" / recording / "order_sweep", title)
            embedding_plot(sub, root / "plots/recordings" / recording / "embedding_sweep", title, config)
            if (i+1) % 20 == 0:
                print(f"Rendered {i+1} recording pairs", flush=True)
    plots = sorted((root / "plots").rglob("*.pdf"))
    write_json(root / "figure_manifest.json", dict(figures=len(plots), formats=["pdf", "svg", "png"],
        png_dpi=300, plot_source_sha256=sha256(Path(__file__)),
        input_tables={str(p.relative_to(root)): sha256(p) for p in (root / "tables").glob("*.csv")},
        artifacts={str(p.relative_to(root)): sha256(p) for p in (root / "plots").rglob("*") if p.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=["verify", "tables", "plots", "all"], default="all")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage in ("verify", "all"):
        print(verify(root), flush=True)
    if args.stage in ("tables", "all"):
        collect(root)
        print("Frozen tables complete", flush=True)
    if args.stage in ("plots", "all"):
        render(root, not args.summary_only)
        print("Standalone figures complete", flush=True)


if __name__ == "__main__":
    main()
