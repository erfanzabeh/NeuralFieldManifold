"""Plot-only inspection of frozen fixed-delay fits; never invokes a fitter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from prego_geometric_fits import (
    OUTPUT, MODELS, NATIVE_METRICS, array_hash, example_rows, recompute_distances, sha256,
    summarize_fit, write_json,
)

COLORS = {"one_torus": "#247887", "two_torus": "#BE583E"}
NAMES = {"one_torus": "1-torus: planar band", "two_torus": "2-torus: donut volume"}
METRICS = ["normalized_3d_set_error", "rms_3d_set_distance", "mean_3d_set_distance"] + [
    f"native_{key}" for key in NATIVE_METRICS]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def summarize(table):
    rows = []
    for model, group in table.groupby("model", sort=False):
        for metric in METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(dict(model=model, metric=metric, n_results=len(group), n_finite=len(finite),
                             n_returned=int((group.status == "returned").sum()),
                             n_converged=int(group.optimizer_success.fillna(False).sum()),
                             n_flagged=int(group["flags"].fillna("").ne("").sum()),
                             n_nonconverged=int(group["flags"].fillna("").str.contains("nonconvergence").sum()),
                             n_collapsed_hole=int(group["flags"].fillna("").str.contains("collapsed_hole").sum()),
                             n_bound_hits=int(group["flags"].fillna("").str.contains("optimizer_bound").sum()),
                             median=finite.median(), q25=finite.quantile(.25), q75=finite.quantile(.75),
                             mean=finite.mean(), sd=finite.std(ddof=1), minimum=finite.min(), maximum=finite.max()))
    return pd.DataFrame(rows)


def model_mesh(fit, model, resolution=49):
    t = np.linspace(0, 2*np.pi, resolution)
    if model == "one_torus":
        fraction = np.linspace(0, 1, 12)[:, None]
        a = fit["R1_in"]+(fit["R1"]-fit["R1_in"])*fraction
        b = fit["R2_in"]+(fit["R2"]-fit["R2_in"])*fraction
        x, y = a*np.cos(t), b*np.sin(t)
        z = np.zeros_like(x)
    else:
        phi, theta = np.meshgrid(t, t)
        denom = np.sqrt((fit["R2"]*np.cos(phi))**2+(fit["R1"]*np.sin(phi))**2)+1e-12
        x = fit["R1"]*np.cos(phi)+fit["minor_radius"]*np.cos(theta)*fit["R2"]*np.cos(phi)/denom
        y = fit["R2"]*np.sin(phi)+fit["minor_radius"]*np.cos(theta)*fit["R1"]*np.sin(phi)/denom
        z = fit["minor_radius"]*np.sin(theta)
    return (np.asarray(fit["center"])+x[..., None]*np.asarray(fit["u_axis"])
            + y[..., None]*np.asarray(fit["v_axis"])+z[..., None]*np.asarray(fit["direction"]))


def read_fits(root, row):
    return {model: json.loads((root/"checkpoints"/f"trial_{row:03d}_{model}.json").read_text()) for model in MODELS}


def display_note(result, model):
    if result["status"] != "returned":
        return "FIT FAILED"
    flags = result["summary"]["flags"]
    notes = []
    for flag, label in (("nonconvergence", "NOT CONVERGED"), ("collapsed_hole", "collapsed hole"),
                        ("invalid_inner_outer_radii", "invalid band radii"),
                        ("optimizer_bound", "at/near bound"),
                        ("nearly_circular_orientation", "near-circular axes")):
        if flag in flags:
            notes.append(label)
    label = "; ".join(notes) or "no fit flags"
    if model == "two_torus":
        label += "\nParametric sweep; not a volume boundary."
    return label


def pair_limits(points, results):
    arrays = [points]
    for model, result in results.items():
        if result["status"] == "returned":
            mesh = model_mesh(result["fit"], model)
            if np.isfinite(mesh).all():
                arrays.append(mesh.reshape(-1, 3))
    everything = np.concatenate(arrays)
    center = (everything.min(axis=0)+everything.max(axis=0))/2
    radius = max(np.ptp(everything, axis=0).max()*.56, .01)
    return [(v-radius, v+radius) for v in center]


def draw_cloud(ax, points, result, model, limits):
    if result["status"] == "returned":
        mesh = model_mesh(result["fit"], model)
        if np.isfinite(mesh).all():
            ax.plot_surface(*np.moveaxis(mesh, -1, 0), color=COLORS[model], alpha=.22,
                            linewidth=0, antialiased=True, rcount=28, ccount=36)
    ax.plot(*points.T, color="#252525", linewidth=.55, alpha=.65)
    ax.scatter(*points.T, color="#141414", s=1.2, alpha=.65, depthshade=False)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x(t)",
           ylabel="x(t - 16 ms)", zlabel="x(t - 32 ms)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(pad=0)
    ax.set_title(NAMES[model], color=COLORS[model], fontweight="bold", pad=0)


def pair_figure(points, results, trial):
    fig = plt.figure(figsize=(10.8, 6.6))
    fig.suptitle(f"Direction {int(trial.direction)} | original trial {int(trial.original_trial_number)}",
                 fontsize=15, fontweight="bold", y=.96)
    fig.text(.5, .905, "T, channel 7 | final 500 ms before GO | m = 3 | tau = 16 ms | 468 points",
             ha="center", fontsize=10)
    limits = pair_limits(points, results)
    for column, model in enumerate(MODELS):
        ax = fig.add_axes([.02+.50*column, .27, .46, .57], projection="3d")
        result = results[model]
        draw_cloud(ax, points, result, model, limits)
        if result["status"] == "returned":
            row = result["summary"]
            flags = (row['flags'] or 'none').replace('_', ' ').replace(';', '; ')
            label = (f"Normalized 3D set error: {row['normalized_3d_set_error']:.4f}\n"
                     f"Native coverage: {row['native_frac_inside']:.3f} | nfev: {row['nfev']}\n"
                     + textwrap.fill(f"Flags: {flags}", width=55))
        else:
            label = "FIT FAILED\n"+textwrap.fill(result['reason'], width=55)
        fig.text(.06+.50*column, .205, label, va="top", fontsize=9, linespacing=1.2)
    fig.text(.5, .018, "Same observed cloud and scale. Band versus solid-tube containment; not a topology test.\n"
             "Tube overlay is a parametric sweep, not a volume boundary; thick-tube sweeps can self-intersect.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    return fig


def example_grid(root, clouds, trials, rows):
    fig = plt.figure(figsize=(11.7, 10.0))
    fig.suptitle("Fixed-delay geometry | matched examples", fontsize=17, fontweight="bold", y=.97)
    fig.text(.5, .935, "Lowest original trial ID in each direction; selected without fitting scores", ha="center")
    for line, row in enumerate(rows):
        trial = trials.iloc[row]
        results = read_fits(root, row)
        limits = pair_limits(clouds[row], results)
        for column, model in enumerate(MODELS):
            ax = fig.add_subplot(3, 2, line*2+column+1, projection="3d")
            draw_cloud(ax, clouds[row], results[model], model, limits)
            note = display_note(results[model], model).split("\n")[0]
            ax.set_title(f"D{int(trial.direction)} / trial {int(trial.original_trial_number)} | {NAMES[model]}\n{note}",
                         fontsize=9, color=COLORS[model], pad=3)
    fig.subplots_adjust(top=.89, bottom=.08, left=.01, right=.97, hspace=.27, wspace=.09)
    fig.text(.5, .022, "Original lag coordinates; no PCA reduction. Both fits use all 468 points.\n"
             "Tube overlays are parametric sweeps, not volume boundaries; they can self-intersect. Not a topology test.",
             ha="center", fontsize=9)
    return fig


def metric_figure(table):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8))
    pivot = table.pivot(index="row_index", columns="model", values="normalized_3d_set_error")
    pairs = pivot.dropna()
    for _, pair in pairs.iterrows():
        axes[0].plot([0, 1], pair[list(MODELS)], color="#b5b5b5", alpha=.3, lw=.65)
    for pos, model in enumerate(MODELS):
        v = table.loc[table.model == model, "normalized_3d_set_error"].dropna().to_numpy()
        axes[0].scatter(np.full(len(v), pos), v, s=9, color=COLORS[model], alpha=.4)
        axes[0].plot([pos-.15, pos+.15], [np.median(v)]*2, color="black", lw=2)
        axes[1].step(np.sort(v), np.arange(1, len(v)+1)/len(v), where="post", color=COLORS[model], label=NAMES[model])
    axes[0].set(xticks=[0, 1], xticklabels=["Planar band", "Donut volume"],
                ylabel="Normalized 3D distance-to-set error", title="Paired trials; black lines = medians")
    axes[0].set_ylim(bottom=0)
    axes[1].set(xlabel="Normalized 3D distance-to-set error", ylabel="Fraction of trials", title="Full distributions")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].set_xlim(left=0)
    axes[2].scatter(pairs.one_torus, pairs.two_torus, s=16, color="#4b4b4b", alpha=.6)
    limit = max(pairs.max().max()*1.08, .01)
    axes[2].plot([0, limit], [0, limit], "--", color="#777777", lw=1)
    axes[2].set(xlim=(0, limit), ylim=(0, limit), xlabel="Planar-band normalized error",
                ylabel="Donut-volume normalized error", title=f"Matched comparison (n = {len(pairs)})")
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle("Geometric fitting errors | all finite returned fits retained", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=.075, right=.985, top=.81, bottom=.25, wspace=.42)
    fig.text(.5, .055, "Different fitted sets have different flexibility. Lower containment error does not prove a 2-torus.\n"
             "These are native approximate geometric distances, not time-trajectory residuals or prediction errors.",
             ha="center", fontsize=9)
    return fig


def native_figure(table):
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 7.4))
    titles = ["Native MSE", "Native mean error", "Native coverage", "Native R-squared", "Native hole quality", "Native hole fraction"]
    for metric, title, ax in zip(NATIVE_METRICS, titles, axes.flat):
        for index, model in enumerate(MODELS):
            vals = table.loc[table.model == model, f"native_{metric}"].dropna().to_numpy()
            offsets = .15*np.sin(np.arange(len(vals))*2.39996)
            ax.scatter(index+offsets, vals, color=COLORS[model], s=5, alpha=.5)
            ax.boxplot([vals], positions=[index], widths=.46, showfliers=False,
                       medianprops=dict(color="black"), boxprops=dict(color=COLORS[model]))
        ax.set(title=title, xticks=[0, 1], xticklabels=["Planar band", "Donut volume"])
        ax.grid(axis="y", color="#dddddd", lw=.5)
    fig.suptitle("Native metrics | definitions differ between fitters", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.075, right=.97, top=.88, bottom=.17, hspace=.42, wspace=.3)
    fig.text(.5, .045, "Band: coverage, mean error, and R-squared ignore out-of-plane offsets; MSE includes them.\n"
             "Donut: errors measure only points outside the solid tube. Neither coverage score establishes topology.", ha="center")
    return fig


def parameter_figure(table):
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.8))
    for ax, key, label in zip(axes, ("R1", "R2", "minor_radius"),
                               ("R1: outer band / donut backbone", "R2: outer band / donut backbone", "Band width / donut tube radius")):
        for i, model in enumerate(MODELS):
            v = table.loc[table.model == model, key].dropna().to_numpy()
            ax.scatter(i+.14*np.sin(np.arange(len(v))*2.39996), v, s=6, color=COLORS[model], alpha=.55)
            ax.boxplot([v], positions=[i], widths=.42, showfliers=False, medianprops=dict(color="black"))
        ax.set(title=label, xticks=[0, 1], xticklabels=["Planar band", "Donut volume"], ylabel="Normalized-signal units")
    fig.suptitle("Fitted geometric parameters | native definitions", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.8, wspace=.4)
    fig.text(.5, .065, "Band minor_radius = average radial half-width, not out-of-plane thickness.\n"
             "Complete centers, axes, inner radii, bounds, and optimizer diagnostics are saved per trial.", ha="center")
    return fig


def export_figure(fig, root, name, caption, pages=None):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(root/"figures"/f"{name}.{suffix}", dpi=600)
    (root/"figures"/f"{name}.caption.txt").write_text(caption+"\n")
    if pages is not None:
        pages.savefig(fig)
    plt.close(fig)


def overview_page(table, summary):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.07, .92, "T, channel 7 | geometric fitting audit", fontsize=20, fontweight="bold")
    fig.text(.07, .86, "198 short-delay trials | 33 per direction | final 500 ms before GO", fontsize=13)
    fig.text(.07, .815, "m = 3 | tau = 16 ms for every trial | 468 points per cloud | no decoding", fontsize=12)
    y = .72
    for model in MODELS:
        s = summary[(summary.model == model)&(summary.metric == "normalized_3d_set_error")].iloc[0]
        fig.text(.07, y, NAMES[model], fontsize=14, fontweight="bold", color=COLORS[model])
        fig.text(.07, y-.045, f"Normalized 3D error: median {s['median']:.4f}  |  IQR {s.q25:.4f} to {s.q75:.4f}", fontsize=12)
        fig.text(.07, y-.085, f"Returned: {s.n_returned}/198  |  Converged: {s.n_converged}/198  |  Any flag: {s.n_flagged}/198", fontsize=11)
        fig.text(.07, y-.12, f"Collapsed-hole flag: {s.n_collapsed_hole}/198  |  Active/near bounds: {s.n_bound_hits}/198", fontsize=11)
        y -= .185
    notes = ["Both fitters use nonlinear least squares with their unchanged native objectives, bounds, and Huber loss.",
             "The 1-torus routine fits a planar annular band. The 2-torus routine fits a solid donut-shaped tube.",
             "The shared normalization divides squared 3D distances by cloud variance. It does not remove differences in model flexibility.",
             "Native coverage and R-squared have different definitions. Numerical convergence or high containment is not proof of torus topology.",
             "The previous 0.80 residual measured a time-parameterized oscillatory trajectory, not these geometric distances."]
    for note in notes:
        text = textwrap.fill(note, 112)
        fig.text(.07, y, text, va="top", fontsize=10, linespacing=1.35)
        y -= .062
    return fig


def verify_export_row(exported, result, trial, points):
    def compare(key, expected):
        actual = exported[key]
        if isinstance(expected, str):
            actual = "" if pd.isna(actual) else str(actual)
            if actual != expected:
                raise ValueError(f"Export mismatch: {key}")
        else:
            np.testing.assert_allclose(actual, np.nan if expected is None else expected,
                                       atol=1e-12, rtol=1e-12, equal_nan=True, err_msg=f"Export mismatch: {key}")
    for key in ("row_index", "original_trial_number", "raw_trial_index", "direction", "heldout_fold_zero_based", "delay"):
        compare(key, trial[key])
        if key in result and result[key] != trial[key]:
            raise ValueError(f"Checkpoint identity mismatch: {key}")
    for key in ("status", "reason", "cloud_sha256", "model"):
        compare(key, result[key])
    if result["cloud_sha256"] != array_hash(points):
        raise ValueError("Changed fitting cloud")
    if result["status"] == "returned":
        recomputed = summarize_fit(points, result["fit"], result["model"])
        for key, expected in recomputed.items():
            compare(key, expected)
            saved = result["summary"][key]
            if isinstance(expected, str):
                if saved != expected:
                    raise ValueError(f"Checkpoint summary mismatch: {key}")
            else:
                np.testing.assert_allclose(np.nan if saved is None else saved, expected,
                                           atol=1e-12, rtol=1e-12, equal_nan=True)
    else:
        compare("optimizer_success", False)
        compare("flags", "fit_failed")


def verify(root, clouds, trials, table):
    if len(table) != 396 or len(clouds) != 198 or clouds.shape != (198, 468, 3):
        raise ValueError("Incomplete experiment")
    if table.duplicated(["row_index", "model"]).any():
        raise ValueError("Duplicate results")
    sources = json.loads((root/"provenance.json").read_text())
    for path, digest in sources.items():
        snapshot = root/"inputs"/Path(path).name
        if snapshot.exists() and sha256(snapshot) != digest:
            raise ValueError(f"Changed frozen input: {snapshot}")
    pd.testing.assert_frame_equal(trials, pd.read_csv(root/"inputs/selected_trials.csv"))
    with np.load(root/"inputs/selected_raw_prego_epochs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(trials.row_index, np.arange(198))
        for column, key in (("original_trial_number", "original_trial_number"), ("direction", "labels"),
                            ("raw_trial_index", "raw_trial_indices"), ("heldout_fold_zero_based", "folds")):
            np.testing.assert_array_equal(trials[column], saved[key])
    checks = []
    for row in trials.row_index:
        for model, result in read_fits(root, int(row)).items():
            recorded = table[(table.row_index == row)&(table.model == model)].iloc[0]
            verify_export_row(recorded, result, trials.iloc[row], clouds[row])
            if result["status"] != "returned":
                checks.append(dict(row_index=row, model=model, status="explicit_failure", max_distance_delta=np.nan))
                continue
            fit = result["fit"]
            distance = recompute_distances(clouds[row], fit, model)
            native = np.asarray(fit["diagnostics"]["distance_3d"])
            np.testing.assert_allclose(distance, native, atol=2e-9, rtol=2e-9)
            norm = np.sum(distance**2)/np.sum((clouds[row]-clouds[row].mean(axis=0))**2)
            np.testing.assert_allclose(norm, recorded.normalized_3d_set_error, atol=2e-9, rtol=2e-9)
            for metric in NATIVE_METRICS:
                np.testing.assert_allclose(fit[metric], recorded[f"native_{metric}"], atol=1e-12)
            checks.append(dict(row_index=row, model=model, status="reproduced", max_distance_delta=float(np.max(np.abs(distance-native)))))
    pd.DataFrame(checks).to_csv(root/"tables/metric_verification.csv", index=False)
    old = json.loads((root/"prior_outputs_manifest.json").read_text())
    changed = [path for path, digest in old.items() if not Path(path).exists() or sha256(path) != digest]
    if changed:
        raise ValueError(f"Previous outputs changed: {changed[:5]}")
    changed_sources = [path for path, digest in sources.items() if sha256(path) != digest]
    if changed_sources:
        raise ValueError(f"Input/fitting code changed: {changed_sources}")
    return dict(n_trials=198, n_model_results=396, points_per_cloud=468, metric_reproduction=True,
                identity_parameter_and_diagnostic_reproduction=True,
                previous_output_files_unchanged=len(old), source_files_unchanged=len(sources),
                no_decoding=True, no_pca_reduction=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-atlas", action="store_true", help="Re-render summary only; does not fit")
    args = parser.parse_args()
    root = args.output
    (root/"figures").mkdir(exist_ok=True)
    table = pd.read_csv(root/"tables/trial_fits.csv")
    table["flags"] = table["flags"].fillna("")
    trials = pd.read_csv(root/"inputs/selected_trials.csv")
    with np.load(root/"clouds.npz", allow_pickle=False) as data:
        clouds = data["clouds"]
    checks = verify(root, clouds, trials, table)
    summary = summarize(table)
    summary.to_csv(root/"tables/fit_summary.csv", index=False)
    table.pivot(index="original_trial_number", columns="model", values=METRICS).to_csv(root/"tables/paired_metrics.csv")
    selected = example_rows(trials)
    trials.iloc[selected].to_csv(root/"tables/example_trials.csv", index=False)
    flags = table.assign(flag=table["flags"].str.split(";")).explode("flag")
    flags[flags.flag != ""].groupby(["model", "flag"]).size().rename("count").to_csv(root/"tables/flag_counts.csv")
    with PdfPages(root/"Fitting_Report.pdf") as pages:
        fig = overview_page(table, summary)
        pages.savefig(fig)
        plt.close(fig)
        for index, rows in enumerate((selected[:3], selected[3:]), 1):
            export_figure(example_grid(root, clouds, trials, rows), root, f"examples_{index}",
                          "Lowest original trial ID per direction. Identical full-coordinate clouds, camera, and pairwise scale.\n"
                          "Collapse/convergence/boundary warnings shown per example. Tube overlays are parametric sweeps, not volume boundaries, and may self-intersect.", pages)
        export_figure(metric_figure(table), root, "fit_error_comparison",
                      "All finite returned fits retained, including flagged and nonconverged fits. Lines pair the same trial.\n"
                      "Normalized 3D error = sum(native approximate 3D set distance squared) / sum(centered cloud coordinates squared).", pages)
        export_figure(native_figure(table), root, "native_metrics", "Native metrics are not interchangeable between the planar-band and solid-tube models.", pages)
        export_figure(parameter_figure(table), root, "geometric_parameters", "Native radii have model-specific definitions; no decoding feature vector is constructed.", pages)
    for row in selected:
        trial = trials.iloc[row]
        export_figure(pair_figure(clouds[row], read_fits(root, row), trial), root,
                      f"direction_{int(trial.direction)}_trial_{int(trial.original_trial_number)}",
                      "Selected by lowest original trial number within direction, independent of model fit quality.\n"
                      "Tube overlay is a parametric sweep, not a volume boundary; it can self-intersect for thick tubes.")
    if not args.skip_atlas:
        with PdfPages(root/"All_198_Trials_Atlas.pdf") as atlas:
            for count, trial in enumerate(trials.itertuples(index=False), 1):
                fig = pair_figure(clouds[trial.row_index], read_fits(root, trial.row_index), trial)
                atlas.savefig(fig)
                plt.close(fig)
                if count % 25 == 0:
                    print(f"Atlas: {count}/198 trial pairs", flush=True)
    shutil.copy2(__file__, root/"source"/Path(__file__).name)
    write_json(root/"report_provenance.json", {str(Path(__file__).resolve()): sha256(__file__)})
    write_json(root/"verification.json", checks)
    figures = sorted((root/"figures").glob("*"))
    write_json(root/"figure_manifest.json", {str(p.relative_to(root)): sha256(p) for p in figures})
    write_readme(root, summary, table, checks)
    print(f"Report saved: {root/'Fitting_Report.pdf'}", flush=True)


def write_readme(root, summary, table, checks):
    lines = ["# T channel 7: fixed-delay geometric fitting", "", "No decoding was run.", "",
             "- Recording: `monkeyT_session-y070316009-12_lfp-7`.",
             "- Exactly 198 saved short-delay trials, 33 per direction; original trial IDs retained.",
             "- Audited 500-ms epochs [4000:4500] at 1000 Hz; all samples strictly before GO.",
             "- Existing per-trial detrend, median/MAD normalization, zero-phase 2-55 Hz filter.",
             "- m=3, tau=16 ms for EVERY trial; 468 points per cloud, no coordinate reduction.",
             "- Original package geometric fitters: nonlinear least squares, Huber loss, max_nfev=6000.",
             "- Native SVD is used for initialization and extents only; all original coordinates enter the objectives.",
             "- Saved PSD peak near 22 Hz is context only. Both candidate shapes are imposed and tested, not inferred from fitting scores.",
             "", "## Read first", "", "[Compact fitting report](Fitting_Report.pdf)", "",
             "[All 198 trial pairs](All_198_Trials_Atlas.pdf)", "", "## Results", "",
             "Model | Returned | Converged | Flagged | Collapsed hole | Median normalized 3D error | IQR",
             "--- | ---: | ---: | ---: | ---: | ---: | ---"]
    for model in MODELS:
        s = summary[(summary.model == model)&(summary.metric == "normalized_3d_set_error")].iloc[0]
        lines.append(f"{NAMES[model]} | {s.n_returned}/198 | {s.n_converged}/198 | {s.n_flagged}/198 | {s.n_collapsed_hole}/198 | {s['median']:.6f} | {s.q25:.6f}-{s.q75:.6f}")
    lines += ["", "All finite returned fits, including flagged/nonconverged fits, contribute to distributions.",
              "Exceptions remain explicit rows with missing numerical metrics. No favorable-result filtering.", "",
              "## Metric definitions and interpretation", "",
              "- 1-torus: a planar annular band, not a thin ellipse. Native coverage and mean_error use in-plane distances only. Native R-squared uses in-plane squared error divided by full-cloud variance. Native MSE includes normal-plane offset.",
              "- 2-torus: containment in a solid elliptical tube. Native errors are zero inside the volume; native R-squared uses outside-only squared distance divided by full-cloud variance. They do not measure distance to the donut surface.",
              "- Shared normalized_3d_set_error = sum(native 3D distances squared) / sum((points - cloud mean) squared). For the band, include out-of-plane offset; for the donut, use outside-only tube distance.",
              "- The native four-step nearest-ellipse calculation is an approximation, preserved unchanged. Shared normalization does not make the geometric model classes equally flexible.",
              "- A lower donut containment error is not proof of K=2. High coverage, high native R-squared, and optimizer success are not evidence of genuine torus topology.",
              "- Tube overlays depict the native parametric normal sweep, not the boundary of the solid fitted volume. Thick-tube sweeps can self-intersect and include internal surfaces; collapsed-hole flags are displayed and retained.",
              "- Do not compare these numbers to the old 0.80 time-trajectory SSE/TSS. The residual definitions and fitted models differ.",
              "- One channel, one session/day: 198 trial fits are not 198 independent recordings or animals.",
              "- Flags: optimizer nonconvergence, active/near parameter bounds (relative tolerance 1e-5), nonfinite/degenerate geometry, collapsed holes, invalid band radii, and nearly circular orientation (axis ratio <1.05). Flags do not cause trial removal.",
              "", "## Saved artifacts", "",
              "- `clouds.npz`: processed epochs, complete original-coordinate clouds, labels, original trial IDs, fixed embedding metadata.",
              "- `checkpoints/`: 396 trial/model JSON files with native fit parameters, optimizer diagnostics, per-point residuals, and exact input-cloud hash.",
              "- `tables/trial_fits.csv`: all results/failures; `fit_summary.csv`: medians/IQR and counts; `paired_metrics.csv`: matched trial values; `flag_counts.csv`: diagnostics.",
              "- `tables/example_trials.csv`: label-blind lowest-ID example selection; `metric_verification.csv`: independent distance/metric reproduction.",
              "- `figures/`: independent editable SVG, vector PDF, 600-dpi PNG and captions. Numerical source for summary figures is in `tables/`; example clouds and meshes reproduce from `clouds.npz` and the corresponding checkpoints.",
              "- `source/`, `provenance.json`, `config.json`, `prior_outputs_manifest.json`: code/input/config tracking and prior-output protection.",
              "", "## Reproduce", "", "Run from the repository root using the neuralmanifold environment:", "", "```bash",
              "python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/run_prego_geometric_fits.py --resume --jobs 4",
              "python notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/report_prego_geometric_fits.py", "```", "",
              "The report entrypoint only reads frozen results; it cannot trigger fitting. Existing output directories require matching configuration and source/input hashes to resume.",
              f"Verified unchanged: {checks['previous_output_files_unchanged']} previous output files.", "",
              "**Stop point: fitting inspection only. No feature matrix, LDA, F1, or behavioral model selection.**"]
    (root/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
