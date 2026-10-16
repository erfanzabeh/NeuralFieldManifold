"""Fixed K=1, m=3 visual delay sweep using frozen preprocessed example epochs."""
from __future__ import annotations

import argparse
import importlib
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

from motor_lfp_utils import lag_embed
from prego_geometric_fits import (
    OUTPUT as SOURCE, RECORDING, UNIT, array_hash, example_rows, fit_one,
    recompute_distances, sha256, summarize_fit, write_json,
)
from report_prego_geometric_fits import model_mesh, display_note

OUTPUT = UNIT / "outputs/prego_T_y070316009_12_ch7_K1_m3_tau1_20_examples_v1"
TAUS = tuple(range(1, 21))
COLOR = "#247887"


def make_sweep(processed):
    x = np.asarray(processed)
    if x.shape != (500,) or not np.isfinite(x).all():
        raise ValueError("Expected one finite, already-preprocessed 500-sample epoch")
    return {tau: lag_embed(x, dim=3, tau=tau) for tau in TAUS}


def sweep_limits(clouds, results):
    arrays = list(clouds.values())
    for result in results.values():
        if result["status"] == "returned":
            mesh = model_mesh(result["fit"], "one_torus")
            if np.isfinite(mesh).all():
                arrays.append(mesh.reshape(-1, 3))
    points = np.concatenate(arrays)
    center = (points.min(axis=0)+points.max(axis=0))/2
    radius = max(np.ptp(points, axis=0).max()*.56, .01)
    return [(float(v-radius), float(v+radius)) for v in center]


def draw(ax, points, result, tau, limits, overlay=False, compact=False):
    if overlay and result["status"] == "returned":
        mesh = model_mesh(result["fit"], "one_torus")
        ax.plot_surface(*np.moveaxis(mesh, -1, 0), color=COLOR, alpha=.3,
                        linewidth=0, rcount=12, ccount=49)
    ax.plot(*points.T, color="#222222", lw=.5 if compact else .7, alpha=.8)
    ax.scatter(*points.T, c="#222222", s=.8 if compact else 2, alpha=.6, depthshade=False)
    labels = ("x(t)", "x(t - tau)", "x(t - 2 tau)") if compact else (
        "x(t)", f"x(t - {tau} ms)", f"x(t - {2*tau} ms)")
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2],
           xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.set_major_locator(plt.MaxNLocator(3))
        axis.label.set_size(8 if compact else 10)
    ax.tick_params(pad=0, labelsize=8)


def grid_figure(clouds, results, trial, limits, overlay=False):
    fig = plt.figure(figsize=(17.5, 14.5))
    kind = "1-torus overlays" if overlay else "Observed lag embeddings"
    fig.suptitle(f"{kind} | direction {int(trial.direction)}, original trial {int(trial.original_trial_number)}",
                 fontsize=20, fontweight="bold", y=.977)
    fig.text(.5, .949, "T, channel 7 | same preprocessed 500-ms pre-GO epoch | fixed K = 1, m = 3",
             ha="center", fontsize=12)
    for tau in TAUS:
        ax = fig.add_subplot(4, 5, tau, projection="3d")
        draw(ax, clouds[tau], results[tau], tau, limits, overlay, compact=True)
        if overlay:
            result = results[tau]
            note = display_note(result, "one_torus")
            error = result.get("summary", {}).get("normalized_3d_set_error", np.nan)
            title = f"tau = {tau} ms | error = {error:.3f}\n{note}"
        else:
            title = f"tau = {tau} ms | {len(clouds[tau])} points"
        ax.set_title(title, fontsize=9, color=COLOR if overlay else "#222222", pad=9)
    fig.subplots_adjust(left=.02, right=.96, top=.905, bottom=.085, wspace=.19, hspace=.32)
    note = ("Teal = fitted planar annular band; black = observed trajectory. Error includes out-of-plane distance."
            if overlay else "Black = observed trajectory; no fitted surface. Time samples are linked in their original order.")
    fig.text(.5, .032, note+"\nIdentical camera and axis limits across all 20 delays. No PCA, new filtering, or decoding.",
             ha="center", fontsize=10, linespacing=1.5)
    return fig


def detail_figure(points, result, trial, tau, limits):
    fig = plt.figure(figsize=(11, 7))
    fig.suptitle(f"Direction {int(trial.direction)} | original trial {int(trial.original_trial_number)} | tau = {tau} ms",
                 fontsize=16, fontweight="bold", y=.97)
    fig.text(.5, .913, f"T, channel 7 | K = 1, m = 3 | {len(points)} points | same preprocessed 500-ms pre-GO epoch",
             ha="center", fontsize=10)
    for column, overlay in enumerate((False, True)):
        ax = fig.add_axes([.02+column*.50, .27, .45, .55], projection="3d")
        draw(ax, points, result, tau, limits, overlay)
        ax.set_title("Observed cloud" if not overlay else "1-torus: planar annular band",
                     color=COLOR if overlay else "#222222", fontweight="bold", pad=8)
    if result["status"] == "returned":
        row = result["summary"]
        description = (f"Normalized 3D set error: {row['normalized_3d_set_error']:.5f} | "
                       f"converged: {row['optimizer_success']} | evaluations: {row['nfev']}\n"
                       + textwrap.fill("Flags: "+(row["flags"] or "none").replace("_", " "), width=112))
    else:
        description = "FIT FAILED: "+textwrap.fill(result["reason"], width=110)
    fig.text(.06, .19, description, fontsize=10, va="top", linespacing=1.5)
    fig.text(.5, .035, "Same camera and fixed axis limits throughout this trial's sweep. No projection or rescaling.\n"
             "K = 1 is the working assumption, not a topological finding. No delay selected automatically.",
             ha="center", fontsize=9, color="#555555", linespacing=1.4)
    return fig


def trace_figure(epochs, trials):
    fig, axes = plt.subplots(3, 2, figsize=(11.7, 8.3), sharex=True)
    for x, (_, trial), ax in zip(epochs, trials.iterrows(), axes.flat):
        ax.plot(np.arange(-500, 0), x, color="#222222", lw=.8)
        ax.set(title=f"D{int(trial.direction)} | original trial {int(trial.original_trial_number)}",
               ylabel="Preprocessed amplitude", xlabel="Time before GO (ms)", xlim=(-500, 0))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Six fixed example epochs | T, channel 7", fontsize=17, fontweight="bold")
    fig.subplots_adjust(top=.88, bottom=.14, left=.09, right=.97, hspace=.52, wspace=.25)
    fig.text(.5, .03, "Lowest saved original trial ID in each direction; no selection using fit or decoding scores.\n"
             "Reused exactly: detrending, robust normalization, zero-phase 2-55 Hz filtering already applied.",
             ha="center", fontsize=10)
    return fig


def error_figure(table):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for direction, group in table.groupby("direction"):
        group = group.sort_values("tau_ms")
        ax.plot(group.tau_ms, group.normalized_3d_set_error, marker="o", ms=3,
                label=f"D{direction} / trial {int(group.original_trial_number.iloc[0])}")
    ax.set(xlabel="Delay tau (ms)", ylabel="Normalized 3D distance-to-band error",
           xticks=TAUS, xlim=(.5, 20.5), ylim=(0, None))
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(alpha=.2)
    fig.suptitle("Fit error is not a delay-selection criterion here", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=.1, right=.98, top=.86, bottom=.24)
    fig.text(.5, .06, "Very short delays can produce nearly collinear clouds and deceptively small geometric errors.\n"
             "Six descriptive examples only; no population inference or automatic selection.", ha="center", fontsize=10)
    return fig


def fit_sweep(root, source):
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}; plot-only mode can reuse it")
    config = json.loads((source/"config.json").read_text())
    assert config["recording"] == RECORDING and config["fs_hz"] == 1000
    trials_all = pd.read_csv(source/"inputs/selected_trials.csv")
    rows = example_rows(trials_all)
    trials = trials_all.iloc[rows].copy()
    with np.load(source/"clouds.npz", allow_pickle=False) as saved:
        epochs = saved["processed"][rows].copy()
        assert saved["processed"].shape == (198, 500)
        np.testing.assert_array_equal(saved["original_trial_number"][rows], trials.original_trial_number)
        np.testing.assert_array_equal(saved["labels"][rows], trials.direction)
        for row, epoch in zip(rows, epochs):
            np.testing.assert_array_equal(make_sweep(epoch)[16], saved["clouds"][row])
    if len(trials) != 6 or set(trials.direction) != set(range(1, 7)):
        raise ValueError("Expected one example in each direction")
    source_files = [Path(__file__), UNIT/"prego_geometric_fits.py", UNIT/"motor_lfp_utils.py",
                    UNIT/"report_prego_geometric_fits.py",
                    Path(importlib.import_module("NeuralFieldManifold.fits.one_torus").__file__)]
    provenance = dict(source_experiment=str(source),
                      previous_files={str(p): sha256(p) for p in source.rglob("*") if p.is_file()},
                      source_files={str(p): sha256(p) for p in source_files})
    for sub in ("checkpoints", "tables", "source", "figures", "individual"):
        (root/sub).mkdir(parents=True, exist_ok=True)
    write_json(root/"provenance.json", provenance)
    write_json(root/"config.json", dict(recording=RECORDING, K=1, dimension=3, fs_hz=1000,
               tau_samples=TAUS, tau_ms=TAUS, preprocessing="reused frozen processed array; not repeated",
               points_per_cloud="500 - 2*tau", selection="lowest original trial ID per direction",
               models={"one_torus": config["models"]["one_torus"]}, loss=config["loss"],
               fixed_axis_limits_per_trial=True, decoding=False, pca_reduction=False, automatic_selection=False))
    for path in source_files:
        shutil.copy2(path, root/"source"/path.name)
    trials.to_csv(root/"tables/example_trials.csv", index=False)
    np.savez_compressed(root/"processed_examples.npz", processed=epochs,
                        original_trial_number=trials.original_trial_number.to_numpy(),
                        row_index=trials.row_index.to_numpy(), time_ms=np.arange(-500, 0))
    table, cloud_arrays = [], {}
    for epoch, (_, trial) in zip(epochs, trials.iterrows()):
        for tau, points in make_sweep(epoch).items():
            name = f"row_{int(trial.row_index):03d}_tau_{tau:02d}"
            result = fit_one(points, "one_torus")
            result.update(row_index=int(trial.row_index), original_trial_number=int(trial.original_trial_number),
                          direction=int(trial.direction), tau_ms=tau)
            write_json(root/"checkpoints"/f"{name}.json", result)
            record = dict(row_index=int(trial.row_index), original_trial_number=int(trial.original_trial_number),
                          direction=int(trial.direction), tau_ms=tau, n_points=len(points),
                          status=result["status"], reason=result.get("reason", ""))
            record.update(result.get("summary", {}))
            table.append(record)
            cloud_arrays[name] = points
        print(f"Fitted direction {int(trial.direction)} / trial {int(trial.original_trial_number)}: all 20 delays", flush=True)
    np.savez_compressed(root/"sweep_clouds.npz", **cloud_arrays)
    pd.DataFrame(table).to_csv(root/"tables/tau_sweep_metrics.csv", index=False)
    verify(root)


def verify(root):
    provenance = json.loads((root/"provenance.json").read_text())
    for path, expected in provenance["previous_files"].items():
        if sha256(path) != expected:
            raise ValueError(f"Previous file changed: {path}")
    table = pd.read_csv(root/"tables/tau_sweep_metrics.csv").fillna("")
    trials = pd.read_csv(root/"tables/example_trials.csv")
    with np.load(root/"processed_examples.npz") as data:
        epochs = data["processed"].copy()
    source = Path(provenance["source_experiment"])
    with np.load(source/"clouds.npz") as original:
        np.testing.assert_array_equal(epochs, original["processed"][trials.row_index.to_numpy()])
    with np.load(root/"sweep_clouds.npz") as saved:
        assert len(saved.files) == len(table) == 120
        for epoch, (_, trial) in zip(epochs, trials.iterrows()):
            group = table[table.row_index == trial.row_index]
            assert sorted(group.tau_ms.tolist()) == list(TAUS)
            for tau, points in make_sweep(epoch).items():
                name = f"row_{int(trial.row_index):03d}_tau_{tau:02d}"
                np.testing.assert_array_equal(saved[name], points)
                result = json.loads((root/"checkpoints"/f"{name}.json").read_text())
                assert result["cloud_sha256"] == array_hash(points)
                row = group[group.tau_ms == tau].iloc[0]
                assert result["original_trial_number"] == trial.original_trial_number == row.original_trial_number
                assert result["direction"] == trial.direction == row.direction
                assert result["row_index"] == trial.row_index and result["tau_ms"] == tau
                assert row.status == result["status"] and row.n_points == len(points)
                if result["status"] == "returned":
                    np.testing.assert_allclose(recompute_distances(points, result["fit"], "one_torus"),
                                               result["fit"]["diagnostics"]["distance_3d"], atol=1e-10)
                    summary = summarize_fit(points, result["fit"], "one_torus")
                    for key, value in summary.items():
                        if isinstance(value, (str, bool)):
                            assert row[key] == value, (name, key)
                        elif value is not None:
                            np.testing.assert_allclose(float(row[key]), value, atol=1e-10, equal_nan=True)
    result = dict(n_example_trials=6, n_delays=20, n_results=120,
                  unchanged_source_files=len(provenance["previous_files"]),
                  epochs_reused_exactly=True, clouds_and_metrics_reproduced=True,
                  no_decoding=True, no_pca_reduction=True, no_new_preprocessing=True)
    write_json(root/"verification.json", result)
    return result


def export(fig, folder, name, dpi=300):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(folder/f"{name}.{suffix}", dpi=dpi)


def plot_sweep(root):
    verify(root)
    trials = pd.read_csv(root/"tables/example_trials.csv")
    table = pd.read_csv(root/"tables/tau_sweep_metrics.csv")
    with np.load(root/"processed_examples.npz") as data:
        epochs = data["processed"].copy()
    limits_all = {}
    with PdfPages(root/"Tau_1_to_20_Observed_Clouds.pdf") as raw_pages, \
         PdfPages(root/"Tau_1_to_20_One_Torus_Overlays.pdf") as overlay_pages, \
         PdfPages(root/"All_120_Enlarged_Comparisons.pdf") as detail_pages:
        for epoch, (_, trial) in zip(epochs, trials.iterrows()):
            clouds = make_sweep(epoch)
            results = {tau: json.loads((root/"checkpoints"/f"row_{int(trial.row_index):03d}_tau_{tau:02d}.json").read_text())
                       for tau in TAUS}
            limits = sweep_limits(clouds, results)
            limits_all[str(int(trial.row_index))] = limits
            prefix = f"D{int(trial.direction)}_trial_{int(trial.original_trial_number)}"
            for overlay, pages, suffix in ((False, raw_pages, "observed"), (True, overlay_pages, "one_torus")):
                fig = grid_figure(clouds, results, trial, limits, overlay)
                pages.savefig(fig)
                export(fig, root/"figures", f"{prefix}_{suffix}")
                plt.close(fig)
            for tau in TAUS:
                fig = detail_figure(clouds[tau], results[tau], trial, tau, limits)
                detail_pages.savefig(fig)
                export(fig, root/"individual", f"{prefix}_tau_{tau:02d}")
                plt.close(fig)
            print(f"Rendered {prefix}: 20 observed clouds, 20 overlays, 20 enlarged comparisons", flush=True)
    fig = trace_figure(epochs, trials)
    export(fig, root/"figures", "example_epochs")
    plt.close(fig)
    fig = error_figure(table)
    export(fig, root/"figures", "fit_error_by_tau")
    plt.close(fig)
    write_json(root/"display_limits.json", limits_all)
    write_json(root/"plot_provenance.json", {str(p): sha256(p) for p in (
        Path(__file__), UNIT/"report_prego_geometric_fits.py")})
    (root/"README.md").write_text("""# T channel 7: fixed K=1, m=3; tau=1-20 ms

Start with `Tau_1_to_20_Observed_Clouds.pdf`: six pages, one selected trial per
page, all 20 delays. The overlay PDF has the same views plus the existing
1-torus planar annular-band fits. `All_120_Enlarged_Comparisons.pdf` contains
one page per trial/delay: observed cloud beside fitted band. Every large
comparison is also exported separately in `individual/` as PDF, SVG, and
300-dpi PNG; overview sheets and source traces are in `figures/`.

## Fixed inputs and interpretation

- Recording: monkeyT_session-y070316009-12_lfp-7, short delay, final 500 ms before GO.
- Six examples: lowest original trial ID in each direction, unchanged from the
  tau=16 report. This is not a sweep of all 198 trials.
- The saved processed epochs are reused exactly. No re-filtering, envelope
  normalization, PCA, AR signal replacement, PINN, or decoding is performed.
- K=1 is a fixed working assumption. m=3 for every plot. At 1000 Hz, one sample
  is one millisecond. All valid lag vectors are retained: 498 points at tau=1,
  468 at tau=16, 460 at tau=20. The earliest endpoint moves with delay; the
  source epoch and final endpoint remain fixed. No trial boundaries are crossed.
- All axes have equal physical scale; limits and camera are fixed across a
  trial's complete sweep and across observed/overlay views. Different example
  trials may have different amplitude limits. There is no per-delay rescaling.
- The existing one_torus_fit objective, bounds, Huber loss, lam=0.1,
  hole_ratio=0.5, three angular penalty harmonics, and 6000-evaluation budget
  are unchanged. Penalty harmonics are not K.
- No best tau is selected. Near-zero delay can produce a nearly collinear
  cloud and small fitting errors without resolving the underlying geometry.
- Common error: squared native 3D distances to the fitted planar band divided
  by total centered cloud sum of squares. This includes out-of-plane error.
  Native coverage and native R-squared ignore out-of-plane displacement.
  These are geometric metrics, not prediction/decoding accuracy or topology tests.
- All failed/nonconverged or boundary-flagged fits remain in checkpoints/tables.

## Reproduction

Run `prego_tau_sweep.py fit` for a NEW experiment directory (existing outputs
are protected), then `prego_tau_sweep.py plot` for plot-only regeneration.
`prego_tau_sweep.py verify` checks arrays, identities, and saved fit metrics.
All commands accept `--output`; fit additionally accepts `--source`.
Frozen inputs, original-coordinate clouds, per-fit parameters/diagnostics,
source snapshots/hashes, and the metric CSV are included here.
""")
    verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "plot", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    if args.action == "fit":
        fit_sweep(args.output.resolve(), args.source.resolve())
    elif args.action == "plot":
        plot_sweep(args.output.resolve())
    else:
        print(verify(args.output.resolve()))


if __name__ == "__main__":
    main()
