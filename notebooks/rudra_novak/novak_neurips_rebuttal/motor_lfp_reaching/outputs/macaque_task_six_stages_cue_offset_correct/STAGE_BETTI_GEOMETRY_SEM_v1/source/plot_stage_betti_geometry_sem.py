"""Stage-wise mean/SEM bars from frozen persistence intervals and fit features."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from plot_stage_persistent_homology import style
from run_stage_persistent_homology import OUTPUT as PH_ROOT, STAGES, LABELS, COLORS, digest, write_json, verify_manifest

UNIT = Path(__file__).resolve().parent
COLLECTION = UNIT / "outputs/macaque_task_six_stages_cue_offset_correct"
FEATURE_ROOT = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
OUTPUT = COLLECTION / "STAGE_BETTI_GEOMETRY_SEM_v1"
FEATURES = ("R1", "R2", "band_half_width", "normal_x", "normal_y", "normal_z",
    "u_x", "u_y", "u_z", "v_x", "v_y", "v_z", "mse", "mean_error", "frac_inside")
TITLES = {"R1": "Outer radius R1", "R2": "Outer radius R2", "band_half_width": "Band half-width",
          "mse": "Mean squared fit error", "mean_error": "Mean in-plane error",
          "frac_inside": "Fraction inside"}
for prefix, name in (("normal", "Plane normal"), ("u", "Major-axis vector"),
                     ("v", "Minor-axis vector")):
    for component in "xyz":
        TITLES[f"{prefix}_{component}"] = f"{name}: {component}"


def validate_cohort(meta, features):
    if len(meta) != 1188 or len(features) != 1188:
        raise ValueError("Expected 1,188 saved windows in both datasets.")
    if not np.array_equal(meta.window_index, np.arange(1188)) or not np.array_equal(features.window_index, meta.window_index):
        raise ValueError("Window ordering differs between saved results.")
    for column in ("trial_index", "original_trial_number", "stage", "stage_name", "direction"):
        if not np.array_equal(meta[column].to_numpy(), features[column].to_numpy()):
            raise ValueError(f"Trial/stage identity mismatch: {column}")
    if (meta.trial_index.nunique() != 198 or not meta.status.eq("success").all()
            or not meta.normalization_valid.all() or not (meta.geometry_usable.to_numpy() == features.usable.to_numpy()).all()):
        raise ValueError("Expected 198 complete PH trials and unchanged geometric fit validity.")
    if not meta.groupby("trial_index").stage.nunique().eq(6).all():
        raise ValueError("Each trial must have six stages.")
    if not meta.groupby(["stage", "direction"]).size().eq(33).all():
        raise ValueError("Expected 33 trials per reach direction per stage.")
    if not all(row.stage_name == STAGES[row.stage] for row in meta.itertuples()):
        raise ValueError("Unexpected task stage labels.")
    if not np.isfinite(meta.centered_rms.to_numpy()).all() or not meta.centered_rms.gt(0).all():
        raise ValueError("Invalid centered RMS radii.")
    config = json.loads((FEATURE_ROOT / "config.json").read_text())
    expected_anchors = ("TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO")
    if (tuple(config["stage_anchors"]) != expected_anchors
            or (config["m"], config["tau_ms"], config["samples_per_window"]) != (3, 3, 300)
            or not features.delay.eq("short").all()
            or not np.array_equal(features.anchor_name.to_numpy(),
                                  np.asarray(expected_anchors)[features.stage.to_numpy()])
            or not (features.end_sample_exclusive-features.start_sample).eq(300).all()):
        raise ValueError("Corrected short-delay cue-offset window definition changed.")
    if set(FEATURES) != set(config["feature_columns"]):
        raise ValueError("The frozen 15-feature definition changed.")
    if len(features.loc[features.usable]) != 1177 or not np.isfinite(features.loc[features.usable, FEATURES].to_numpy()).all():
        raise ValueError("Expected exactly 1,177 complete usable feature rows.")


def average_betti_counts(intervals, meta, endpoint, normalized):
    """Exact 1/D integral of beta_h(r) from 0 to a common D, per window."""
    if endpoint <= 0 or not np.isfinite(endpoint):
        raise ValueError("Expected positive, finite distance endpoint.")
    window = intervals.window_index.to_numpy(dtype=int)
    homology = intervals.homology_dimension.to_numpy(dtype=int)
    if (window.min() < 0 or window.max() >= len(meta) or not np.isin(homology, [0, 1, 2]).all()):
        raise ValueError("Unknown window or homology dimension.")
    if not np.array_equal(intervals.stage_name.to_numpy(), meta.stage_name.to_numpy()[window]):
        raise ValueError("Interval stage does not match the window metadata.")
    if not np.array_equal(intervals.original_trial_number.to_numpy(), meta.original_trial_number.to_numpy()[window]):
        raise ValueError("Interval trial does not match the window metadata.")
    scale = meta.centered_rms.to_numpy()[window] if normalized else 1.
    birth = intervals.birth.to_numpy(dtype=float) / scale
    death = intervals.death.to_numpy(dtype=float) / scale
    if (not np.isfinite(birth).all() or (birth < 0).any() or np.isnan(death).any()
            or (death < birth).any() or np.any(np.isinf(death) & (homology != 0))):
        raise ValueError("Invalid stored persistence intervals.")
    observed = np.bincount(window*3+homology, minlength=len(meta)*3).reshape(len(meta), 3)
    reference = meta[[f"H{h}_intervals" for h in range(3)]].to_numpy()
    np.testing.assert_array_equal(observed, reference)
    lifetime_in_range = np.maximum(0., np.minimum(death, endpoint)-np.maximum(birth, 0.))
    totals = np.bincount(window*3+homology, weights=lifetime_in_range/endpoint,
                         minlength=len(meta)*3).reshape(len(meta), 3)
    if not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("Invalid integrated Betti count.")
    return totals


def stage_summary(table, names):
    rows = []
    for name in names:
        for stage in range(6):
            group = table.loc[table.stage.eq(stage) & np.isfinite(table[name]), name].to_numpy(dtype=float)
            if len(group) < 2:
                raise ValueError(f"Too few observations for {name} / {STAGES[stage]}")
            sd = float(group.std(ddof=1))
            rows.append(dict(measure=name, stage=stage, stage_name=STAGES[stage],
                n=len(group), mean=float(group.mean()), sd=sd, sem=sd/np.sqrt(len(group))))
    return pd.DataFrame(rows)


def betti_panel(summary, convention):
    subset = summary[summary.distance_convention.eq(convention)]
    fig, ax = plt.subplots(figsize=(5.1, 3.6))
    fig.subplots_adjust(left=.14, right=.97, bottom=.18, top=.73)
    width = .12
    for stage, (label, color) in enumerate(zip(LABELS, COLORS)):
        part = subset[subset.stage.eq(stage)].set_index("measure").loc[["beta0", "beta1", "beta2"]]
        x = np.arange(3) + (stage-2.5)*width
        ax.bar(x, part["mean"], width=width*.93, color=color, edgecolor="none", label=label, zorder=2)
        ax.errorbar(x, part["mean"], yerr=part["sem"], fmt="none", ecolor="#252525",
                    elinewidth=.75, capsize=1.5, capthick=.75, zorder=3)
    ax.set_xticks(range(3), [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$"])
    ax.set_xlim(-.52, 2.52)
    ax.set_ylabel("Mean Betti count across distance")
    ax.set_yscale("symlog", linthresh=.01, linscale=1)
    ax.set_yticks([0, .01, .1, 1, 10, 100], ["0", ".01", ".1", "1", "10", "100"])
    ax.set_ylim(0, 120)
    ax.tick_params(axis="x", length=0)
    fig.legend(loc="upper center", bbox_to_anchor=(.55, .99), ncol=3,
               frameon=False, columnspacing=1.2, handlelength=1.3)
    return fig


def feature_panel(summary, feature):
    subset = summary[summary.measure.eq(feature)].sort_values("stage")
    if len(subset) != 6:
        raise ValueError(f"Expected six stage summaries for {feature}")
    fig, ax = plt.subplots(figsize=(4.15, 3.15))
    fig.subplots_adjust(left=.19, right=.97, bottom=.25, top=.88)
    ax.bar(np.arange(6), subset["mean"], color=COLORS, width=.72, edgecolor="none", zorder=2)
    ax.errorbar(np.arange(6), subset["mean"], yerr=subset["sem"], fmt="none",
        ecolor="#252525", elinewidth=.8, capsize=2, capthick=.8, zorder=3)
    low = min(0., float((subset["mean"]-subset["sem"]).min()))
    high = max(0., float((subset["mean"]+subset["sem"]).max()))
    if low < 0:
        low *= 1.12
    if high > 0:
        high *= 1.12
    ax.set_ylim(low, high)
    ax.axhline(0, color="#383838", lw=.65)
    ax.set_xlim(-.55, 5.55)
    ax.set_xticks(range(6), LABELS, rotation=32, ha="right")
    ax.tick_params(axis="x", length=0)
    if feature.startswith(("normal_", "u_", "v_")):
        unit = "Orientation component"
    elif feature == "mse":
        unit = "Squared normalized units"
    elif feature == "frac_inside":
        unit = "Fraction inside"
    else:
        unit = "Normalized signal units"
    ax.set_ylabel(unit)
    ax.set_title(TITLES[feature], pad=7)
    ticks = MaxNLocator(5).tick_values(*ax.get_ylim())
    ax.set_yticks(ticks[(ticks >= ax.get_ylim()[0]) & (ticks <= ax.get_ylim()[1])])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    return fig


def export_png(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(matplotlib.text.Text):
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            raise ValueError(f"Text clipped in {path.name}: {label.get_text()}")
    fig.savefig(path, dpi=600, metadata={"Software": "NeuralFieldManifold saved-results renderer"})
    plt.close(fig)
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        rgb = np.asarray(im.convert("RGB"))
        if not dpi or not all(abs(d-600) < .1 for d in dpi):
            raise ValueError(f"Wrong DPI for {path.name}")
        if min(im.size) < 1800 or not np.any(rgb < 240):
            raise ValueError(f"Blank or undersized panel: {path.name}")
        if not all(np.all(edge > 245) for edge in (rgb[:3], rgb[-3:], rgb[:, :3], rgb[:, -3:])):
            raise ValueError(f"Artwork touches canvas edge: {path.name}")
        return dict(file=path.name, sha256=digest(path), pixels=list(im.size), dpi=list(dpi))


def main():
    style()
    verify_manifest(PH_ROOT / "provenance/table_hashes.json")
    ph_meta_path = PH_ROOT / "tables/window_measurements.csv"
    interval_path = PH_ROOT / "tables/persistence_intervals.csv.gz"
    grid_path = PH_ROOT / "tables/grid_definitions.json"
    feature_path = FEATURE_ROOT / "tables/features.csv"
    config_path = FEATURE_ROOT / "config.json"
    sources = [ph_meta_path, interval_path, grid_path, feature_path, config_path]
    source_hashes = {str(path): digest(path) for path in sources}
    meta = pd.read_csv(ph_meta_path, float_precision="round_trip")
    features = pd.read_csv(feature_path, float_precision="round_trip")
    validate_cohort(meta, features)
    intervals = pd.read_csv(interval_path, float_precision="round_trip")
    grids = json.loads(grid_path.read_text())
    finite_h0 = intervals[intervals.homology_dimension.eq(0) & np.isfinite(intervals.death)]
    raw_h0_max = float(finite_h0.death.max())
    scaled_h0_max = float((finite_h0.death.to_numpy()
        / meta.centered_rms.to_numpy()[finite_h0.window_index.to_numpy()]).max())
    np.testing.assert_allclose(raw_h0_max, grids["raw_H0"]["max_distance"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaled_h0_max, grids["rms_normalized_H0"]["max_distance"], rtol=0, atol=1e-12)
    window_betti_parts = []
    summaries = []
    endpoints = {}
    for mode in ("raw", "rms_normalized"):
        endpoint = float(grids[f"{mode}_H0"]["max_distance"])
        endpoints[mode] = endpoint
        totals = average_betti_counts(intervals, meta, endpoint, mode == "rms_normalized")
        values = meta[["window_index", "trial_index", "original_trial_number",
                       "stage", "stage_name", "direction"]].copy()
        values["distance_convention"] = mode
        for h in range(3):
            values[f"beta{h}"] = totals[:, h]
        window_betti_parts.append(values)
        summary = stage_summary(values, ["beta0", "beta1", "beta2"])
        summary["distance_convention"] = mode
        summary["distance_endpoint"] = endpoint
        summaries.append(summary)
    beta_summary = pd.concat(summaries, ignore_index=True)
    window_betti = pd.concat(window_betti_parts, ignore_index=True)
    valid = features[features.usable].copy()
    geometry_summary = stage_summary(valid, FEATURES)
    counts = geometry_summary.groupby("stage_name").n.first().reindex(STAGES)
    for folder in ("png", "tables", "captions", "source", "validation"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    window_betti.to_csv(OUTPUT / "tables/integrated_betti_per_window.csv", index=False)
    beta_summary.to_csv(OUTPUT / "tables/betti_mean_sem.csv", index=False)
    geometry_summary.to_csv(OUTPUT / "tables/geometry_mean_sem.csv", index=False)
    valid[["window_index", "trial_index", "original_trial_number", "stage", "stage_name", *FEATURES]].to_csv(
        OUTPUT / "tables/usable_geometry_values.csv", index=False)
    exports = []
    for mode in ("raw", "rms_normalized"):
        name = f"betti_0_1_2_mean_sem_{mode}"
        exports.append(export_png(betti_panel(beta_summary, mode), OUTPUT / "png" / f"{name}.png"))
        (OUTPUT / "captions" / f"{name}.txt").write_text(
            "Mean and SEM of per-trial integrated Betti counts across six corrected task stages "
            "(198 trials per stage). For each cloud and homology dimension, integrate beta_h(distance) "
            f"from 0 to {endpoints[mode]:.6g} and divide by that common interval length. "
            "The endpoint is the largest finite H0 death across all saved clouds and is the same "
            "for beta0, beta1, beta2 and all stages in this panel. "
            f"Distance convention: {mode}. Y axis uses a zero-preserving symmetric log scale "
            "because the dimensions span orders of magnitude. Error bars are trial SEM, not a "
            "test of stage differences. All 1,188 saved topology clouds contribute.\n")
    for feature in FEATURES:
        name = f"geometry_mean_sem_{feature}"
        exports.append(export_png(feature_panel(geometry_summary, feature), OUTPUT / "png" / f"{name}.png"))
        (OUTPUT / "captions" / f"{name}.txt").write_text(
            f"{TITLES[feature]}: mean and SEM across usable geometric fits in each corrected "
            "task stage. Trial counts in stage order: " + ", ".join(map(str, counts.to_numpy())) + ". "
            "All 15 values are the frozen features used in the geometric decoder; no imputation "
            "enters these descriptive plots. Repeated stages belong to the same trials.\n")
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Scientific source changed during plotting: {path}")
    verify_manifest(PH_ROOT / "provenance/table_hashes.json")
    if "ripser" in sys.modules or any(name.startswith("NeuralFieldManifold.fits") for name in sys.modules):
        raise ValueError("Plotting imported an analysis fitter.")
    shutil.copy2(__file__, OUTPUT / "source" / Path(__file__).name)
    write_json(OUTPUT / "validation/manifest.json", dict(passed=True,
        source_sha256=digest(__file__), source_hashes=source_hashes, endpoints=endpoints,
        n_trials=198, n_topology_windows=1188, n_geometry_usable=1177,
        geometry_counts_by_stage=counts.to_dict(), ph_calls=0, fitter_calls=0,
        decoder_calls=0, files=exports))
    links = "\n".join(f"- [{entry['file']}](png/{entry['file']})" for entry in exports)
    (OUTPUT / "README.md").write_text(f"""# Stage Betti and Geometry Mean/SEM Bars

Monkey T, session y070316009-12, channel 7. All 198 short-delay trials were
included in topology; 1,177 of 1,188 geometric fits were usable. Post-TC starts
at tone offset and post-SC at spatial-instruction end. The existing stage colors
are reused in every panel. All exports are standalone 600-dpi PNGs.

{links}

## Betti definition

A Betti number changes with distance. Each trial's plotted value is the exact
average Betti count from zero to a **common endpoint** (the maximum finite H0
death over all saved clouds): raw distance {endpoints['raw']:.6g}, or RMS-normalized
distance {endpoints['rms_normalized']:.6g}. The three dimensions share the same
endpoint within a plot. Saved birth/death intervals are integrated exactly,
including each H0 infinite interval. The y axis uses a zero-preserving symmetric
log scale so beta0, beta1, and beta2 are visible together.

SEM is SD/sqrt(n) across trials within a stage, not across animals. The stages
reuse the same trials; these error bars do not test paired stage differences.
The geometric panels use only valid saved 15-dimensional feature vectors and
exclude the 11 unusable fits without imputation. Radii and width use per-window
normalized signal units; orientation vectors are in lag coordinates. The 15
features differ in scale, so each panel has its own y axis.

No persistence calculation, geometric fitting, preprocessing, or decoding was
run. See `tables/` for every per-trial value and the exact means, SDs, and SEMs;
see `captions/`, `source/`, and `validation/` for details.
""")
    print(json.dumps(dict(folder=str(OUTPUT), panels=len(exports), betti_windows=1188,
        usable_geometry_windows=1177, validated=True), indent=2))


if __name__ == "__main__":
    main()
