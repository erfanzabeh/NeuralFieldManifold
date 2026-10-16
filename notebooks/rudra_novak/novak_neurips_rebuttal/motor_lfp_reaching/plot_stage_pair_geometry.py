"""Plot saved, corrected pre/post-SC and pre/post-GO geometry only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image


UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
COLLECTION = UNIT / "outputs/macaque_task_six_stages_cue_offset_correct"
OUTPUT = COLLECTION / "stage_pair_geometry_v1"
EXAMPLE_TRIALS = (6, 9, 10)
INK = "#263238"
PAIRS = {
    "SC": ("pre_SC", "post_SC", "#E8B1A3", "#BE5A47"),
    "GO": ("pre_GO", "post_GO", "#9DC5B5", "#38816A"),
}
RADIAL = ("R1", "R2", "band_half_width")
ORIENTATION = tuple(f"{vector}_{axis}" for vector in ("normal", "u", "v") for axis in "xyz")
QUALITY = ("mse", "mean_error", "frac_inside")
FEATURES = RADIAL + ORIENTATION + QUALITY
TITLES = {
    "R1": "Outer radius R1", "R2": "Outer radius R2", "band_half_width": "Band half-width",
    "mse": "Mean squared fit error", "mean_error": "Mean in-plane error",
    "frac_inside": "Fraction inside",
    **{f"{v}_{a}": f"{name}: {a}" for v, name in (
        ("normal", "Plane normal"), ("u", "Major-axis direction"),
        ("v", "Minor-axis direction")) for a in "xyz"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "text.color": INK, "axes.grid": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, name, manifest):
    path = OUTPUT / name
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    with Image.open(path) as image:
        pixels = image.size
        rgb = np.asarray(image.convert("RGB"))
        if min(pixels) < 1000 or np.std(rgb) < 5:
            raise ValueError(f"Blank or unexpectedly small figure: {path}")
    manifest.append({"file": name, "pixels": list(pixels), "sha256": sha256(path)})


def load():
    feature_path = COLLECTION / "tables/features.csv"
    windows_path = SOURCE / "inputs/windows.npz"
    table = pd.read_csv(feature_path).sort_values("window_index").reset_index(drop=True)
    if len(table) != 1188 or not np.array_equal(table.window_index, np.arange(1188)):
        raise ValueError("Expected all 1,188 saved windows in original order")
    if table.original_trial_number.nunique() != 198 or not (table.delay == "short").all():
        raise ValueError("Saved 198-trial short-delay cohort changed")
    if not np.array_equal(table.groupby("original_trial_number").stage.nunique().to_numpy(),
                          np.full(198, 6)):
        raise ValueError("Each trial must contain all six stages")
    for _, (before, after, _, _) in PAIRS.items():
        if (table.stage_name.eq(before).sum(), table.stage_name.eq(after).sum()) != (198, 198):
            raise ValueError("Incorrect stage counts")
    for stage, code in (("pre_SC", 204), ("post_SC", 206),
                        ("pre_GO", 207), ("post_GO", 207)):
        rows = table.loc[table.stage_name == stage]
        boundary = rows.start_sample if stage.startswith("post") else rows.end_sample_exclusive
        if not (rows.anchor_code == code).all() or not np.array_equal(boundary, rows.event_sample):
            raise ValueError(f"Incorrect cue-offset or GO alignment for {stage}")
        if not (rows.end_sample_exclusive - rows.start_sample == 300).all():
            raise ValueError(f"Incorrect window duration for {stage}")
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv").sort_values("window_index")
    for column in ("window_index", "original_trial_number", "stage_name", "start_sample",
                   "end_sample_exclusive", "anchor_code"):
        np.testing.assert_array_equal(table[column].to_numpy(), metadata[column].to_numpy())
    with np.load(windows_path, allow_pickle=False) as saved:
        processed = saved["processed"]
        clouds = saved["clouds"]
    if clouds.shape != (1188, 294, 3) or processed.shape != (1188, 300):
        raise ValueError("Saved window or cloud shape changed")
    np.testing.assert_array_equal(
        clouds, np.stack((processed[:, 6:], processed[:, 3:-3], processed[:, :-6]), axis=2)
    )
    for row in table.itertuples():
        if array_hash(clouds[row.window_index]) != row.cloud_sha256:
            raise ValueError(f"Cloud hash changed at window {row.window_index}")
    return table, clouds, [feature_path, windows_path, SOURCE / "inputs/window_metadata.csv"]


def pair_table(table, before, after):
    left = table.loc[table.stage_name == before, ["original_trial_number", "direction", "usable", *FEATURES]]
    right = table.loc[table.stage_name == after, ["original_trial_number", "direction", "usable", *FEATURES]]
    pair = left.merge(right, on="original_trial_number", suffixes=("_pre", "_post"), validate="one_to_one")
    if len(pair) != 198 or not np.array_equal(pair.direction_pre, pair.direction_post):
        raise ValueError("Stage pairing or reach-direction metadata changed")
    pair["paired_usable"] = pair.usable_pre & pair.usable_post
    for feature in FEATURES:
        pair[f"delta_{feature}"] = pair[f"{feature}_post"] - pair[f"{feature}_pre"]
        pair.loc[~pair.paired_usable, f"delta_{feature}"] = np.nan
        if not np.isfinite(pair.loc[pair.paired_usable, [f"{feature}_pre", f"{feature}_post"]]).all().all():
            raise ValueError(f"Nonfinite usable feature: {feature}")
    return pair


def summary_rows(pair, event):
    usable = pair.loc[pair.paired_usable]
    rows = []
    for feature in FEATURES:
        a = usable[f"{feature}_pre"].to_numpy(dtype=float)
        b = usable[f"{feature}_post"].to_numpy(dtype=float)
        d = usable[f"delta_{feature}"].to_numpy(dtype=float)
        rows.append({
            "event": event, "feature": feature, "n_matched_trials": len(usable),
            "pre_median": float(np.median(a)), "post_median": float(np.median(b)),
            "delta_median": float(np.median(d)), "delta_q1": float(np.quantile(d, .25)),
            "delta_q3": float(np.quantile(d, .75)),
        })
    return rows


def radial_figure(event, pair, before, after, colors, manifest):
    paired = pair.loc[pair.paired_usable]
    rng = np.random.default_rng(20260929)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.3), gridspec_kw={"height_ratios": [2.2, 1]},
                             constrained_layout=True)
    for column, feature in enumerate(RADIAL):
        ax, delta_ax = axes[:, column]
        values = [paired[f"{feature}_pre"].to_numpy(dtype=float),
                  paired[f"{feature}_post"].to_numpy(dtype=float)]
        for position, (v, color) in enumerate(zip(values, colors)):
            jitter = rng.uniform(-.11, .11, len(v))
            ax.scatter(position + jitter, v, s=6, color=color, alpha=.35,
                       linewidths=0, rasterized=True, zorder=1)
        boxes = ax.boxplot(values, positions=(0, 1), widths=.26, showfliers=False, whis=[5, 95],
                           patch_artist=True, medianprops={"color": INK, "linewidth": 1.2},
                           boxprops={"color": INK, "linewidth": .8},
                           whiskerprops={"color": INK, "linewidth": .8},
                           capprops={"color": INK, "linewidth": .8})
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(.45)
        ax.set_xticks((0, 1), (before.replace("_", "-"), after.replace("_", "-")))
        ax.set_xlim(-.4, 1.4)
        ax.set_title(TITLES[feature])
        if column == 0:
            ax.set_ylabel("Normalized LFP units")
        delta = paired[f"delta_{feature}"].to_numpy(dtype=float)
        delta_ax.axvline(0, color="#9AA4A9", linewidth=.9, linestyle="--", zorder=0)
        delta_ax.scatter(delta, rng.uniform(-.3, .3, len(delta)), s=7, color=INK,
                         alpha=.3, linewidths=0, rasterized=True)
        delta_ax.axvline(float(np.median(delta)), color=colors[1], linewidth=2)
        delta_ax.set_yticks([])
        delta_ax.set_ylim(-.38, .38)
        delta_ax.set_xlabel("Post - pre")
        if column == 0:
            delta_ax.set_ylabel("Paired trials")
    save(fig, f"{event.lower()}_radii_width.png", manifest)


def ecdf(ax, values, color):
    ordered = np.sort(np.asarray(values, dtype=float))
    ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
            where="post", color=color, linewidth=1.5)


def ecdf_figure(event, pair, features, name, colors, manifest):
    paired = pair.loc[pair.paired_usable]
    rows = 3 if len(features) == 9 else 1
    fig, axes = plt.subplots(rows, 3, figsize=(9.0, 6.6 if rows == 3 else 2.6),
                             sharey=True, constrained_layout=True)
    for index, (ax, feature) in enumerate(zip(np.ravel(axes), features)):
        ecdf(ax, paired[f"{feature}_pre"], colors[0])
        ecdf(ax, paired[f"{feature}_post"], colors[1])
        ax.set_title(TITLES[feature])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Value")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if index % 3 == 0:
            ax.set_ylabel("Cumulative fraction")
    fig.legend(("Pre", "Post"), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.18 if rows == 1 else 1.04))
    save(fig, f"{event.lower()}_{name}.png", manifest)


def fitted_outlines(result):
    fit = result["fit"]
    t = np.linspace(0, 2 * np.pi, 241)
    center = np.asarray(fit["center"], dtype=float)
    u = np.asarray(fit["u_axis"], dtype=float)
    v = np.asarray(fit["v_axis"], dtype=float)
    return [center + (a * np.cos(t))[:, None] * u + (b * np.sin(t))[:, None] * v
            for a, b in ((fit["R1"], fit["R2"]), (fit["R1_in"], fit["R2_in"]))]


def geometry_figure(event, table, clouds, before, after, colors, manifest):
    chosen = table.loc[table.original_trial_number.isin(EXAMPLE_TRIALS) &
                       table.stage_name.isin((before, after))].copy()
    if len(chosen) != 6 or not chosen.usable.all():
        raise ValueError("Predeclared example trials are incomplete or have unusable fits")
    fits = {}
    extents = []
    for row in chosen.itertuples():
        path = SOURCE / "checkpoints" / f"window_{row.window_index:04d}.json"
        result = json.loads(path.read_text())
        if result["cloud_sha256"] != row.cloud_sha256 or result["status"] != "returned":
            raise ValueError(f"Saved fit does not match cloud {row.window_index}")
        fits[row.window_index] = result
        extents.append(np.max(np.abs(clouds[row.window_index])))
        extents.extend(np.max(np.abs(outline)) for outline in fitted_outlines(result))
    limit = 1.08 * max(extents)
    fig = plt.figure(figsize=(7.4, 8.6))
    for row_idx, trial in enumerate(EXAMPLE_TRIALS):
        for col_idx, (stage, color) in enumerate(zip((before, after), colors)):
            row = chosen.loc[(chosen.original_trial_number == trial) &
                             (chosen.stage_name == stage)].iloc[0]
            cloud = clouds[int(row.window_index)]
            ax = fig.add_subplot(3, 2, row_idx * 2 + col_idx + 1, projection="3d")
            ax.plot(*cloud.T, color=color, linewidth=.8, alpha=.8)
            ax.scatter(*cloud.T, s=2.0, color=color, alpha=.55,
                       depthshade=False, linewidths=0, rasterized=True)
            for outline, linestyle in zip(fitted_outlines(fits[int(row.window_index)]), ("-", "--")):
                ax.plot(*outline.T, color=INK, linewidth=.9, linestyle=linestyle)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=24, azim=-58)
            ax.set_xticks((-5, 0, 5))
            ax.set_yticks((-5, 0, 5))
            ax.set_zticks((-5, 0, 5))
            ax.tick_params(labelsize=7, pad=-1)
            ax.set_xlabel("x(t)", fontsize=8, labelpad=-1)
            ax.set_ylabel("x(t - 3 ms)", fontsize=8, labelpad=-1)
            ax.set_zlabel("x(t - 6 ms)", fontsize=8, labelpad=-1)
            ax.set_title(f"{stage.replace('_', '-')} | trial {trial}, direction {int(row.direction)}",
                         fontsize=10, pad=3)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor("white")
            ax.grid(False)
    fig.subplots_adjust(left=.02, right=.94, top=.97, bottom=.04, wspace=.02, hspace=.14)
    save(fig, f"{event.lower()}_example_clouds_trials_6_9_10.png", manifest)
    return {"event": event, "trials": list(EXAMPLE_TRIALS), "common_axis_limit": float(limit),
            "window_indices": chosen.window_index.astype(int).tolist()}


def main():
    style()
    table, clouds, source_paths = load()
    source_paths.append(Path(__file__))
    example_rows = table.loc[table.original_trial_number.isin(EXAMPLE_TRIALS) &
                             table.stage_name.isin(("pre_SC", "post_SC", "pre_GO", "post_GO"))]
    source_paths.extend(SOURCE / "checkpoints" / f"window_{i:04d}.json"
                        for i in example_rows.window_index.astype(int))
    before_hashes = {str(path): sha256(path) for path in source_paths}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, summaries, paired_frames, examples = [], [], [], []
    for event, (before, after, pre_color, post_color) in PAIRS.items():
        pair = pair_table(table, before, after)
        pair.insert(0, "comparison", f"{before}_vs_{after}")
        paired_frames.append(pair)
        summaries.extend(summary_rows(pair, event))
        colors = (pre_color, post_color)
        radial_figure(event, pair, before, after, colors, manifest)
        ecdf_figure(event, pair, ORIENTATION, "orientation", colors, manifest)
        ecdf_figure(event, pair, QUALITY, "fit_quality", colors, manifest)
        examples.append(geometry_figure(event, table, clouds, before, after,
                                        colors, manifest))
    if any(sha256(path) != before_hashes[str(path)] for path in source_paths):
        raise ValueError("A protected source changed during plotting")
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "feature_summary.csv", index=False)
    pd.concat(paired_frames, ignore_index=True).to_csv(OUTPUT / "paired_trial_features.csv", index=False)
    (OUTPUT / "example_index.json").write_text(json.dumps(examples, indent=2) + "\n")
    (OUTPUT / "source_hashes.json").write_text(json.dumps(before_hashes, indent=2) + "\n")
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    n_sc = int(paired_frames[0].paired_usable.sum())
    n_go = int(paired_frames[1].paired_usable.sum())
    readme = f"""# Corrected stage-pair geometry: Monkey T, channel 7

Plot-only comparisons from the saved 198 short-delay trials of session y070316009-12.
Each stage uses its corrected 300-ms window, m=3, tau=3 ms, and 294 observed points.
Post-SC begins after the spatial instruction ends (event 206), not at SC onset.
The existing 1-torus fits and 15-feature table are reused without refitting or decoding.
Band half-width is `(R1 - R1_inner + R2 - R2_inner) / 4` in normalized signal units.

## Files

| File | Content |
| --- | --- |
| `sc_example_clouds_trials_6_9_10.png` | Pre-SC versus post-SC observed clouds and saved annular fit outlines, for original trials 6, 9, 10. |
| `go_example_clouds_trials_6_9_10.png` | Pre-GO versus post-GO for the same trials. |
| `sc_radii_width.png` / `go_radii_width.png` | R1, R2, and band half-width: matched-trial distributions above; within-trial post-minus-pre values below. |
| `sc_orientation.png` / `go_orientation.png` | Empirical cumulative distributions of all nine saved orientation components. |
| `sc_fit_quality.png` / `go_fit_quality.png` | Empirical cumulative distributions of MSE, mean in-plane error, and fraction inside. |
| `feature_summary.csv` | Matched-trial medians and quartiles of each post-minus-pre change. |
| `paired_trial_features.csv` | All 198 trial identities and their pre/post features, usability, and differences for each comparison. |
| `example_index.json` | Example trial and saved-window identities, with 3D axis limits. |
| `source_hashes.json` / `figure_manifest.json` | Source and figure integrity records. |

SC has {n_sc} matched trials with both fits usable; GO has {n_go}. Unusable fits remain
in `paired_trial_features.csv` but are absent from plotted geometric distributions.
The example trial numbers were fixed by the earlier six-stage figure; they were not
selected for visually large differences. Outer fit boundary is solid charcoal;
inner fit boundary is dashed. Identical coordinate limits and viewpoints are used
within each two-stage example figure. Plots are descriptive, from one channel and
one recording; no significance testing or decoder rerun was performed. Every
300-ms window was normalized separately before embedding, so a radius change
cannot be interpreted as a change in raw LFP amplitude.
"""
    (OUTPUT / "README.md").write_text(readme)
    print(f"Saved {len(manifest)} PNGs to {OUTPUT}")
    print(f"Matched usable trials: SC={n_sc}, GO={n_go}")
    print(summary.loc[summary.feature.isin(RADIAL), ["event", "feature", "delta_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
