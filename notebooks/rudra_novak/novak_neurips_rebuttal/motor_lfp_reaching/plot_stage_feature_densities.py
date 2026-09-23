"""Stage-colored distributions of all frozen geometric decoder features."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from plot_stage_geometry_decoding import save, style
from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, COLUMNS, LABELS, COLORS, STAGES

DESTINATION = OUTPUT / "feature_distributions_v1"
ORDER = ["R1", "R2", "band_half_width", "normal_x", "normal_y", "normal_z",
         "u_x", "u_y", "u_z", "v_x", "v_y", "v_z", "mse", "mean_error", "frac_inside"]
TITLES = dict(R1="Outer radius R1", R2="Outer radius R2", band_half_width="Band half-width",
              mse="Mean squared fit error", mean_error="Mean in-plane error", frac_inside="Fraction inside")
for prefix, label in (("normal", "Plane normal"), ("u", "Major-axis vector"), ("v", "Minor-axis vector")):
    for component in "xyz":
        TITLES[f"{prefix}_{component}"] = f"{label}: {component}"


def feature_bounds(column):
    if column.startswith(("normal_", "u_", "v_")):
        return -1., 1.
    return (0., 1.) if column == "frac_inside" else (0., None)


def feature_densities(table, column):
    groups = [table.loc[table.stage == stage, column].to_numpy(dtype=float) for stage in range(6)]
    if any(len(values) < 2 or not np.isfinite(values).all() or np.std(values, ddof=1) == 0 for values in groups):
        raise ValueError(f"Cannot estimate a nondegenerate density for {column}")
    pooled = np.concatenate(groups)
    # A single absolute Scott bandwidth prevents stage-specific smoothing from shaping comparisons.
    bandwidth = float(np.std(pooled, ddof=1) * np.mean([len(v) for v in groups]) ** (-.2))
    lower, upper = feature_bounds(column)
    if pooled.min() < lower or (upper is not None and pooled.max() > upper):
        raise ValueError(f"Feature outside its physical range: {column}")
    left = max(lower, float(pooled.min() - 4 * bandwidth))
    right = float(pooled.max() + 4 * bandwidth)
    if upper is not None:
        right = min(upper, right)
    grid = np.linspace(left, right, 512)
    curves = []
    for values in groups:
        kde = gaussian_kde(values, bw_method=bandwidth / np.std(values, ddof=1))
        np.testing.assert_allclose(kde.covariance[0, 0], bandwidth**2, rtol=1e-12)
        density = kde(grid)
        # Reflect near physical boundaries, rather than displaying impossible feature values.
        density += kde(2 * lower - grid)
        if upper is not None:
            density += kde(2 * upper - grid)
        if not np.isfinite(density).all() or np.any(density < 0):
            raise AssertionError("Invalid density")
        area = np.trapezoid(density, grid)
        if not np.isclose(area, 1., atol=.002):
            raise AssertionError(f"Density mass {area} for {column}")
        curves.append(density)
    return grid, np.asarray(curves), bandwidth


def density_axis(ax, column, saved):
    grid, curves, _ = saved[column]
    for stage, color in enumerate(COLORS):
        ax.fill_between(grid, curves[stage], color=color, alpha=.07, linewidth=0)
        ax.plot(grid, curves[stage], color=color, lw=1.35)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, float(curves.max()) * 1.08)
    ax.set_title(TITLES[column], pad=7)
    if column.startswith(("normal_", "u_", "v_")):
        label = "Orientation component"
    elif column == "frac_inside":
        label = "Fraction"
    elif column == "mse":
        label = "Squared normalized units"
    else:
        label = "Normalized signal units"
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    for bounds, setter in ((ax.get_xlim(), ax.set_xticks), (ax.get_ylim(), ax.set_yticks)):
        ticks = MaxNLocator(4, min_n_ticks=3).tick_values(*bounds)
        setter(ticks[(ticks >= bounds[0]) & (ticks <= bounds[1])])
    ax.ticklabel_format(axis="both", useOffset=False, style="plain")
    ax.tick_params(length=3, pad=3)
    ax.grid(False)


def add_legend(fig, y):
    handles = [Line2D([0], [0], color=color, lw=1.7, label=label) for label, color in zip(LABELS, COLORS)]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(.5, y),
               frameon=False, fontsize=9, handlelength=1.7, columnspacing=1.3)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    folder = DESTINATION / "png"
    folder.mkdir(exist_ok=True)
    protected = {str(p): sha256(p) for p in sorted(OUTPUT.rglob("*"))
                 if p.is_file() and not p.is_relative_to(DESTINATION)}
    features = pd.read_csv(OUTPUT / "tables/features.csv")
    assert len(features) == 1188 and features.original_trial_number.nunique() == 198
    assert set(ORDER) == set(COLUMNS) and len(ORDER) == 15
    table = features.loc[features.usable].copy()
    assert len(table) == 1177 and np.isfinite(table[COLUMNS].to_numpy()).all()
    counts = table.groupby("stage").size()
    np.testing.assert_array_equal(counts, [197, 195, 196, 198, 196, 195])
    saved = {column: feature_densities(table, column) for column in ORDER}
    summaries, density_rows = [], []
    previous = pd.read_csv(OUTPUT / "tables/feature_summaries.csv")
    for column in ORDER:
        grid, curves, bandwidth = saved[column]
        for stage in range(6):
            values = table.loc[table.stage == stage, column].to_numpy()
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            old = previous[(previous.feature == column) & (previous.stage_name == STAGES[stage])].iloc[0]
            np.testing.assert_allclose([q1, median, q3], [old.q25, old["median"], old.q75], rtol=1e-10)
            assert old.n == len(values)
            summaries.append(dict(feature=column, stage=stage, stage_name=STAGES[stage], n=len(values),
                                  mean=float(values.mean()), sd=float(values.std(ddof=1)),
                                  q25=q1, median=median, q75=q3, minimum=float(values.min()),
                                  maximum=float(values.max()), bandwidth=bandwidth))
            density_rows.extend(dict(feature=column, stage=stage, x=x, density=y)
                                for x, y in zip(grid, curves[stage]))
    pd.DataFrame(summaries).to_csv(DESTINATION / "feature_summaries.csv", index=False)
    pd.DataFrame(density_rows).to_csv(DESTINATION / "density_curves.csv", index=False)
    table[["original_trial_number", "window_index", "stage", "stage_name"] + ORDER].to_csv(
        DESTINATION / "plotted_feature_values.csv", index=False)
    style()
    plt.rcParams.update({"pdf.fonttype": 42, "pdf.compression": 6})
    exports = []
    pdf_path = DESTINATION / "all_geometric_feature_distributions.pdf"
    with PdfPages(pdf_path, metadata={"Title": "All geometric features by trial stage"}) as pdf:
        fig, axes = plt.subplots(5, 3, figsize=(10.8, 13.4))
        for column, ax in zip(ORDER, axes.flat):
            density_axis(ax, column, saved)
        add_legend(fig, .997)
        fig.subplots_adjust(left=.075, right=.96, bottom=.055, top=.945, hspace=.87, wspace=.38)
        pdf.savefig(fig)
        save(fig, folder, "all_geometric_feature_distributions", exports)
        for column in ORDER:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            density_axis(ax, column, saved)
            handles = [Line2D([0], [0], color=color, lw=1.7, label=label)
                       for label, color in zip(LABELS, COLORS)]
            fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.54, .985),
                       frameon=False, fontsize=9, handlelength=1.7, columnspacing=1.5)
            fig.subplots_adjust(left=.15, right=.955, bottom=.17, top=.72)
            pdf.savefig(fig)
            save(fig, folder, f"distribution_{column}", exports)
            print(f"Exported {column}", flush=True)
    outside = {e["file"]: e["outside_text"] for e in exports if e["outside_text"]}
    assert len(exports) == 16 and not outside, outside
    with fitz.open(pdf_path) as document:
        assert len(document) == 16
        assert not any(page.get_images() for page in document)
        document.set_toc([[1, "All 15 features", 1]] + [[1, TITLES[c], i+2] for i, c in enumerate(ORDER)])
        document.saveIncr()
    # The PDF and raster exports are generated from the same figures and saved curves.
    reconstructed = pd.read_csv(DESTINATION / "density_curves.csv")
    for column in ORDER:
        for stage in range(6):
            rows = reconstructed[(reconstructed.feature == column) & (reconstructed.stage == stage)]
            np.testing.assert_allclose(rows.x, saved[column][0], atol=1e-14)
            np.testing.assert_allclose(rows.density, saved[column][1][stage], atol=1e-12)
    assert protected == {p: sha256(p) for p in protected}, "Existing experiment files changed"
    shutil.copy2(__file__, DESTINATION / Path(__file__).name)
    caption = (
        "Stage-colored probability density distributions of all 15 saved geometric decoder features, "
        "using usable fits only and no imputation. Stage counts: Pre-TC 197, Post-TC 195, Pre-SC 196, "
        "Post-SC 198, Pre-GO 196, Post-GO 195, from 198 short-delay trials of Monkey T, session "
        "y070316009-12, channel 7. The 11 unusable fits are excluded from these descriptive plots, "
        "not from the original decoder. Each stage distribution integrates to approximately one. "
        "Gaussian density curves use one absolute bandwidth per feature, based on pooled standard "
        "deviation and mean stage sample size (Scott rule), shared across all stages. Reflection "
        "at physical boundaries prevents density being assigned to impossible values. Curves are "
        "descriptive smoothing, not additional fitted geometric models or significance tests.\n\n"
        "All data values are displayed in their original saved units; no observations are trimmed. "
        "Axis ranges vary between features, especially for the tightly concentrated orientation "
        "components. Orientations describe the lag-coordinate frame, not anatomical direction. "
        "Band half-width is (R1_out - R1_in + R2_out - R2_in)/4, not R1 - R2. Radii and half-width "
        "are in per-window-normalized signal units. Repeated stages belong to the same trials. "
        "No inference of significant separation is made from visual differences. No preprocessing, "
        "geometric fitting, feature extraction, or decoding was rerun, and existing files remain unchanged.\n"
    )
    (DESTINATION / "captions.md").write_text(caption)
    write_json(DESTINATION / "provenance.json", dict(
        source_sha256=sha256(__file__), protected_files=protected, previous_files_unchanged=True,
        features=ORDER, stage_counts=counts.tolist(), usable_windows=1177, excluded_windows=11,
        method="Gaussian KDE, shared absolute Scott bandwidth per feature, boundary reflection",
        grid_points=512, trimming=False, imputation=False, geometric_fitting_called=False, decoding_called=False,
        bandwidths={column: saved[column][2] for column in ORDER}, exports=exports,
        pdf=dict(path=str(pdf_path), pages=16, sha256=sha256(pdf_path))))
    print(f"Validated 15 features, 90 distributions, 16 PNGs and a 16-page vector PDF; protected {len(protected)} existing files.")


if __name__ == "__main__":
    main()
