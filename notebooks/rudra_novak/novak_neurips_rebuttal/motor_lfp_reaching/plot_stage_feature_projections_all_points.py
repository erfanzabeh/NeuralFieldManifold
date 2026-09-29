"""Descriptive 2D projections of all corrected-stage feature windows."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import plot_stage_feature_projections as frozen

OUTPUT = frozen.UNIT / "outputs/stage_tau3_feature_projections_all_points_v2"
METHODS = ("pca", "lda", "umap")


def preprocess(values):
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError("Invalid feature matrix")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(values)
    if imputed.shape != values.shape:
        raise ValueError("Imputation lost a feature")
    scaled = scaler.fit_transform(imputed)
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite scaled features")
    return scaled, imputer, scaler


def project(scaled, stages, method):
    if method == "pca":
        model = PCA(n_components=2, svd_solver="full")
        xy = model.fit_transform(scaled)
        details = {"explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                   "uses_stage_labels": False}
    elif method == "lda":
        model = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2)
        xy = model.fit(scaled, stages).transform(scaled)
        details = {"solver": "eigen", "shrinkage": "auto", "uses_stage_labels": True}
    elif method == "umap":
        from umap import UMAP

        model = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                     metric="euclidean", random_state=frozen.SEED, n_jobs=1)
        xy = model.fit_transform(scaled)
        details = {"n_neighbors": 30, "min_dist": 0.1, "metric": "euclidean",
                   "random_state": frozen.SEED, "n_jobs": 1, "uses_stage_labels": False}
    else:
        raise ValueError(method)
    if xy.shape != (len(scaled), 2) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid {method} coordinates")
    return xy, details


def coordinate_table(geometry, xy):
    columns = ["window_index", "trial_index", "original_trial_number", "stage",
               "stage_name", "direction"]
    table = geometry[columns].copy()
    table["x"] = xy[:, 0]
    table["y"] = xy[:, 1]
    return table


def validate_coordinates(table, geometry):
    if len(table) != 1188:
        raise ValueError("Projection lost windows")
    for name in ("window_index", "trial_index", "original_trial_number", "stage",
                 "stage_name", "direction"):
        if not np.array_equal(table[name].to_numpy(), geometry[name].to_numpy()):
            raise ValueError(f"Projection identity mismatch: {name}")
    if not np.isfinite(table[["x", "y"]].to_numpy()).all():
        raise ValueError("Projection contains nonfinite coordinates")
    if table.groupby("stage").size().tolist() != [198] * 6:
        raise ValueError("Stage counts changed")


def plot_one(table, method, title, details, destination):
    frozen.style()
    fig, ax = plt.subplots(figsize=(6.35, 4.6), facecolor="white")
    for index, color in enumerate(frozen.COLORS):
        subset = table.stage.eq(index)
        ax.scatter(table.loc[subset, "x"], table.loc[subset, "y"],
                   s=15, c=color, alpha=.63, edgecolors="none", rasterized=True)
    axes = {"pca": ("PC1", "PC2"), "lda": ("LD1", "LD2"),
            "umap": ("UMAP1", "UMAP2")}[method]
    if method == "pca":
        axes = tuple(f"PC{i + 1} ({100 * fraction:.1f}% variance)"
                     for i, fraction in enumerate(details["explained_variance_ratio"]))
    label = "LDA (supervised)" if method == "lda" else method.upper()
    ax.set(xlabel=axes[0], ylabel=axes[1])
    ax.set_title(f"{label}  |  {title}", loc="left", pad=9)
    ax.tick_params(direction="out", length=3, width=.7)
    ax.margins(x=.065, y=.07)
    for get_limits, get_ticks, set_ticks in (
            (ax.get_xlim, ax.get_xticks, ax.set_xticks),
            (ax.get_ylim, ax.get_yticks, ax.set_yticks)):
        low, high = get_limits()
        set_ticks([value for value in get_ticks() if low < value < high])
    handles = [Line2D([0], [0], marker="o", linestyle="none", color="none",
                      markerfacecolor=color, markeredgecolor="none", markersize=6,
                      label=stage)
               for color, stage in zip(frozen.COLORS, frozen.LABELS)]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(.52, .945), frameon=False, fontsize=8.5,
               columnspacing=1.8, handletextpad=.4)
    fig.subplots_adjust(left=.16, right=.95, top=.77, bottom=.15)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                                         or box.x1 > fig.bbox.width + 1
                                         or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Clipped text in {destination}: {item.get_text()}")
    fig.savefig(destination, dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(destination) as image:
        pixels = np.asarray(image.convert("RGB"))
        dpi = image.info.get("dpi", (0, 0))
        if (image.size != (3810, 2760) or not all(abs(value - 600) < 1 for value in dpi)
                or not np.any(pixels < 240)):
            raise ValueError(f"Blank or wrong-size PNG: {destination}")
    return {"file": str(destination.relative_to(OUTPUT)), "sha256": frozen.digest(destination)}


def run_compute(geometry, sets):
    stages = geometry.stage.to_numpy()
    feature_index = []
    for folder, title, key in frozen.SPECS:
        values, columns = sets[key]
        if values.shape != (1188, len(columns)):
            raise ValueError(f"Unexpected feature width: {folder}")
        destination = OUTPUT / folder
        destination.mkdir(parents=True, exist_ok=True)
        scaled, imputer, scaler = preprocess(values)
        methods = {}
        for method in METHODS:
            xy, details = project(scaled, stages, method)
            table = coordinate_table(geometry, xy)
            validate_coordinates(table, geometry)
            table.to_csv(destination / f"{method}.csv", index=False, float_format="%.17g")
            methods[method] = details
            print(f"Computed {folder}/{method}", flush=True)
        missing = int(np.isnan(values).any(axis=1).sum())
        frozen.save_json(destination / "methods.json", {
            "title": title, "columns": list(columns), "dimensions": len(columns),
            "n_windows": 1188, "missing_rows_imputed": missing,
            "imputation": "median fitted on all 1188 windows",
            "scaling": "StandardScaler fitted on all 1188 windows",
            "imputer_statistics": imputer.statistics_.tolist(),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "methods": methods})
        feature_index.append({"folder": folder, "title": title, "dimensions": len(columns),
                              "feature_columns": ";".join(columns), "missing_rows_imputed": missing})
    pd.DataFrame(feature_index).to_csv(OUTPUT / "feature_sets.csv", index=False)


def run_plot(geometry):
    manifest = []
    for folder, title, _ in frozen.SPECS:
        destination = OUTPUT / folder
        metadata = json.loads((destination / "methods.json").read_text())
        if metadata["title"] != title or metadata["n_windows"] != 1188:
            raise ValueError(f"Wrong figure metadata: {folder}")
        for method in METHODS:
            table = pd.read_csv(destination / f"{method}.csv", float_precision="round_trip")
            validate_coordinates(table, geometry)
            manifest.append(plot_one(table, method, title, metadata["methods"][method],
                                     destination / f"{method}.png"))
            print(f"Rendered {folder}/{method}.png", flush=True)
    if len(manifest) != 24:
        raise ValueError("Expected 24 figures")
    frozen.save_json(OUTPUT / "figure_manifest.json", manifest)
    cell_width, cell_height = 580, 425
    sheet = Image.new("RGB", (3 * cell_width, 8 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for row, (folder, _, _) in enumerate(frozen.SPECS):
        for column, method in enumerate(METHODS):
            with Image.open(OUTPUT / folder / f"{method}.png") as source:
                thumbnail = source.copy()
            thumbnail.thumbnail((cell_width - 12, cell_height - 40), Image.Resampling.LANCZOS)
            x = column * cell_width + (cell_width - thumbnail.width) // 2
            y = row * cell_height + 30
            sheet.paste(thumbnail, (x, y))
            draw.text((column * cell_width + 8, row * cell_height + 6),
                      f"{folder} / {method}", font=font, fill=frozen.INK)
    inspection = OUTPUT / "inspection"
    inspection.mkdir(exist_ok=True)
    sheet.save(inspection / "contact_sheet.png")
    frozen.save_json(OUTPUT / "validation.json", {
        "passed": True, "n_windows": 1188, "n_trials": 198, "n_feature_sets": 8,
        "n_figures": 24, "all_points_used_to_fit_and_display_each_projection": True,
        "all_coordinate_tables_checked": True, "all_pngs_checked": True,
        "geometric_fitting_calls": 0, "persistent_homology_calls": 0, "decoding_calls": 0})
    links = "\n".join(f"- {title}: [PCA]({folder}/pca.png) | [LDA]({folder}/lda.png) | "
                      f"[UMAP]({folder}/umap.png)" for folder, title, _ in frozen.SPECS)
    (OUTPUT / "README.md").write_text(
        "# Six-stage feature projections: all points\n\n"
        "Monkey T, y070316009-12, channel 7; native tau = 3 ms. Each panel contains "
        "all 1,188 corrected 300-ms trial-stage windows (198 per stage), using one "
        "identical point style and color only for stage. There is no train/test marker "
        "distinction in these descriptive views. Median imputation and standardization "
        "are fitted on all windows; 11 unusable geometry fits are imputed, not dropped.\n\n"
        + links + "\n\n"
        "PCA and UMAP use feature values only. LDA uses stage labels to fit its axes "
        "and is explicitly supervised; apparent stage separation is not held-out "
        "decoding evidence. These are descriptive projections, not a new decoder. "
        "Axes and coordinate distances are specific to each plot. Betti values are "
        "the saved distance-averaged counts, with raw and RMS-normalized conventions "
        "kept separate. No geometry, Ripser, or decoder was rerun.\n\n"
        "[All-panel inspection sheet](inspection/contact_sheet.png) | "
        "[Validation](validation.json)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("compute", "plot", "all"), default="all")
    args = parser.parse_args()
    source_hashes = {str(path.resolve()): frozen.digest(path) for path in frozen.sources()}
    geometry, values, betti, columns = frozen.load_frozen_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = {"recording": "monkeyT_session-y070316009-12_lfp-7",
              "stages": list(frozen.STAGES), "native_tau_ms": 3, "m": 3,
              "point_definition": "one corrected 300-ms trial-stage window",
              "n_points": 1188, "projection_fit": "all windows", "display": "uniform point style",
              "seed": frozen.SEED, "feature_sets": [row[0] for row in frozen.SPECS],
              "source_hashes": source_hashes}
    config_path = OUTPUT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing all-points projection configuration differs")
    frozen.save_json(config_path, config)
    if args.phase in ("compute", "all"):
        run_compute(geometry, frozen.feature_sets(values, betti, columns))
        import scipy
        import sklearn
        import umap

        frozen.save_json(OUTPUT / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "umap": umap.__version__})
    if args.phase in ("plot", "all"):
        run_plot(geometry)
    if source_hashes != {str(path.resolve()): frozen.digest(path) for path in frozen.sources()}:
        raise AssertionError("Frozen source file changed")
    print(f"Verified unchanged inputs and {args.phase} outputs in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
