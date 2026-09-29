"""Train-only 2D views of frozen corrected-stage geometry and Betti summaries."""
from __future__ import annotations

import argparse
import hashlib
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

UNIT = Path(__file__).resolve().parent
OUTPUT = UNIT / "outputs/stage_tau3_feature_projections_v1"
GEOMETRY = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
BETTI = (UNIT / "outputs/macaque_task_six_stages_cue_offset_correct"
         / "STAGE_BETTI_GEOMETRY_SEM_v1/tables/integrated_betti_per_window.csv")
BETTI_CONFIG = (UNIT / "outputs/macaque_task_six_stages_cue_offset_correct"
                / "BETTI_STAGE_v1/config.json")
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")
RADII = ("R1", "R2", "band_half_width")
QUALITY = ("mse", "mean_error", "frac_inside")
ORIENTATION = tuple(f"{prefix}_{axis}" for prefix in ("normal", "u", "v") for axis in "xyz")
BETTI_COLUMNS = ("beta0", "beta1", "beta2")
INK = "#27323A"
SEED = 42
SPECS = (
    ("geometry_all_15", "Geometry: all 15 features", "geometry_all"),
    ("geometry_radii_width_3", "Geometry: radii and width", "radii"),
    ("geometry_orientation_9", "Geometry: orientation", "orientation"),
    ("geometry_fit_quality_3", "Geometry: fit quality", "quality"),
    ("betti_raw_3", "Betti: raw distance", "betti_raw"),
    ("betti_rms_normalized_3", "Betti: RMS-normalized distance", "betti_rms"),
    ("geometry_plus_betti_raw_18", "Geometry + Betti: raw distance", "combined_raw"),
    ("geometry_plus_betti_rms_normalized_18", "Geometry + Betti: RMS-normalized", "combined_rms"),
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sources():
    return [GEOMETRY / "config.json", GEOMETRY / "features.npz",
            GEOMETRY / "tables/features.csv", BETTI, BETTI_CONFIG]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_frozen_data():
    config = json.loads((GEOMETRY / "config.json").read_text())
    betti_config = json.loads(BETTI_CONFIG.read_text())
    columns = tuple(config["feature_columns"])
    if (config["m"], config["tau_ms"], config["points_per_cloud"]) != (3, 3, 294):
        raise ValueError("Expected original native tau=3 geometry")
    if tuple(config["stage_names"]) != STAGES or tuple(config["stage_anchors"]) != (
            "TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO"):
        raise ValueError("Geometry is not from corrected cue-offset stages")
    if (betti_config["m"], betti_config["tau_ms"], betti_config["n_points"]) != (3, 3, 294):
        raise ValueError("Betti summaries are not from native tau=3 clouds")
    if len(columns) != 15 or set(columns) != set(RADII + QUALITY + ORIENTATION):
        raise ValueError("Unexpected geometric feature definition")
    geometry = pd.read_csv(GEOMETRY / "tables/features.csv", float_precision="round_trip")
    with np.load(GEOMETRY / "features.npz", allow_pickle=False) as saved:
        x = saved["x"].copy()
        labels = saved["labels"].copy()
        folds = saved["folds"].copy()
        trial_ids = saved["trial_ids"].copy()
    if (len(geometry) != 1188 or x.shape != (1188, 15)
            or not np.array_equal(geometry.window_index.to_numpy(), np.arange(1188))
            or not np.allclose(geometry[list(columns)].to_numpy(), x, rtol=0, atol=1e-12,
                               equal_nan=True)):
        raise ValueError("Geometry feature rows or values changed")
    for name, array in (("stage", labels), ("heldout_fold_zero_based", folds),
                        ("trial_index", trial_ids)):
        if not np.array_equal(geometry[name].to_numpy(), array):
            raise ValueError(f"Geometry {name} disagrees with frozen feature archive")
    if (not geometry.delay.eq("short").all() or geometry.trial_index.nunique() != 198
            or tuple(geometry.groupby("stage").size()) != (198,) * 6
            or set(np.unique(folds)) != set(range(5))
            or len(np.unique(trial_ids[folds == 0])) != 40
            or int(np.sum(folds == 0)) != 240
            or geometry.loc[geometry.stage_name.eq("post_TC"), "anchor_code"].ne(203).any()
            or geometry.loc[geometry.stage_name.eq("post_SC"), "anchor_code"].ne(206).any()):
        raise ValueError("Corrected stage cohort or held-out fold changed")
    for trial in np.unique(trial_ids):
        mask = trial_ids == trial
        if np.unique(folds[mask]).size != 1 or not np.array_equal(np.sort(labels[mask]), np.arange(6)):
            raise ValueError(f"Stage windows cross folds or are incomplete: trial {trial}")
    if not np.array_equal(geometry.stage_name.to_numpy(), np.asarray(STAGES)[labels]):
        raise ValueError("Unexpected stage labels")
    missing_rows = np.isnan(x).all(axis=1)
    if (missing_rows.sum() != 11 or not np.array_equal(missing_rows, ~geometry.usable.to_numpy())
            or np.isinf(x).any() or np.isnan(x[~missing_rows]).any()):
        raise ValueError("Geometry fit-failure pattern changed")

    betti = pd.read_csv(BETTI, float_precision="round_trip")
    if len(betti) != 2376 or set(betti.distance_convention) != {"raw", "rms_normalized"}:
        raise ValueError("Expected both complete saved Betti conventions")
    matrices = {}
    for convention in ("raw", "rms_normalized"):
        part = betti.loc[betti.distance_convention.eq(convention)].sort_values("window_index")
        if len(part) != 1188 or not np.array_equal(part.window_index, geometry.window_index):
            raise ValueError(f"Missing or duplicated Betti window: {convention}")
        for name in ("trial_index", "original_trial_number", "stage", "stage_name", "direction"):
            if not np.array_equal(part[name].to_numpy(), geometry[name].to_numpy()):
                raise ValueError(f"Betti/geometry {name} mismatch: {convention}")
        values = part[list(BETTI_COLUMNS)].to_numpy(dtype=float)
        if values.shape != (1188, 3) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid Betti values: {convention}")
        matrices[convention] = values
    return geometry, x, matrices, columns


def feature_sets(geometry_values, betti_values, geometry_columns):
    lookup = {name: i for i, name in enumerate(geometry_columns)}

    def select(names):
        return geometry_values[:, [lookup[name] for name in names]]

    raw = betti_values["raw"]
    rms = betti_values["rms_normalized"]
    return {
        "geometry_all": (geometry_values, geometry_columns),
        "radii": (select(RADII), RADII),
        "orientation": (select(ORIENTATION), ORIENTATION),
        "quality": (select(QUALITY), QUALITY),
        "betti_raw": (raw, BETTI_COLUMNS),
        "betti_rms": (rms, BETTI_COLUMNS),
        "combined_raw": (np.column_stack((geometry_values, raw)), geometry_columns + BETTI_COLUMNS),
        "combined_rms": (np.column_stack((geometry_values, rms)), geometry_columns + BETTI_COLUMNS),
    }


def training_preprocess(values, train):
    """Fit all preprocessing on the saved training trials, then transform both splits."""
    if values.ndim != 2 or train.shape != (len(values),) or np.isinf(values).any():
        raise ValueError("Invalid feature matrix or split")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    training = imputer.fit_transform(values[train])
    if training.shape[1] != values.shape[1]:
        raise ValueError("Training data lost a feature during imputation")
    scaler.fit(training)
    scaled = scaler.transform(imputer.transform(values))
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite scaled features")
    return scaled, imputer, scaler


def project(scaled, stage, train, method):
    if method == "pca":
        model = PCA(n_components=2, svd_solver="full")
        model.fit(scaled[train])
        xy = model.transform(scaled)
        details = {"explained_variance_ratio": model.explained_variance_ratio_.tolist()}
    elif method == "lda":
        model = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2)
        model.fit(scaled[train], stage[train])
        xy = model.transform(scaled)
        details = {"solver": "eigen", "shrinkage": "auto", "supervised": True}
    elif method == "umap":
        from umap import UMAP

        model = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                     metric="euclidean", random_state=SEED, transform_seed=SEED,
                     n_jobs=1)
        fitted = model.fit_transform(scaled[train])
        xy = np.empty((len(scaled), 2), dtype=float)
        xy[train] = fitted
        xy[~train] = model.transform(scaled[~train])
        details = {"n_neighbors": 30, "min_dist": 0.1, "metric": "euclidean",
                   "random_state": SEED, "transform_seed": SEED, "n_jobs": 1}
    else:
        raise ValueError(method)
    if xy.shape != (len(scaled), 2) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid {method} projection")
    return xy, details


def coordinate_table(geometry, xy):
    columns = ["window_index", "trial_index", "original_trial_number", "stage",
               "stage_name", "direction", "heldout_fold_zero_based"]
    result = geometry[columns].copy()
    result["split"] = np.where(geometry.heldout_fold_zero_based.eq(0), "heldout", "training")
    result["x"] = xy[:, 0]
    result["y"] = xy[:, 1]
    return result


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "text.color": INK, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "axes.grid": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def plot_one(table, method, title, details, destination):
    style()
    fig, ax = plt.subplots(figsize=(6.35, 4.6), facecolor="white")
    train = table.split.eq("training").to_numpy()
    stage = table.stage.to_numpy()
    for index, color in enumerate(COLORS):
        subset = train & (stage == index)
        ax.scatter(table.loc[subset, "x"], table.loc[subset, "y"], s=11,
                   facecolors="none", edgecolors=color, alpha=.28, linewidths=.65,
                   rasterized=True, zorder=1)
    for index, color in enumerate(COLORS):
        subset = ~train & (stage == index)
        ax.scatter(table.loc[subset, "x"], table.loc[subset, "y"], s=23,
                   facecolors=color, edgecolors=INK, alpha=.88, linewidths=.35,
                   rasterized=True, zorder=2)
    axis_names = {"pca": ("PC1", "PC2"), "lda": ("LD1", "LD2"),
                  "umap": ("UMAP1", "UMAP2")}
    names = list(axis_names[method])
    if method == "pca":
        variance = details["explained_variance_ratio"]
        names = [f"PC{i+1} ({100*value:.1f}% variance)" for i, value in enumerate(variance)]
    ax.set_xlabel(names[0], labelpad=5)
    ax.set_ylabel(names[1], labelpad=5)
    ax.set_title(f"{method.upper()}  |  {title}", loc="left", pad=8)
    ax.tick_params(direction="out", length=3, width=.7)
    ax.margins(x=.065, y=.07)
    for get_limits, get_ticks, set_ticks in ((ax.get_xlim, ax.get_xticks, ax.set_xticks),
                                              (ax.get_ylim, ax.get_yticks, ax.set_yticks)):
        low, high = get_limits()
        set_ticks([value for value in get_ticks() if low <= value <= high])
    stage_handles = [Line2D([0], [0], marker="o", linestyle="none", color="none",
                            markerfacecolor=color, markeredgecolor=INK,
                            markeredgewidth=.3, markersize=6.5, label=label)
                     for color, label in zip(COLORS, LABELS)]
    split_handles = [Line2D([0], [0], marker="o", linestyle="none", color="none",
                            markerfacecolor="none", markeredgecolor="#6A7580",
                            markersize=6, label="Training (948)"),
                     Line2D([0], [0], marker="o", linestyle="none", color="none",
                            markerfacecolor="#6A7580", markeredgecolor=INK,
                            markersize=6, label="Held out (240)")]
    first = fig.legend(handles=stage_handles, loc="upper center", ncol=3,
                       bbox_to_anchor=(.52, .935), frameon=False, fontsize=8.5,
                       columnspacing=1.8, handletextpad=.4)
    fig.add_artist(first)
    fig.legend(handles=split_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(.52, -.005), frameon=False, fontsize=8,
               columnspacing=2.0, handletextpad=.4)
    fig.subplots_adjust(left=.16, right=.94, top=.76, bottom=.18)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                                         or box.x1 > fig.bbox.width + 1
                                         or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Text clipped in {destination}: {text.get_text()} "
                             f"bbox={tuple(round(v, 1) for v in box.bounds)} "
                             f"canvas={fig.bbox.width:.1f}x{fig.bbox.height:.1f} "
                             f"position={text.get_position()} axes={text.axes}")
    fig.savefig(destination, dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(destination) as image:
        rgb = np.asarray(image.convert("RGB"))
        dpi = image.info.get("dpi", (0, 0))
        if (image.size != (3810, 2760) or not all(abs(value - 600) < 1 for value in dpi)
                or not np.any(rgb < 240)):
            raise ValueError(f"Blank or wrong-size PNG: {destination}")
    return dict(file=str(destination.relative_to(OUTPUT)), sha256=digest(destination),
                width_px=3810, height_px=2760)


def validate_coordinates(table, geometry):
    if len(table) != 1188 or not np.array_equal(table.window_index, geometry.window_index):
        raise ValueError("Projection coordinates lost windows")
    for name in ("trial_index", "original_trial_number", "stage", "stage_name",
                 "direction", "heldout_fold_zero_based"):
        if not np.array_equal(table[name].to_numpy(), geometry[name].to_numpy()):
            raise ValueError(f"Projection identity mismatch: {name}")
    if table.split.value_counts().to_dict() != {"training": 948, "heldout": 240}:
        raise ValueError("Projection split changed")
    if not np.isfinite(table[["x", "y"]].to_numpy()).all():
        raise ValueError("Projection contains nonfinite coordinates")


def run_compute(geometry, sets):
    train = geometry.heldout_fold_zero_based.ne(0).to_numpy()
    stage = geometry.stage.to_numpy()
    index = []
    for folder, title, key in SPECS:
        values, columns = sets[key]
        if values.shape != (1188, len(columns)):
            raise ValueError(f"Unexpected feature width in {folder}")
        destination = OUTPUT / folder
        destination.mkdir(parents=True, exist_ok=True)
        scaled, imputer, scaler = training_preprocess(values, train)
        methods = {}
        for method in ("pca", "lda", "umap"):
            xy, details = project(scaled, stage, train, method)
            table = coordinate_table(geometry, xy)
            validate_coordinates(table, geometry)
            table.to_csv(destination / f"{method}.csv", index=False, float_format="%.17g")
            methods[method] = details
            print(f"Computed {folder}/{method}", flush=True)
        missing_rows = int(np.isnan(values).any(axis=1).sum())
        save_json(destination / "methods.json", dict(
            title=title, columns=list(columns), dimensions=len(columns),
            missing_rows_imputed=missing_rows, training_windows=948, heldout_windows=240,
            imputation="training median", scaling="training StandardScaler",
            imputer_statistics=imputer.statistics_.tolist(),
            scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
            methods=methods))
        index.append(dict(folder=folder, title=title, dimensions=len(columns),
                          feature_columns=";".join(columns), missing_rows_imputed=missing_rows))
    pd.DataFrame(index).to_csv(OUTPUT / "feature_sets.csv", index=False)


def run_plot(geometry):
    manifest = []
    for folder, title, _ in SPECS:
        destination = OUTPUT / folder
        details = json.loads((destination / "methods.json").read_text())
        if details["title"] != title:
            raise ValueError(f"Figure title mismatch: {folder}")
        for method in ("pca", "lda", "umap"):
            table = pd.read_csv(destination / f"{method}.csv", float_precision="round_trip")
            validate_coordinates(table, geometry)
            manifest.append(plot_one(table, method, title, details["methods"][method],
                                     destination / f"{method}.png"))
            print(f"Rendered {folder}/{method}.png", flush=True)
    if len(manifest) != 24:
        raise ValueError("Expected 24 standalone projection figures")
    save_json(OUTPUT / "figure_manifest.json", manifest)
    thumbnail_width, thumbnail_height = 580, 425
    sheet = Image.new("RGB", (3 * thumbnail_width, 8 * thumbnail_height), "white")
    drawer = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for row, (folder, _, _) in enumerate(SPECS):
        for column, method in enumerate(("pca", "lda", "umap")):
            with Image.open(OUTPUT / folder / f"{method}.png") as image:
                thumbnail = image.copy()
            thumbnail.thumbnail((thumbnail_width - 12, thumbnail_height - 40), Image.Resampling.LANCZOS)
            x = column * thumbnail_width + (thumbnail_width - thumbnail.width) // 2
            y = row * thumbnail_height + 30
            sheet.paste(thumbnail, (x, y))
            drawer.text((column * thumbnail_width + 8, row * thumbnail_height + 6),
                        f"{folder} / {method}", font=font, fill=INK)
    inspection = OUTPUT / "inspection"
    inspection.mkdir(exist_ok=True)
    sheet.save(inspection / "contact_sheet.png")
    save_json(OUTPUT / "validation.json", dict(
        passed=True, n_windows=1188, n_trials=198, n_training_windows=948,
        n_heldout_windows=240, n_feature_sets=8, n_figures=24,
        all_coordinate_tables_checked=True, all_pngs_checked=True,
        geometric_fitting_calls=0, persistent_homology_calls=0, decoding_calls=0))
    links = "\n".join(
        f"- {title}: [PCA]({folder}/pca.png) | [LDA]({folder}/lda.png) | "
        f"[UMAP]({folder}/umap.png)" for folder, title, _ in SPECS)
    (OUTPUT / "README.md").write_text(
        "# Six-stage feature projections (native tau = 3 ms)\n\n"
        "Monkey T, y070316009-12, channel 7. All 198 short-delay trials "
        "contribute six corrected 300-ms stages each. Each figure shows one point "
        "per trial-stage window: open faint points are the 948 training windows, "
        "and filled points are the 240 held-out windows from saved fold 0. "
        "The same split, stage colors, and training-only median imputation/scaling "
        "are used in every panel. The 11 unusable geometry fits are imputed, not dropped.\n\n"
        + links + "\n\n"
        "PCA and UMAP do not use stage labels during fitting. LDA uses the six training-stage "
        "labels, so training separation is not independent evidence of decoding. "
        "Held-out points are transformed without refitting. The displayed axes "
        "are method- and feature-set-specific; do not compare coordinate distances across "
        "figures. Betti values are the saved exact distance-averaged counts, with raw "
        "and RMS-normalized distance conventions kept separate. No geometry, Ripser, "
        "or decoder was rerun. Coordinate CSVs and method details accompany each PNG.\n\n"
        "[Qualitative interpretation](INTERPRETATION.md) | "
        "[All-panel inspection sheet](inspection/contact_sheet.png) | "
        "[Validation](validation.json)\n")
    (OUTPUT / "INTERPRETATION.md").write_text(
        "# Qualitative reading\n\n"
        "The six stage colors overlap substantially in the 2D PCA, LDA, and UMAP "
        "views of the geometric, Betti, and combined feature sets. Some plots show "
        "continuous structure or local concentrations, but none shows six distinct "
        "held-out stage clusters. The supervised LDA views do not reverse this pattern. "
        "These are descriptive projections, not a new decoding test or a significance "
        "analysis; 2D overlap also cannot rule out information in the full feature space.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("compute", "plot", "all"), default="all")
    args = parser.parse_args()
    source_hashes = {str(path.resolve()): digest(path) for path in sources()}
    geometry, x, betti, columns = load_frozen_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = dict(recording="monkeyT_session-y070316009-12_lfp-7",
                  stages=list(STAGES), native_tau_ms=3, m=3, source_hashes=source_hashes,
                  fold_for_heldout=0, seed=SEED, feature_sets=[row[0] for row in SPECS],
                  point_definition="one saved 300-ms trial-stage window",
                  betti_definition="saved distance-averaged beta0/beta1/beta2 per window")
    path = OUTPUT / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing projection configuration differs")
    save_json(path, config)
    if args.phase in ("compute", "all"):
        run_compute(geometry, feature_sets(x, betti, columns))
        import scipy
        import sklearn
        import umap

        save_json(OUTPUT / "environment.json", dict(
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            scipy=scipy.__version__, sklearn=sklearn.__version__,
            matplotlib=matplotlib.__version__, umap=umap.__version__))
    if args.phase in ("plot", "all"):
        run_plot(geometry)
    if source_hashes != {str(path.resolve()): digest(path) for path in sources()}:
        raise AssertionError("A frozen source file changed")
    print(f"Verified unchanged sources and {args.phase} outputs in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
