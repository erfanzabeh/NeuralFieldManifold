"""Rectangular stage contrasts from frozen Betti curves; no analysis reruns."""
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
from run_stage_persistent_homology import OUTPUT as SOURCE, STAGES, LABELS, digest, write_json, verify_manifest

OUTPUT = SOURCE / "rectangular_heatmaps_v1"
MODES = ("raw", "rms_normalized")


def deviation_table(curves, grid, metadata, mode):
    curves, grid = np.asarray(curves), np.asarray(grid)
    if curves.shape != (len(metadata), len(grid)) or len(grid) < 2:
        raise ValueError("Curve dimensions do not match the metadata/grid.")
    if not np.array_equal(metadata.window_index, np.arange(len(metadata))):
        raise ValueError("Window metadata must match the saved curve row order.")
    if (not metadata.status.eq("success").all() or not np.isfinite(curves).all()
            or (curves < 0).any() or not np.equal(curves, np.floor(curves)).all()):
        raise ValueError("Expected successful, finite nonnegative integer Betti counts.")
    if not np.isfinite(grid).all() or grid[0] != 0 or not (np.diff(grid) > 0).all():
        raise ValueError("Expected an increasing common distance grid starting at zero.")
    if set(metadata.stage) != set(range(6)):
        raise ValueError("Expected exactly the six task stages.")
    if metadata.duplicated(["trial_index", "stage"]).any():
        raise ValueError("Duplicate trial/stage observations.")
    if not all(row.stage_name == STAGES[row.stage] for row in metadata.itertuples()):
        raise ValueError("Stage identities do not match their saved indices.")
    # Pool individual windows, not independently rescaled rows or stage means.
    pooled = curves.mean(axis=0)
    rows = []
    for stage, name in enumerate(STAGES):
        group = curves[metadata.stage.to_numpy() == stage]
        means = group.mean(axis=0)
        rows.append(pd.DataFrame(dict(distance_convention=mode, stage=stage,
            stage_name=name, grid_index=np.arange(len(grid)), distance=grid,
            n_windows=len(group), mean_betti=means, pooled_mean_betti=pooled,
            delta_betti=means-pooled)))
    return pd.concat(rows, ignore_index=True)


def load_tables():
    metadata = pd.read_csv(SOURCE / "tables/window_measurements.csv", float_precision="round_trip")
    if len(metadata) != 1188 or metadata.trial_index.nunique() != 198:
        raise ValueError("Expected the frozen 198-trial, 1188-window cohort.")
    if not metadata.groupby("trial_index").stage.nunique().eq(6).all():
        raise ValueError("Every trial must retain all six stages.")
    if not metadata.groupby(["stage", "direction"]).size().eq(33).all():
        raise ValueError("Expected 33 trials per reach direction at each stage.")
    summary = pd.read_csv(SOURCE / "tables/betti_summary.csv", float_precision="round_trip")
    tables = {}
    with np.load(SOURCE / "tables/betti_curves.npz", allow_pickle=False) as saved:
        for mode in MODES:
            curves, grid = saved[f"{mode}_H1"], saved[f"{mode}_H1_grid"]
            if curves.shape != (1188, 512):
                raise ValueError("Expected the original 512-point grid, without resampling.")
            table = deviation_table(curves, grid, metadata, mode)
            reference = summary[(summary.cohort == "available") & (summary.homology_dimension == 1)
                & (summary.distance_convention == mode)].sort_values(["stage", "grid_index"])
            np.testing.assert_array_equal(table.stage, reference.stage)
            np.testing.assert_array_equal(table.grid_index, reference.grid_index)
            np.testing.assert_array_equal(table.n_windows, reference.n_windows)
            np.testing.assert_allclose(table.distance, reference.distance, rtol=0, atol=1e-14)
            np.testing.assert_allclose(table.mean_betti, reference["mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(table.groupby("grid_index").delta_betti.mean(), 0, atol=1e-12)
            tables[mode] = table
    return tables


def heatmap_panel(table, limit, zoom=False):
    matrix = table.pivot(index="stage", columns="distance", values="delta_betti").sort_index()
    if np.abs(matrix.to_numpy()).max() > limit:
        raise ValueError("Color scale must not clip any saved contrast.")
    mode = table.distance_convention.unique()
    if len(mode) != 1:
        raise ValueError("One distance convention per panel.")
    fig, ax = plt.subplots(figsize=(5.3, 3.15))
    fig.subplots_adjust(left=.17, right=.79, bottom=.20, top=.96)
    mesh = ax.pcolormesh(matrix.columns.to_numpy(), np.arange(6), matrix.to_numpy(),
        shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit,
        edgecolors="none", antialiased=False)
    ax.set(xlabel="Distance threshold" if mode[0] == "raw" else "Distance / RMS radius",
        ylabel="Task stage", yticks=np.arange(6), yticklabels=LABELS,
        xlim=(0, min(1., matrix.columns.max()) if zoom else matrix.columns.max()), ylim=(5.5, -.5))
    ax.tick_params(axis="y", length=0, pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    color_ax = fig.add_axes([.83, .20, .025, .76])
    bar = fig.colorbar(mesh, cax=color_ax, ticks=np.linspace(-limit, limit, 5))
    bar.set_label(r"$\Delta\beta_1$ (mean count)", fontsize=9, labelpad=8)
    bar.outline.set_linewidth(.6)
    return fig, ax, mesh


def save_panel(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    off_view = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    off_view.update((tick.label1, tick.label2))
    for text in fig.findobj(matplotlib.text.Text):
        if text in off_view or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, text.get_text()
    fig.savefig(path, dpi=600, metadata={"Software": "NeuralFieldManifold saved Betti heatmap renderer"})
    plt.close(fig)
    with Image.open(path) as im:
        dpi, size = im.info["dpi"], list(im.size)
        assert all(abs(d-600) < .1 for d in dpi)
        assert im.width >= 1800 and im.height >= 1800
        rgb = np.asarray(im.convert("RGB"))
        assert np.any(rgb < 240)
        assert all(np.all(edge > 245) for edge in (rgb[:3], rgb[-3:], rgb[:, :3], rgb[:, -3:]))
    return dict(file=path.name, sha256=digest(path), pixels=size, dpi=list(dpi), outside_text=[])


def main():
    style()
    verify_manifest(SOURCE / "provenance/input_source_hashes.json")
    verify_manifest(SOURCE / "provenance/table_hashes.json")
    protected = {str(p): digest(p) for p in SOURCE.rglob("*")
                 if p.is_file() and not p.is_relative_to(OUTPUT)}
    tables = load_tables()
    limit = max(1., float(np.ceil(max(t.delta_betti.abs().max() for t in tables.values()))))
    for folder in ("tables", "captions", "source", "validation"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    manifests = []
    for mode, table in tables.items():
        table.to_csv(OUTPUT / "tables" / f"betti_h1_stage_deviations_{mode}.csv", index=False)
        for zoom in (False, True):
            name = f"betti_h1_stage_deviation_{mode}" + ("_zoom" if zoom else "")
            fig, ax, mesh = heatmap_panel(table, limit, zoom=zoom)
            manifest = save_panel(fig, OUTPUT / f"{name}.png")
            manifest.update(distance_convention=mode, distance_limits=list(ax.get_xlim()),
                color_limits=list(mesh.get_clim()), zoom=zoom)
            manifests.append(manifest)
            caption = (
                "Monkey T, y070316009-12, channel 7; 198 short-delay trials per stage. "
                "Corrected 300-ms windows; post-TC begins at tone offset, post-SC at spatial-instruction end. "
                "Each cell is the stage mean beta1 minus the mean over all 1188 windows at the same distance. "
                "Red means more loops, blue fewer. No row scaling, smoothing, or optimized distance selection. "
                f"Distance convention: {mode}. All panels share a symmetric +/-{limit:g}-count color scale. "
                "RMS-normalized distances use each cloud's centered RMS radius, not another Ripser run. "
                "This mean-contrast view does not show trial variability or establish statistical significance. "
                "Refer to the original mean/IQR curves and lifetime distributions. "
                + ("Display zoom 0-1 only; data and color scale unchanged." if zoom else "Full saved distance range."))
            (OUTPUT / "captions" / f"{name}.txt").write_text(caption + "\n")
    links = "\n".join(f"- [{m['file']}]({m['file']})" for m in manifests)
    (OUTPUT / "README.md").write_text(f"""# Six-Stage Betti Heatmaps

{links}

Rows are task stages, not reach directions. All 198 trials contribute to every row.
Color is stage mean beta1 minus the pooled mean across all 1,188 windows at the
same distance: red = more loops; blue = fewer. The reference is not a per-stage
mean across distances. Values are count differences, not z-scores or percentages.

Raw and RMS-normalized panels share a +/-{limit:g}-count scale. The zoom panels
show 0-1 on the corresponding distance axis, matching the earlier rectangular
display; no distance was selected to maximize stage contrast. Full-range panels
retain all 512 saved distances. There is no interpolation or smoothing.

These descriptive mean contrasts do not show variability or establish stage
separation. The original trial distributions overlap. Post-TC starts after tone
offset; post-SC starts after the spatial instruction ends.

Saved results only: no Ripser, geometry fitting, preprocessing, or decoding rerun.
All previous results remain unchanged. Supporting CSVs reproduce every cell;
captions, source, and validation are in their respective subfolders.
""")
    for path, expected in protected.items():
        assert digest(path) == expected, f"Protected PH result changed: {path}"
    prior_count = verify_manifest(SOURCE / "provenance/protected_hashes.json")
    assert "ripser" not in sys.modules
    assert not any(name.startswith("NeuralFieldManifold.fits") for name in sys.modules)
    shutil.copy2(__file__, OUTPUT / "source" / Path(__file__).name)
    write_json(OUTPUT / "validation/protected_ph_hashes.json", protected)
    write_json(OUTPUT / "validation/plot_checks.json", dict(passed=True, n_trials=198,
        n_windows=1188, windows_per_stage=198, grid_points=512, homology_dimension=1,
        source_sha256=digest(__file__), shared_color_limit=limit,
        prior_files_unchanged=prior_count, ph_files_unchanged=len(protected),
        imported_ripser=False, fitter_calls=0, decoder_calls=0, panels=manifests))
    print(json.dumps(dict(folder=str(OUTPUT), pngs=len(manifests), validated=True), indent=2))


if __name__ == "__main__":
    main()
