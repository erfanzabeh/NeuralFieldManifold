"""Six-stage circular histograms from saved, usable geometric features only."""
from __future__ import annotations

from pathlib import Path
import shutil

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from plot_stage_geometry_decoding import INK, save, style
from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, COLORS, LABELS, STAGES

DESTINATION = OUTPUT / "radial_feature_distributions_v1"
FEATURES = ("R1", "R2", "band_half_width")
TITLES = ("Outer radius R1", "Outer radius R2", "Band half-width")
N_BINS = 20
CMAP = LinearSegmentedColormap.from_list(
    "trial_fraction", ["#FFFAF7", "#F0C1B4", "#D57660", "#A02B23", "#65150F"])


def stage_histograms(table, feature, edges):
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must increase")
    counts, sizes = [], []
    for stage in range(6):
        values = table.loc[table.stage == stage, feature].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("All stages must have finite observations")
        if np.any(values < edges[0]) or np.any(values > edges[-1]):
            raise ValueError("Bins must retain every observation")
        count, _ = np.histogram(values, bins=edges)
        assert count.sum() == len(values)
        counts.append(count)
        sizes.append(len(values))
    counts, sizes = np.asarray(counts), np.asarray(sizes)
    fractions = counts / sizes[:, None]
    np.testing.assert_allclose(fractions.sum(axis=1), np.ones(6))
    return counts, sizes, fractions


def radial_axis(ax, edges, fractions, title, norm):
    maximum = edges[-1]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    centers = np.arange(6) * np.pi / 3
    width = np.pi / 3 - np.deg2rad(1.5)
    for stage, theta in enumerate(centers):
        bars = ax.bar(np.full(N_BINS, theta), np.diff(edges), bottom=edges[:-1], width=width,
                      color=CMAP(norm(fractions[stage])), edgecolor="none", linewidth=0,
                      align="center", zorder=1)
        assert len(bars) == N_BINS
        ax.bar(theta, maximum * .024, bottom=maximum * 1.015, width=width,
               color=COLORS[stage], edgecolor="none", linewidth=0, zorder=2)
    ax.set_ylim(0, maximum * 1.04)
    ax.set_xticks(centers, LABELS)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_color(INK)
    ticks = MaxNLocator(3).tick_values(0, maximum)
    ticks = ticks[(ticks > 0) & (ticks <= maximum)]
    ticks = np.unique(np.r_[ticks, maximum])
    ax.set_yticks(ticks, [f"{value:g}" for value in ticks])
    ax.set_rlabel_position(30)
    ax.tick_params(axis="y", length=0, pad=1, labelsize=8)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=.9, pad=.5))
    ax.text(0, 0, "0", ha="center", va="center", fontsize=8, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.set_title(title, y=1.18, fontsize=12, pad=0)


def colorbar(fig, rect, norm):
    ax = fig.add_axes(rect)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=ax,
                       ticks=np.linspace(0, norm.vmax, 4))
    bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    bar.set_label("Trials per bin", labelpad=7, fontsize=9)
    bar.ax.tick_params(labelsize=8, length=3)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    folder = DESTINATION / "png"
    folder.mkdir(exist_ok=True)
    protected = {str(p): sha256(p) for p in sorted(OUTPUT.rglob("*"))
                 if p.is_file() and not p.is_relative_to(DESTINATION)}
    original = pd.read_csv(OUTPUT / "tables/features.csv")
    assert len(original) == 1188 and original.original_trial_number.nunique() == 198
    table = original.loc[original.usable].copy()
    assert len(table) == 1177
    records, saved = [], {}
    for feature in FEATURES:
        maximum = float(table[feature].max())
        # Round upward to two significant digits, using pooled values without stage labels.
        step = 10 ** (np.floor(np.log10(maximum)) - 1)
        upper = float(np.ceil(maximum / step) * step)
        edges = np.linspace(0, upper, N_BINS + 1)
        counts, sizes, fractions = stage_histograms(table, feature, edges)
        np.testing.assert_array_equal(sizes, [197, 195, 196, 198, 196, 195])
        saved[feature] = (edges, counts, sizes, fractions)
        for stage in range(6):
            for b in range(N_BINS):
                records.append(dict(feature=feature, stage=stage, stage_name=STAGES[stage],
                                    bin_index=b, bin_lower=edges[b], bin_upper=edges[b+1],
                                    count=int(counts[stage, b]), stage_n=int(sizes[stage]),
                                    fraction=float(fractions[stage, b])))
    vmax = float(np.ceil(max(saved[f][3].max() for f in FEATURES) / .05) * .05)
    norm = Normalize(vmin=0, vmax=vmax, clip=False)
    pd.DataFrame(records).to_csv(DESTINATION / "radial_histograms.csv", index=False)
    table[["original_trial_number", "window_index", "stage", "stage_name", *FEATURES]].to_csv(
        DESTINATION / "plotted_feature_values.csv", index=False)
    style()
    plt.rcParams.update({"pdf.fonttype": 42, "pdf.compression": 6})
    exports = []
    pdf_path = DESTINATION / "radial_stage_distributions.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Geometric feature distributions across six task stages"}) as pdf:
        fig = plt.figure(figsize=(12.0, 4.7))
        for i, (feature, title) in enumerate(zip(FEATURES, TITLES)):
            x = .045 + i * .29
            ax = fig.add_axes([x, .22, .245, .62], projection="polar")
            radial_axis(ax, saved[feature][0], saved[feature][3], title, norm)
            fig.text(x + .1225, .09, "Normalized signal units", ha="center", fontsize=9)
        colorbar(fig, [.927, .28, .011, .43], norm)
        pdf.savefig(fig)
        save(fig, folder, "radial_stage_distributions", exports)
        for feature, title in zip(FEATURES, TITLES):
            fig = plt.figure(figsize=(5.3, 5.1))
            ax = fig.add_axes([.13, .23, .64, .60], projection="polar")
            radial_axis(ax, saved[feature][0], saved[feature][3], title, norm)
            fig.text(.45, .095, "Normalized signal units", ha="center", fontsize=9)
            colorbar(fig, [.85, .30, .024, .42], norm)
            pdf.savefig(fig)
            save(fig, folder, f"radial_{feature}", exports)
    outside = {e["file"]: e["outside_text"] for e in exports if e["outside_text"]}
    assert len(exports) == 4 and not outside, outside
    with fitz.open(pdf_path) as document:
        assert len(document) == 4 and not any(p.get_images() for p in document)
        document.set_toc([[1, "All three features", 1]] + [[1, t, i+2] for i, t in enumerate(TITLES)])
        document.saveIncr()
    restored = pd.read_csv(DESTINATION / "radial_histograms.csv")
    for feature in FEATURES:
        records = restored[restored.feature == feature]
        np.testing.assert_allclose(records.fraction.to_numpy().reshape(6, N_BINS), saved[feature][3])
        np.testing.assert_array_equal(records["count"].to_numpy().reshape(6, N_BINS), saved[feature][1])
    assert protected == {p: sha256(p) for p in protected}, "Existing outputs changed"
    shutil.copy2(__file__, DESTINATION / Path(__file__).name)
    (DESTINATION / "caption.md").write_text(
        "Circular histograms of saved outer radii R1/R2 and annular band half-width for six task "
        "stages. Equal wedges proceed clockwise from Pre-TC at the top: Pre-TC, Post-TC, Pre-SC, "
        "Post-SC, Pre-GO, Post-GO. Radial position encodes feature value, not geometric position "
        "on the fitted torus or reach direction. Each feature uses 20 equal-width bins, from zero "
        "to its pooled maximum rounded upward. Bins are identical across stages within a feature, "
        "but radial ranges and bin widths differ between features. Every observed value is retained. "
        "Color is the fraction of usable trials per bin in each stage; each wedge sums to 100%. "
        "All three features share one linear color scale. The expanding areas of outer bins do "
        "not encode probability; read color intensity, not sector area. Outer colored arcs identify "
        "stages only. No angular interpolation, smoothing, or significance testing is performed.\n\n"
        "Usable stage counts: 197, 195, 196, 198, 196, 195, respectively (1,177 of 1,188 windows; "
        "198 original short-delay trials). Eleven unusable fits are omitted without imputation. "
        "Repeated stages belong to the same trials. Radii and half-width are in per-window-normalized "
        "signal units. Band half-width is (R1_out-R1_in+R2_out-R2_in)/4, not R1-R2. "
        "No preprocessing, embedding, geometric fitting, feature extraction, or decoding was rerun.\n")
    write_json(DESTINATION / "provenance.json", dict(
        source_sha256=sha256(__file__), protected_files=protected, previous_files_unchanged=True,
        features=list(FEATURES), stage_order=list(STAGES), n_bins=N_BINS, shared_color_limits=[0., vmax],
        bin_edges={f: saved[f][0].tolist() for f in FEATURES}, stage_counts=saved[FEATURES[0]][2].tolist(),
        normalization="within-stage fraction per bin", observed_values_trimmed=False,
        imputation=False, geometric_fitting_called=False, decoding_called=False, exports=exports,
        pdf=dict(path=str(pdf_path), pages=4, sha256=sha256(pdf_path))))
    print(f"Verified 18 distributions; {len(protected)} previous files unchanged. Saved {pdf_path}")


if __name__ == "__main__":
    main()
