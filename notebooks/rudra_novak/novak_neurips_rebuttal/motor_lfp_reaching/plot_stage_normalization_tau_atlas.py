"""Three vector PDF atlases: saved MAD versus envelope signals, tau 1-25 ms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from motor_lfp_utils import lag_embed


UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / "outputs/stage_tau3_envelope_geometry_v1"
OUTPUT = UNIT / "outputs/stage_normalization_tau_01_to_25_geometry_v1"
TRIALS = (6, 9, 10)
TAUS = tuple(range(1, 26))
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
CAMERA = dict(elevation=24, azimuth=-58)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    paths = [SOURCE / name for name in ("inputs_and_clouds.npz", "windows.csv", "config.json")]
    paths.extend(SOURCE.glob("*.png"))
    original = json.loads((SOURCE / "protected_hashes.json").read_text())
    paths.extend(Path(path) for path in original)
    result = {str(path.resolve()): sha256(path) for path in sorted(set(paths))}
    assert all(result[path] == digest for path, digest in original.items())
    return result


def prepare():
    config = json.loads((SOURCE / "config.json").read_text())
    metadata = pd.read_csv(SOURCE / "windows.csv")
    assert config["sampling_rate_hz"] == 1000 and config["dimension"] == 3
    assert config["original_trial_numbers"] == list(TRIALS)
    assert tuple(config["stage_names"]) == STAGES
    assert len(metadata) == 18 and metadata.delay.eq("short").all()
    with np.load(SOURCE / "inputs_and_clouds.npz") as saved:
        signals = {"mad": saved["old_processed"], "envelope": saved["envelope_processed"]}
        references = {"mad": saved["old_clouds"], "envelope": saved["envelope_clouds"]}
    for trial in TRIALS:
        rows = metadata.loc[metadata.original_trial_number == trial].sort_values("stage")
        assert tuple(rows.stage_name) == STAGES and rows.stage.tolist() == list(range(6))
        assert rows.direction.nunique() == 1 and rows.heldout_fold_zero_based.nunique() == 1
        for row in rows.itertuples():
            assert row.end_sample_exclusive - row.start_sample == 300
            if row.stage in (1, 3):
                assert row.anchor_name == ("TC_off" if row.stage == 1 else "SC_off")
                assert row.start_sample == row.event_sample
    clouds = {}
    for method, windows in signals.items():
        assert windows.shape == (18, 300) and np.isfinite(windows).all()
        for tau in TAUS:
            points = np.stack([lag_embed(window, dim=3, tau=tau) for window in windows])
            np.testing.assert_array_equal(points, np.stack(
                (windows[:, 2*tau:], windows[:, tau:-tau], windows[:, :-2*tau]), axis=2))
            assert points.shape == (18, 300 - 2*tau, 3) and np.isfinite(points).all()
            if tau == 3:
                np.testing.assert_array_equal(points, references[method])
            clouds[f"{method}_tau_{tau:02d}"] = points
    np.savez_compressed(OUTPUT / "clouds.npz", **clouds)
    metadata.to_csv(OUTPUT / "windows.csv", index=False)
    write_json(OUTPUT / "config.json", dict(
        source=str(SOURCE), recording=config["recording"], trial_numbers=list(TRIALS),
        stage_names=list(STAGES), stage_labels=config["stage_labels"],
        stage_colors=config["stage_colors"], dimension=3, sampling_rate_hz=1000,
        delays_ms=list(TAUS), window_samples=300, methods=["mad", "envelope"],
        normalization=config["normalization_formula"], normalization_recomputed=False,
        layout="MAD left two columns; envelope right two columns; TC/SC/GO row pairs",
        camera=CAMERA, axis_limits="fixed across all stages and tau per trial and method",
        separate_method_units=True, fitter_called=False, decoder_called=False,
        pca_called=False, ripser_called=False, native_point_counts=True))


def geometry_axis(ax, points, color, limit):
    line, = ax.plot(*points.T, color=color, lw=.75, antialiased=True,
                    solid_capstyle="round", solid_joinstyle="round")
    np.testing.assert_array_equal(np.column_stack(line.get_data_3d()), points)
    assert np.max(np.abs(points)) < limit
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    ax.view_init(elev=CAMERA["elevation"], azim=CAMERA["azimuth"])
    ax.set_axis_off()


def make_page(trial, direction, tau, clouds, indices, limits, config):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(.045, .965, f"Trial {trial} | tau = {tau} ms", fontsize=17, weight="bold", va="top")
    fig.text(.045, .922, f"Monkey T | y070316009-12 | channel 7 | short delay | direction {direction}",
             fontsize=9.5, va="top")
    fig.text(.955, .961, f"{tau} / 25", fontsize=10, ha="right", va="top")
    fig.text(.267, .87, "Window MAD", fontsize=12, weight="bold", ha="center")
    fig.text(.733, .87, "Hilbert envelope", fontsize=12, weight="bold", ha="center")
    for method_index, method in enumerate(("mad", "envelope")):
        for stage, index in enumerate(indices):
            position = (stage // 2) * 4 + method_index * 2 + stage % 2 + 1
            ax = fig.add_subplot(3, 4, position, projection="3d", proj_type="ortho")
            points = clouds[f"{method}_tau_{tau:02d}"][index]
            geometry_axis(ax, points, config["stage_colors"][stage], limits[method])
            ax.set_title(config["stage_labels"][stage], fontsize=10, pad=0)
    fig.subplots_adjust(left=.035, right=.965, top=.83, bottom=.125, wspace=.06, hspace=.09)
    fig.text(.5, .073, f"m = 3 | 300-ms windows | {300-2*tau} observed points per stage", fontsize=9,
             ha="center")
    fig.text(.5, .046, f"Fixed limits: MAD +/-{limits['mad']:.2f} normalized units | "
             f"Envelope +/-{limits['envelope']:.1f} signal units", fontsize=8.5, ha="center")
    fig.text(.5, .022, f"Coordinates: x(t), x(t - {tau} ms), x(t - {2*tau} ms)", fontsize=8.5,
             ha="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *(ax.title for ax in fig.axes)]:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
    return fig


def verify_pdf(path, trial, config):
    with fitz.open(path) as document:
        assert len(document) == 25
        for tau, page in enumerate(document, start=1):
            text = page.get_text()
            assert f"Trial {trial} | tau = {tau} ms" in text
            assert "Window MAD" in text and "Hilbert envelope" in text
            assert all(text.count(label) == 2 for label in config["stage_labels"])
            assert f"{300-2*tau} observed points per stage" in text
            assert not page.get_images() and len(page.get_drawings()) >= 12
        document.set_toc([[1, f"tau = {tau} ms", tau] for tau in TAUS])
        document.saveIncr()


def render():
    config = json.loads((OUTPUT / "config.json").read_text())
    metadata = pd.read_csv(OUTPUT / "windows.csv")
    with np.load(OUTPUT / "clouds.npz") as saved:
        clouds = {key: saved[key] for key in saved.files}
    assert len(clouds) == 50
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#263238", "pdf.fonttype": 42, "pdf.compression": 6})
    exports, pages, scales = [], [], []
    for trial in TRIALS:
        indices = np.flatnonzero(metadata.original_trial_number.to_numpy() == trial)
        assert metadata.iloc[indices].stage.tolist() == list(range(6))
        direction = int(metadata.iloc[indices[0]].direction)
        limits = {method: max(float(np.max(np.abs(clouds[f"{method}_tau_{tau:02d}"][indices])))
                             for tau in TAUS) * 1.08 for method in ("mad", "envelope")}
        scales.append(dict(trial=trial, **limits))
        path = OUTPUT / f"trial_{trial}_MAD_vs_envelope_tau_01_to_25.pdf"
        pdf_metadata = dict(Title=f"Trial {trial}: MAD versus envelope geometry, tau 1-25 ms",
                            Subject="Observed six-stage lag trajectories; no fitting or decoding",
                            Author="NeuralFieldManifold")
        with PdfPages(path, metadata=pdf_metadata) as pdf:
            for tau in TAUS:
                fig = make_page(trial, direction, tau, clouds, indices, limits, config)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(dict(trial=trial, page=tau, tau_ms=tau,
                                  points_per_stage=300-2*tau, pdf=path.name))
                if tau in (1, 5, 10, 15, 20, 25):
                    print(f"Trial {trial}: rendered tau {tau}/25", flush=True)
        verify_pdf(path, trial, config)
        exports.append(dict(name=path.name, pages=25, sha256=sha256(path)))
        print(path, flush=True)
    pd.DataFrame(pages).to_csv(OUTPUT / "page_index.csv", index=False)
    pd.DataFrame(scales).to_csv(OUTPUT / "display_scales.csv", index=False)
    return exports


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    before = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        assert json.loads((OUTPUT / "protected_hashes.json").read_text()) == before
    else:
        if (OUTPUT / "protected_hashes.json").exists():
            raise FileExistsError("Atlas already initialized; use --plot-only")
        prepare()
        write_json(OUTPUT / "protected_hashes.json", before)
    exports = render()
    assert source_hashes() == before
    write_json(OUTPUT / "validation.json", dict(
        trial_numbers=list(TRIALS), delays_ms=list(TAUS), pdf_count=3, total_pages=75,
        observed_clouds=900, tau3_matches_previous_clouds_exact=True,
        direct_delay_coordinates_verified=True, vector_pdf_pages_verified=True,
        all_page_labels_verified=True, common_limits_across_tau_per_method=True,
        previous_results_unchanged=True, normalization_recomputed=False,
        fitter_called=False, decoder_called=False, exports=exports,
        source_sha256=sha256(Path(__file__))))
    (OUTPUT / "README.md").write_text(
        "# MAD versus envelope geometry: tau 1-25 ms\n\n"
        "Three PDFs, one each for the same short-delay trials 6, 9 and 10. Each PDF "
        "contains 25 bookmarked vector pages. Page number equals tau in milliseconds. "
        "MAD geometry is in the left two columns, Hilbert-envelope geometry in the right "
        "two columns. Rows pair pre/post-TC, pre/post-SC and pre/post-GO.\n\n"
        "Uses the exact saved 300-sample signals from stage_tau3_envelope_geometry_v1. "
        "m=3; all native points are retained (298 at tau=1, 250 at tau=25). Each point "
        "is an observed delayed sample; lines connect samples without smoothing. No "
        "normalization, filtering, geometric fitting, decoding or persistent homology "
        "was rerun. The tau=3 clouds reproduce the previous comparison exactly.\n\n"
        "The camera and coordinate limits are fixed across all stages and all 25 pages "
        "within each trial and method. MAD and envelope have different units and separate "
        "numeric limits, printed on each page; absolute sizes cannot be compared across "
        "methods. All earlier input/results were hash-checked unchanged.\n\n"
        "Saved clouds, window identities, page index, scales and hashes support "
        "reproduction. Plot-only regeneration: plot_stage_normalization_tau_atlas.py --plot-only.\n")


if __name__ == "__main__":
    main()
