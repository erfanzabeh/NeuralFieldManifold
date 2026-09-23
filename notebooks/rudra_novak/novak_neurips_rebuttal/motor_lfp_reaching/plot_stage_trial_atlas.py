"""Vector PDF atlases of every frozen six-stage trial, without refitting."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from plot_stage_geometry_decoding import INK, outlines, style
from plot_stage_geometry_trajectories import trajectory_axis
from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, LABELS

DESTINATION = OUTPUT / "all_trials_pdf_v1"


def trial_limit(rows, clouds, results):
    extent = [float(np.max(np.abs(clouds[rows.window_index.to_numpy()])))]
    for row in rows.itertuples():
        if row.usable:
            extent.append(float(np.max(np.abs(outlines(results[row.window_index])))))
    return max(extent) * 1.08


def make_page(rows, clouds, results, limit, page_number, page_count, overlay):
    trial = int(rows.original_trial_number.iloc[0])
    direction = int(rows.direction.iloc[0])
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.045, .955, f"Original trial {trial}", fontsize=18, weight="bold", va="top")
    fig.text(.045, .912, f"Monkey T  |  y070316009-12  |  Channel 7  |  Reach direction {direction}  |  Short delay",
             fontsize=10, va="top")
    fig.text(.96, .95, f"{page_number} / {page_count}", fontsize=10, ha="right", va="top")
    kind = "Observed trajectories + saved fits" if overlay else "Observed trajectories"
    fig.text(.96, .866, kind, fontsize=10, ha="right", va="top")
    for row in rows.itertuples():
        position = (row.stage % 2) * 3 + row.stage // 2 + 1
        ax = fig.add_subplot(2, 3, position, projection="3d")
        trajectory_axis(ax, clouds[row.window_index], results[row.window_index], row.stage, limit, overlay)
        timing = "-300 to 0 ms" if row.stage % 2 == 0 else "0 to 300 ms"
        ax.set_title(f"{LABELS[row.stage]}  |  {timing}", fontsize=11, pad=3)
        if overlay and not row.usable:
            ax.text2D(.5, .98, "Fit unusable; observed trajectory only", transform=ax.transAxes,
                      ha="center", va="top", fontsize=8, color=INK)
    fig.subplots_adjust(left=.025, right=.955, top=.82, bottom=.13, wspace=.07, hspace=.31)
    fig.text(.045, .066, "m = 3  |  delay = 3 ms  |  300-ms windows  |  294 observed points per stage",
             fontsize=9)
    fig.text(.045, .042, f"Identical view and limits within this trial: +/-{limit:.2f}. Limits vary between trials. Normalized signal units.",
             fontsize=8)
    if overlay:
        usable = int(rows.usable.sum())
        fig.text(.045, .019, f"Gray solid: fitted outer ellipse. Gray dashed: fitted inner ellipse. Usable fits: {usable}/6.",
                 fontsize=8)
    return fig


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width > 0 and box.height > 0 and (
            box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1
        ):
            outside.append(artist.get_text())
    if outside:
        raise AssertionError(f"Text outside page: {outside}")


def finish_pdf(path, trials, direction_by_trial):
    with fitz.open(path) as document:
        if len(document) != len(trials):
            raise AssertionError("Missing trial pages")
        for page, trial in zip(document, trials):
            text = page.get_text()
            if f"Original trial {trial}\n" not in text:
                raise AssertionError(f"Incorrect trial label on page {page.number + 1}")
            for label in LABELS:
                if label not in text:
                    raise AssertionError(f"Missing stage {label}")
            if page.get_images():
                raise AssertionError("Scientific artwork should remain vector")
        document.set_toc([[1, f"Trial {trial} | Direction {direction_by_trial[trial]}", i + 1]
                          for i, trial in enumerate(trials)])
        document.saveIncr()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    protected = {str(p): sha256(p) for p in sorted(OUTPUT.rglob("*"))
                 if p.is_file() and not p.is_relative_to(DESTINATION)}
    features = pd.read_csv(OUTPUT / "tables/features.csv")
    with np.load(OUTPUT / "inputs/windows.npz") as saved:
        clouds = saved["clouds"]
    assert clouds.shape == (1188, 294, 3) and np.isfinite(clouds).all()
    assert len(features) == 1188 and features.window_index.nunique() == 1188
    assert set(features.window_index) == set(range(1188))
    assert features.groupby("original_trial_number").size().eq(6).all()
    assert features.groupby("original_trial_number").stage.nunique().eq(6).all()
    assert features.groupby("original_trial_number").direction.nunique().eq(1).all()
    assert features.delay.eq("short").all()
    trials = sorted(int(t) for t in features.original_trial_number.unique())
    assert len(trials) == 198
    results = {}
    for row in features.itertuples():
        result = json.loads((OUTPUT / "checkpoints" / f"window_{row.window_index:04d}.json").read_text())
        assert result["original_trial_number"] == row.original_trial_number
        assert result["stage_name"] == row.stage_name
        assert result["cloud_sha256"] == row.cloud_sha256
        result["usable"] = bool(row.usable)
        results[row.window_index] = result
    style()
    plt.rcParams.update({"pdf.fonttype": 42, "pdf.compression": 6})
    paths = [DESTINATION / "all_198_trials_observed.pdf", DESTINATION / "all_198_trials_with_fits.pdf"]
    page_index = []
    metadata = {"Title": "Monkey T, channel 7: all 198 six-stage trials",
                "Subject": "Frozen m=3, tau=3 ms clouds; vector trajectories; no refitting or decoding",
                "Author": "NeuralFieldManifold"}
    with PdfPages(paths[0], metadata=metadata) as observed, PdfPages(paths[1], metadata=metadata) as fitted:
        for page_number, trial in enumerate(trials, start=1):
            rows = features[features.original_trial_number == trial].sort_values("stage")
            assert rows.stage.tolist() == list(range(6))
            limit = trial_limit(rows, clouds, results)
            for overlay, pdf in ((False, observed), (True, fitted)):
                fig = make_page(rows, clouds, results, limit, page_number, len(trials), overlay)
                check_layout(fig)
                pdf.savefig(fig)
                plt.close(fig)
            failed = rows.loc[~rows.usable, "stage_name"].tolist()
            page_index.append(dict(page=page_number, original_trial_number=trial,
                                   direction=int(rows.direction.iloc[0]), usable_fits=int(rows.usable.sum()),
                                   unusable_stages=";".join(failed), axis_minimum=-limit, axis_maximum=limit))
            if page_number % 20 == 0 or page_number == len(trials):
                print(f"Rendered {page_number}/{len(trials)} trials in both PDFs", flush=True)
    direction_by_trial = {r["original_trial_number"]: r["direction"] for r in page_index}
    for path in paths:
        finish_pdf(path, trials, direction_by_trial)
    index = pd.DataFrame(page_index)
    index.to_csv(DESTINATION / "trial_page_index.csv", index=False)

    # Render representative pages, including a failed fit and the largest display span.
    inspection = DESTINATION / "inspection"
    inspection.mkdir(exist_ok=True)
    review_trials = sorted({trials[0], trials[len(trials)//2], trials[-1],
                            int(index.loc[index.usable_fits < 6, "original_trial_number"].iloc[0]),
                            int(index.loc[index.axis_maximum.idxmax(), "original_trial_number"])})
    for path in paths:
        with fitz.open(path) as document:
            assert len(document.get_toc()) == 198
            for trial in review_trials:
                page_number = trials.index(trial)
                pixmap = document[page_number].get_pixmap(dpi=150, alpha=False)
                pixmap.save(inspection / f"{path.stem}_trial_{trial}.png")

    assert protected == {p: sha256(p) for p in protected}, "Previous results changed"
    shutil.copy2(__file__, DESTINATION / Path(__file__).name)
    write_json(DESTINATION / "provenance.json", dict(
        source_sha256=sha256(__file__), protected_files=protected, previous_files_unchanged=True,
        trial_count=198, window_count=1188, usable_fits=int(features.usable.sum()),
        pages_per_pdf=198, ordering="ascending original trial number", all_trials_included=True,
        scale="same across all six stages and both PDFs per trial; varies across trials",
        camera=dict(elevation=24, azimuth=-58), geometry_unchanged=True,
        fitting_called=False, decoding_called=False, inspected_trial_ids=review_trials,
        exports=[dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size) for p in paths]))
    (DESTINATION / "README.md").write_text(
        "# All-trial geometry review\n\n"
        "Each PDF has 198 pages, ordered by original trial ID, with a bookmark per trial. "
        "The same trial appears on the same page in both versions. Each page shows six "
        "300-ms observed stage trajectories; one version also shows saved annular fit outlines. "
        "Every original trial is retained. The 11 unusable fits have their observed trajectories "
        "shown and their outlines omitted with an explicit label. No fitting or decoding was run.\n\n"
        "Viewpoint and scale are identical across stages within each trial and across the two "
        "versions. Scale varies between trials and is printed on each page; do not compare "
        "apparent sizes across pages without reading the axes. All paths are vector artwork "
        "connecting existing samples in temporal order, without smoothing or resampling. "
        "Previous experiment files are hash-verified unchanged.\n")
    print(f"Verified 396 vector pages, 1188 windows, and {len(protected)} unchanged files", flush=True)
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
