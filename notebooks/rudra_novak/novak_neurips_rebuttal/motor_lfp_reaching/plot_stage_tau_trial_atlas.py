"""Plot saved native-tau stage geometry for selected trials; no fitting or decoding."""
from __future__ import annotations

import json
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from motor_lfp_utils import lag_embed
from plot_stage_geometry_decoding import INK, style
from plot_tau9_geometry_examples import outlines
from prego_geometric_fits import array_hash, sha256, write_json
from stage_geometry_decoding import COLORS, LABELS, OUTPUT as SOURCE, STAGES, UNIT

ROOT = UNIT / "outputs/tau_sweep_v1"
DESTINATION = ROOT / "trial_211_579_native_tau_geometry_atlas_v1"
TRIALS = (211, 579)
TAUS = tuple(range(1, 21))


def load_records():
    if not (ROOT / "validation.json").is_file():
        raise ValueError("The tau sweep has not been validated")
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv")
    with np.load(SOURCE / "inputs/windows.npz") as saved:
        processed = saved["processed"]
    if processed.shape != (1188, 300) or not np.isfinite(processed).all():
        raise ValueError("Unexpected frozen stage signals")

    records = {}
    extent = 0.0
    for trial in TRIALS:
        rows = metadata.loc[metadata.original_trial_number == trial].sort_values("stage")
        if len(rows) != 6 or tuple(rows.stage_name) != STAGES:
            raise ValueError(f"Incomplete or misordered original trial {trial}")
        if rows.direction.nunique() != 1 or not rows.delay.eq("short").all():
            raise ValueError(f"Unexpected condition for original trial {trial}")
        for tau in TAUS:
            with np.load(ROOT / "predictions" / f"native_tau_{tau:02d}.npz") as saved:
                predictions = saved["predictions"]
                labels = saved["labels"]
                trial_ids = saved["trial_ids"]
            fits = pd.read_csv(ROOT / "tables" / f"fits_native_tau_{tau:02d}.csv")
            for row in rows.itertuples():
                index = int(row.window_index)
                if labels[index] != row.stage or trial_ids[index] != row.trial_index:
                    raise ValueError(f"Saved prediction identity mismatch: {trial}, {tau}, {row.stage_name}")
                points = lag_embed(processed[index], dim=3, tau=tau)
                if points.shape != (300 - 2 * tau, 3):
                    raise ValueError("Wrong native point count")
                checkpoint = ROOT / "checkpoints/native" / f"tau_{tau:02d}" / f"window_{index:04d}.json"
                result = json.loads(checkpoint.read_text())
                if (result["cloud_sha256"] != array_hash(points)
                        or result["window_index"] != index
                        or result["original_trial_number"] != trial
                        or result["stage_name"] != row.stage_name
                        or result["tau_ms"] != tau):
                    raise ValueError(f"Saved fit/cloud mismatch: {checkpoint}")
                fit_row = fits.loc[fits.window_index == index]
                if len(fit_row) != 1 or int(fit_row.iloc[0].original_trial_number) != trial:
                    raise ValueError(f"Saved fit-table identity mismatch: {trial}, {tau}, {index}")
                usable = bool(fit_row.iloc[0].usable)
                boundary = outlines(result["fit"]) if usable else None
                extent = max(extent, float(np.max(np.abs(points))))
                if boundary is not None:
                    extent = max(extent, *(float(np.max(np.abs(line))) for line in boundary))
                records[(trial, tau, row.stage)] = dict(
                    points=points, boundary=boundary, usable=usable,
                    reason="" if usable else str(fit_row.iloc[0].unusable_reason),
                    index=index, direction=int(row.direction),
                    predicted=int(predictions[index]), correct=bool(predictions[index] == row.stage),
                    cloud_sha256=result["cloud_sha256"], checkpoint=checkpoint,
                )
    if len(records) != len(TRIALS) * len(TAUS) * len(STAGES):
        raise ValueError("Incomplete atlas")
    return records, extent * 1.06


def make_page(trial, tau, records, limit, overlay):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    entries = [records[(trial, tau, stage)] for stage in range(6)]
    direction = entries[0]["direction"]
    correct = sum(entry["correct"] for entry in entries)
    fig.text(.045, .955, f"Original trial {trial}  |  tau = {tau} ms", fontsize=17,
             weight="bold", color=INK, va="top")
    fig.text(.045, .912,
             f"Monkey T  |  y070316009-12  |  Channel 7  |  Reach direction {direction}  |  Short delay",
             fontsize=9.5, color=INK, va="top")
    fig.text(.955, .951, f"{tau} / 20", fontsize=10, color=INK, ha="right", va="top")
    fig.text(.955, .913, f"Held-out stages correct: {correct} / 6", fontsize=9,
             color=INK, ha="right", va="top")
    fig.text(.955, .875, "Observed + saved fit" if overlay else "Observed only",
             fontsize=8.5, color=INK, ha="right", va="top")

    for stage, entry in enumerate(entries):
        position = (stage % 2) * 3 + stage // 2 + 1
        ax = fig.add_subplot(2, 3, position, projection="3d", proj_type="ortho")
        points = entry["points"]
        color = COLORS[stage]
        ax.plot(*points.T, lw=.7, color=color, alpha=.55)
        ax.scatter(*points.T, s=2.8, color=color, alpha=.8,
                   depthshade=False, linewidths=0)
        if overlay and entry["boundary"] is not None:
            outer, inner = entry["boundary"]
            ax.plot(*outer.T, color=INK, lw=1.4)
            ax.plot(*inner.T, color=INK, lw=1.0, ls="--")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        ax.set_box_aspect((1, 1, 1), zoom=1.18)
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()
        ax.set_title(LABELS[stage], fontsize=11, color=INK, pad=1)
        if overlay and not entry["usable"]:
            ax.text2D(.5, .02, "Fit unusable; observed only", transform=ax.transAxes,
                      ha="center", fontsize=8, color=INK)

    fig.subplots_adjust(left=.03, right=.97, top=.84, bottom=.11,
                        wspace=.025, hspace=.08)
    handles = [Line2D([0], [0], color="#71818a", lw=1.5, marker="o", markersize=4,
                      label="Observed lag trajectory")]
    if overlay:
        handles.extend([Line2D([0], [0], color=INK, lw=1.5,
                               label="Fitted outer boundary"),
                        Line2D([0], [0], color=INK, lw=1.2, ls="--",
                               label="Fitted inner boundary")])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .067), fontsize=8.5)
    fig.text(.5, .026,
             f"300-ms windows  |  m = 3  |  {300 - 2*tau} observed points/stage  |  "
             f"common coordinate limits: +/-{limit:.2f}",
             ha="center", fontsize=8, color=INK)
    return fig


def verify_pdf(path, trial, records, overlay):
    with fitz.open(path) as document:
        if len(document) != 20:
            raise AssertionError("Expected 20 tau pages")
        for tau, page in enumerate(document, start=1):
            text = page.get_text()
            if f"Original trial {trial}" not in text or f"tau = {tau} ms" not in text:
                raise AssertionError(f"Incorrect page identity: {path}, page {tau}")
            if any(label not in text for label in LABELS):
                raise AssertionError(f"Missing stage label: {path}, page {tau}")
            if page.get_images():
                raise AssertionError("Atlas art should remain vector")
            if overlay and any(not records[(trial, tau, stage)]["usable"] for stage in range(6)):
                if "Fit unusable; observed only" not in text:
                    raise AssertionError("Unusable fit was not identified")
        document.set_toc([[1, f"tau = {tau} ms", tau] for tau in TAUS])
        document.saveIncr()


def main():
    protected = [SOURCE / "inputs/windows.npz", SOURCE / "inputs/window_metadata.csv",
                 ROOT / "validation.json"]
    protected += sorted((ROOT / "predictions").glob("native_tau_*.npz"))
    protected_hashes = {str(path): sha256(path) for path in protected}
    records, limit = load_records()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    style()
    plt.rcParams.update({"pdf.fonttype": 42, "pdf.compression": 6})
    index = []
    paths = []
    for trial in TRIALS:
        for overlay in (False, True):
            suffix = "with_fits" if overlay else "observed"
            path = DESTINATION / f"trial_{trial}_native_tau_01_to_20_{suffix}.pdf"
            metadata = dict(Title=f"Monkey T trial {trial}: native delay geometry across stages",
                            Subject="Saved original-coordinate lag embeddings and annular fits; no refitting",
                            Author="NeuralFieldManifold")
            with PdfPages(path, metadata=metadata) as pdf:
                for tau in TAUS:
                    fig = make_page(trial, tau, records, limit, overlay)
                    pdf.savefig(fig)
                    plt.close(fig)
                    if overlay:
                        entries = [records[(trial, tau, stage)] for stage in range(6)]
                        index.append(dict(original_trial_number=trial, tau_ms=tau, pdf_page=tau,
                                          direction=entries[0]["direction"],
                                          heldout_stages_correct=sum(e["correct"] for e in entries),
                                          usable_fits=sum(e["usable"] for e in entries),
                                          unusable_stages=";".join(STAGES[i] for i, e in enumerate(entries)
                                                                   if not e["usable"]),
                                          observed_points_per_stage=300 - 2*tau,
                                          coordinate_limit=limit))
            verify_pdf(path, trial, records, overlay)
            paths.append(path)
            print(f"Verified {path}", flush=True)

    pd.DataFrame(index).to_csv(DESTINATION / "page_index.csv", index=False)
    preview = DESTINATION / "inspection"
    preview.mkdir(exist_ok=True)
    for path in paths:
        with fitz.open(path) as document:
            for tau in (1, 3, 9, 20):
                document[tau - 1].get_pixmap(dpi=150, alpha=False).save(
                    preview / f"{path.stem}_tau_{tau:02d}.png")
    if protected_hashes != {str(path): sha256(path) for path in protected}:
        raise AssertionError("A frozen input or prediction changed")
    write_json(DESTINATION / "provenance.json", dict(
        trials=list(TRIALS), taus_ms=list(TAUS), variant="native", pages_per_pdf=20,
        common_coordinate_limit=limit, protected_hashes=protected_hashes,
        no_fitting_or_decoding=True,
        pdfs=[dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)
              for path in paths]))
    (DESTINATION / "README.md").write_text(
        "# Native-delay stage geometry atlases\n\n"
        "Two 20-page PDFs per original trial (211 and 579): observed-only and "
        "observed with saved fit outlines. Each page is a single delay "
        "from 1 through 20 ms and contains the six corrected cue-offset stages. "
        "Trajectories use lag coordinates deterministically rebuilt from the frozen "
        "300-ms processed windows; fit outlines come from the saved native-point "
        "tau sweep. No preprocessing, fitting, or decoding was rerun. "
        "Every page uses the same original-coordinate view and scale across both trials. "
        "Unusable fits are explicitly labeled and have no outline. "
        "The held-out correct count is descriptive, not a new model evaluation. "
        "The page index records counts and fit usability.\n")


if __name__ == "__main__":
    main()
