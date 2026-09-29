"""Verify frozen corrected stage results and collect their presentation outputs."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_cue_offset_v2"
CARTOON = HERE / "outputs/macaque_task_six_stages_cue_offset_v2"
DEST = REPO / "CORRECTED_STAGE_DECODING"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify():
    for filename in ("validation.json", "figure_validation.json"):
        assert json.loads((SOURCE / filename).read_text())["passed"]
    hashes = json.loads((SOURCE / "frozen_input_hashes.json").read_text())
    for filename, expected in hashes.items():
        assert sha256(Path(filename)) == expected, filename
    figures = json.loads((SOURCE / "figure_validation.json").read_text())
    for entry in figures["png_checks"]:
        assert sha256(SOURCE.parent / entry["file"]) == entry["sha256"]

    config = json.loads((SOURCE / "config.json").read_text())
    anchors = dict(zip(config["stage_names"], config["stage_anchors"]))
    assert anchors["post_TC"] == "TC_off" and anchors["post_SC"] == "SC_off"
    events = pd.read_csv(SOURCE / "inputs/event_samples.csv").set_index("trial_index")
    windows = pd.read_csv(SOURCE / "inputs/window_metadata.csv")
    assert len(windows) == 1188 and len(events) == 198
    assert windows.groupby("trial_index").stage.nunique().eq(6).all()
    assert windows.groupby("trial_index").heldout_fold_zero_based.nunique().eq(1).all()
    assert windows.stage_name.value_counts().eq(198).all()
    for row in windows.itertuples():
        anchor = anchors[row.stage_name]
        event = int(events.loc[row.trial_index, anchor + "_sample"])
        start = event - 300 if row.stage_name.startswith("pre_") else event
        assert row.original_trial_number == events.loc[row.trial_index, "original_trial_number"]
        assert row.anchor_name == anchor and row.anchor_code == config["event_codes"][anchor]
        assert row.event_sample == event and row.start_sample == start
        assert row.end_sample_exclusive == start + 300
    for _, group in windows.groupby("trial_index"):
        group = group.sort_values("start_sample")
        assert np.all(group.end_sample_exclusive.to_numpy()[:-1] <= group.start_sample.to_numpy()[1:])

    predictions = pd.read_csv(SOURCE / "tables/heldout_predictions.csv").sort_values("window_index")
    windows = windows.sort_values("window_index")
    for key in ("window_index", "trial_index", "original_trial_number", "stage", "stage_name"):
        assert np.array_equal(predictions[key], windows[key]), key
    assert np.array_equal(predictions.heldout_fold, windows.heldout_fold_zero_based)
    assert predictions.window_index.is_unique and len(predictions) == 1188
    f1 = f1_score(predictions.stage, predictions.predicted_stage, average="macro")
    accuracy = accuracy_score(predictions.stage, predictions.predicted_stage)
    summary = pd.read_csv(SOURCE / "tables/summary.csv").iloc[0]
    np.testing.assert_allclose([f1, accuracy], [summary.macro_f1, summary.accuracy], rtol=0, atol=1e-12)
    matrix = confusion_matrix(predictions.stage, predictions.predicted_stage, labels=range(6), normalize="true")
    saved_matrix = pd.read_csv(SOURCE / "tables/confusion_fraction.csv", index_col=0)
    np.testing.assert_allclose(matrix, saved_matrix.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "trials": 198, "windows": 1188,
        "macro_f1": f1, "accuracy": accuracy,
        "frozen_inputs_verified": len(hashes),
        "validated_png_hashes_verified": len(figures["png_checks"]),
        "post_TC": "300 ms starting at recorded TC offset (203)",
        "post_SC": "300 ms starting at recorded SC instruction end / distractor onset (206)",
        "window_timing_and_trial_grouped_folds_checked": True,
        "scores_and_confusion_reproduced_from_saved_predictions": True,
        "analysis_or_plotting_rerun": False,
    }


def main():
    check = verify()
    sources_before = {p: sha256(p) for root in (SOURCE, CARTOON) for p in root.rglob("*") if p.is_file()}
    main_files = {
        CARTOON / "macaque_task_six_stages.png": "01_task_timeline.png",
        SOURCE / "png/stage_confusion.png": "02_stage_confusion.png",
        SOURCE / "f1_heatmap_v1/stage_f1_heatmap.png": "03_stage_f1_heatmap.png",
        SOURCE / "png/stage_f1.png": "04_stage_f1_intervals.png",
        SOURCE / "feature_distributions_v1/png/all_geometric_feature_distributions.png": "05_all_15_feature_distributions.png",
        SOURCE / "radial_feature_distributions_v1/png/radial_stage_distributions.png": "06_circular_radius_width_distributions.png",
        SOURCE / "trajectory_style_v1/png/geometry_trial_6_fitted.png": "07_trial_6_geometry_with_fits.png",
        SOURCE / "all_trials_pdf_v1/all_198_trials_with_fits.pdf": "08_all_198_trials_with_fits.pdf",
        SOURCE / "all_trials_pdf_v1/all_198_trials_observed.pdf": "09_all_198_trials_observed.pdf",
        SOURCE / "report.md": "REPORT.md",
    }
    copies = [(src, DEST / target) for src, target in main_files.items()]
    groups = {
        "png": "individual_panels/original_point_style",
        "trajectory_style_v1/png": "individual_panels/trajectory_style",
        "alternative_examples_v1/png": "individual_panels/additional_trials",
        "feature_distributions_v1/png": "individual_panels/feature_densities",
        "radial_feature_distributions_v1/png": "individual_panels/circular_distributions",
        "tables": "tables",
    }
    for source_dir, target_dir in groups.items():
        copies.extend((p, DEST / target_dir / p.name) for p in sorted((SOURCE / source_dir).iterdir()) if p.is_file())
    for filename in ("config.json", "validation.json", "figure_validation.json", "captions.md"):
        copies.append((SOURCE / filename, DEST / "verification" / filename))
    for filename in ("window_metadata.csv", "event_samples.csv", "event_durations.csv", "correction_audit.csv"):
        copies.append((SOURCE / "inputs" / filename, DEST / "verification" / filename))
    copies.append((SOURCE / "all_trials_pdf_v1/trial_page_index.csv", DEST / "tables/trial_page_index.csv"))
    for filename in ("caption.md", "provenance.json", "stage_window_labels.csv", "macaque_task_six_stages.svg", "macaque_task_six_stages.pdf"):
        copies.append((CARTOON / filename, DEST / "cartoon_editable" / filename))
    for directory in ("trajectory_style_v1", "alternative_examples_v1", "feature_distributions_v1", "radial_feature_distributions_v1", "all_trials_pdf_v1"):
        for p in (SOURCE / directory).glob("*.md"):
            copies.append((p, DEST / "verification/source_notes" / directory / p.name))
    manifest = []
    for src, dst in copies:
        digest = sources_before[src]
        if dst.exists():
            assert sha256(dst) == digest, f"Refusing to replace changed file: {dst}"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        assert sha256(dst) == digest
        manifest.append({"file": str(dst.relative_to(DEST)), "source": str(src), "sha256": digest})
    for src, digest in sources_before.items():
        assert sha256(src) == digest, f"Source changed: {src}"
    check["source_files_unchanged"] = len(sources_before)
    check["copied_files_verified"] = len(manifest)
    (DEST / "verification/collection_check.json").write_text(json.dumps(check, indent=2) + "\n")
    (DEST / "verification/file_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"folder": str(DEST), **check}, indent=2))


if __name__ == "__main__":
    main()
