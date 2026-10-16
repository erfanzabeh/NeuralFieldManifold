#!/usr/bin/env python
"""Audit trial separation, saved predictions, statistics, and standalone exports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from plot_prego_panels import DEFAULT_OUTPUT, load_tables
from prego_decoding import heldout_predictions, learn_embedding, stable_seed
from prego_statistics import cluster_interval, holm, prediction_metrics, select_example
from run_prego_single_channel import CONVERTED_DIR, load_recording


def check_recording(output, recording):
    with threadpool_limits(limits=1):
        root = output / "cache" / recording
        manifest = json.loads((root / "decode_complete.json").read_text())
        n = 0
        for item in manifest["analyses"]:
            if item["status"] != "included":
                continue
            folder = root / item["analysis"]
            with np.load(folder / "features.npz") as data:
                z = {key: data[key] for key in data.files}
            pred = heldout_predictions(z["power"], z["frequency"], z["bands"], z["torus"], z["labels"], z["folds"])
            saved = pd.read_csv(folder / "predictions.csv")
            for method, values in pred.items():
                np.testing.assert_array_equal(values, saved[saved.method == method].predicted_direction.values)
            for fold in range(5):
                model = json.loads((folder / f"fold_{fold}.json").read_text())
                training = np.array(model["training_trial_indices"])
                test = z["raw_trial_indices"][z["folds"] == fold]
                assert not set(training) & set(test)
                np.testing.assert_array_equal(training, z["raw_trial_indices"][z["folds"] != fold])
                assert model["n_training_trials"] == len(training)
                assert 500-(model["dimension"]-1)*model["tau"] >= 300
            np.testing.assert_array_equal(np.bincount(z["labels"])[1:], np.full(6, len(z["labels"])//6))
            assert len(np.unique(z["raw_trial_indices"])) == len(z["labels"])
            assert len(np.unique(z["original_trial_number"])) == len(z["labels"])
            n += 1
        return n


def validate(output, jobs=8):
    tables = load_tables(output)
    scores = tables["recording_scores"]
    predictions = tables["heldout_predictions"]
    keys = ["recording", "monkey", "day", "analysis", "method"]
    score_index = scores.set_index(keys)
    dir_index = tables["direction_scores"].set_index(keys+["direction"])
    matrix_index = tables["recording_confusions"].set_index(keys+["true_direction", "predicted_direction"])
    for values, group in predictions.groupby(keys):
        f1, per_direction, matrix = prediction_metrics(group.true_direction, group.predicted_direction)
        np.testing.assert_allclose(f1, score_index.loc[values, "macro_f1"], atol=1e-14)
        np.testing.assert_allclose(per_direction, dir_index.loc[values, "f1"].values, atol=1e-14)
        np.testing.assert_allclose(matrix, matrix_index.loc[values, "fraction"].values.reshape(6, 6), atol=1e-14)
    for (analysis, monkey, method), group in scores.groupby(["analysis", "monkey", "method"]):
        summary = tables["summary"]
        row = summary[(summary.analysis == analysis) & (summary.monkey == monkey) & (summary.method == method)].iloc[0]
        np.testing.assert_allclose([row["mean"], row.sd], [group.macro_f1.mean(), group.macro_f1.std()], atol=1e-14)
        assert row.n_lfps == group.recording.nunique() and row.n_days == group.day.nunique()
    main = scores[scores.analysis == "short"]
    for monkey, group in main.groupby("monkey"):
        cohorts = group.groupby("method").recording.apply(lambda x: tuple(sorted(x)))
        assert cohorts.nunique() == 1
        assert group.groupby("recording").n_trials.nunique().eq(1).all()
        eligible = tables["recording_audit"]
        eligible = eligible[eligible.main_included & (eligible.monkey == monkey)]
        example = tables["example_selection"].set_index("monkey").loc[monkey, "recording"]
        assert select_example(eligible) == example
        _, segments, good, _ = load_recording(CONVERTED_DIR / (example+".npz"))
        reference = json.loads((output / "cache" / example / "short" / "fold_0.json").read_text())
        training_ids = reference.pop("training_trial_indices")
        with threadpool_limits(limits=1):
            reproduced = learn_embedding(segments[np.searchsorted(good, training_ids)], stable_seed(example), 120)
        assert reproduced == reference, "Training-only reference embedding did not reproduce"
    matched = scores[scores.analysis.str.startswith("matched")]
    counts = matched.groupby(["recording", "analysis"]).n_trials.first().unstack()
    np.testing.assert_array_equal(counts.matched_short, counts.matched_long)
    audit = tables["recording_audit"]
    assert (audit.end_sample_exclusive == audit.go_sample_zero_based).all()
    assert ((audit.end_sample_exclusive - audit.start_sample) == 500).all()
    assert audit[audit.monkey == "M"].go_sample_zero_based.eq(5000).all()
    assert audit[audit.monkey == "T"].go_sample_zero_based.eq(4500).all()
    for row in tables["planned_comparisons"].itertuples():
        pair = tables["paired_differences"]
        pair = pair[(pair.monkey == row.monkey) & (pair.enhanced == row.enhanced) & (pair.baseline == row.baseline)]
        wide = main[main.monkey == row.monkey].pivot(index="recording", columns="method", values="macro_f1")
        expected = wide[row.enhanced]-wide[row.baseline]
        np.testing.assert_allclose(pair.set_index("recording").difference.sort_index(), expected.sort_index(), atol=1e-14)
        interval = cluster_interval(pair, stable_seed(row.monkey+row.enhanced+row.baseline))
        np.testing.assert_allclose(interval, [row.mean_difference, row.ci_low, row.ci_high], atol=1e-14)
        day = pair.groupby("day").difference.mean().round(12)
        p = wilcoxon(day, alternative="two-sided", zero_method="wilcox", method="auto").pvalue if np.any(day != 0) else 1.
        np.testing.assert_allclose(p, row.p_raw, atol=1e-12)
    tests = tables["planned_comparisons"]
    assert len(tests) == 8
    np.testing.assert_allclose(holm(tests.p_raw), tests.p_holm, atol=1e-12)
    for _, group in tables["mean_confusions"].groupby(["analysis", "monkey", "method", "true_direction"]):
        np.testing.assert_allclose(group.fraction.sum(), 1, atol=1e-12)
    records = main.recording.unique().tolist()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        checked = Parallel(n_jobs=jobs)(delayed(check_recording)(output, r) for r in records)
    names = ["A_task_window"]
    for monkey in "MT":
        names += [f"B_geometry_{monkey}_{r}" for r in ("R1", "R2", "r")]
        names += [f"B_preview_{monkey}"]
        names += [f"C_confusion_{monkey}_{m}" for m in ("power_frequency", "torus")]
        names += [f"{p}_{name}_{monkey}" for p, name in (("D", "decoding"), ("E", "added_value"), ("F", "direction_f1"),
                                                            ("S1", "delay_robustness"), ("S2", "dimensionality"))]
    export_info = []
    for name in names:
        folder = output / "panels"
        for suffix in ("png", "pdf", "svg", "csv", "caption.txt"):
            assert (folder / f"{name}.{suffix}").stat().st_size > 20
        with Image.open(folder / f"{name}.png") as im:
            assert min(im.info["dpi"]) >= 599
            pixels = np.array(im.convert("RGB"))
            assert (pixels < 220).any(axis=2).mean() > .01
        with fitz.open(folder / f"{name}.pdf") as doc:
            assert len(doc) == 1
            sizes = [s["size"] for b in doc[0].get_text("dict")["blocks"] if "lines" in b
                     for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            assert min(sizes) >= 7.99, (name, min(sizes))
            assert len(doc[0].get_drawings()) > 0
            assert not doc[0].get_images(), "Expected vector artwork"
            page = doc[0].rect
            for b in doc[0].get_text("dict")["blocks"]:
                if "lines" in b:
                    assert page.contains(fitz.Rect(b["bbox"])), (name, b["bbox"])
            export_info.append(dict(name=name, width_inches=page.width/72, height_inches=page.height/72,
                                    minimum_font_pt=min(sizes), vector_pdf=True, png_dpi=600))
        svg = (folder / f"{name}.svg").read_text()
        assert "<text" in svg and "<image" not in svg
    report = dict(status="passed", main_recordings=len(records), analysis_recordings=sum(checked),
        n_predictions=len(predictions), n_planned_tests=len(tests), exports=export_info,
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "h5py", "joblib")},
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            list(Path(__file__).resolve().parent.glob("*prego*.py")) +
            [Path(__file__).resolve().parent / name for name in ("motor_lfp_utils.py", "select_trace_embedding_parameters.py")]})
    (output / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "exports")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    validate(args.output, args.jobs)
