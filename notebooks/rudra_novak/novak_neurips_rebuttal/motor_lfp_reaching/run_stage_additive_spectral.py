"""Matched spectral and additive decoding from frozen corrected six-stage windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal
from threadpoolctl import threadpool_limits

from motor_lfp_utils import BANDS
from prego_decoding import make_estimator
from stage_geometry_decoding import OUTPUT as SOURCE, STAGES, validate_groups, scores

OUTPUT = SOURCE.parent / "stage_T_y070316009_12_ch7_tau3_stage_additive_spectral_v1"
METHODS = ("relevant_band", "geometry_relevant_band", "average_psd",
           "geometry_average_psd", "geometry", "all_bands", "geometry_all_bands")
MAIN = METHODS[:5]
BANDS_HZ = tuple((name, *bounds) for name, bounds in BANDS.items())
LABELS = {
    "relevant_band": "Relevant band\n(13-30 Hz)",
    "geometry_relevant_band": "Torus +\nrelevant band",
    "average_psd": "Average\nPSD",
    "geometry_average_psd": "Torus +\naverage PSD",
    "geometry": "Torus\nfeatures",
    "all_bands": "All band\npowers",
    "geometry_all_bands": "Torus +\nall bands",
}
COLORS = {
    "relevant_band": "#E1937A",
    "geometry_relevant_band": "#A12E29",
    "average_psd": "#A8A9A7",
    "geometry_average_psd": "#A12E29",
    "geometry": "#7F211C",
    "all_bands": "#777B79",
    "geometry_all_bands": "#A12E29",
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def protected_files():
    return [SOURCE / name for name in (
        "config.json", "features.npz", "inputs/windows.npz",
        "inputs/window_metadata.csv", "heldout_predictions.npz", "bootstrap.npz")]


def load_inputs():
    config = json.loads((SOURCE / "config.json").read_text())
    if (config["m"], config["tau_ms"], config["samples_per_window"],
            config["points_per_cloud"]) != (3, 3, 300, 294):
        raise ValueError("Expected frozen 300-ms native tau=3 geometry")
    if tuple(config["stage_names"]) != STAGES or tuple(config["stage_anchors"]) != (
            "TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO"):
        raise ValueError("Expected corrected cue-offset stages")
    with np.load(SOURCE / "features.npz") as data:
        geometry = data["x"].copy()
        labels = data["labels"].copy()
        folds = data["folds"].copy()
        trial_ids = data["trial_ids"].copy()
    with np.load(SOURCE / "inputs/windows.npz") as data:
        raw = data["raw"].copy()
    metadata = pd.read_csv(SOURCE / "inputs/window_metadata.csv")
    with np.load(SOURCE / "heldout_predictions.npz") as data:
        original = data["predictions"].copy()
        np.testing.assert_array_equal(data["labels"], labels)
        np.testing.assert_array_equal(data["folds"], folds)
        np.testing.assert_array_equal(data["trial_ids"], trial_ids)
    if (geometry.shape != (1188, 15) or raw.shape != (1188, 300)
            or len(metadata) != 1188 or original.shape != (1188,)
            or not np.isfinite(raw).all() or np.isinf(geometry).any()):
        raise ValueError("Saved corrected-stage input shapes changed")
    validate_groups(labels, folds, trial_ids)
    if not np.array_equal(trial_ids, np.repeat(np.arange(198), 6)):
        raise ValueError("Unexpected trial order")
    if not np.array_equal(labels, np.tile(np.arange(6), 198)):
        raise ValueError("Unexpected stage order")
    for name, values in (("window_index", np.arange(1188)), ("trial_index", trial_ids),
                         ("stage", labels), ("heldout_fold_zero_based", folds)):
        if not np.array_equal(metadata[name].to_numpy(), values):
            raise ValueError(f"Window metadata mismatch: {name}")
    if (metadata.stage_name.to_numpy() != np.asarray(STAGES)[labels]).any():
        raise ValueError("Stage names differ from saved labels")
    if not metadata.delay.eq("short").all():
        raise ValueError("Non-short-delay trial detected")
    if (metadata.end_sample_exclusive - metadata.start_sample).ne(300).any():
        raise ValueError("Window duration changed")
    return raw, geometry, labels, folds, trial_ids, original, metadata


def spectral_features(raw):
    """Use the previous additive comparison's raw-signal Welch definition at 300 ms."""
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != 300 or not np.isfinite(x).all():
        raise ValueError("Expected finite raw 300-sample windows")
    x = signal.detrend(x, axis=1, type="linear")
    frequency, density = signal.welch(x, fs=1000, window="hann", nperseg=300,
                                      noverlap=150, nfft=10000, detrend=False, axis=1)
    bands = []
    for name, low, high in BANDS_HZ:
        mask = (frequency >= low) & (frequency <= high)
        if mask.sum() < 2:
            raise ValueError(f"Insufficient Welch bins in {name}")
        integral = np.trapezoid(density[:, mask], frequency[mask], axis=1)
        bands.append(np.log10(np.maximum(integral, 1e-30)))
    bands = np.column_stack(bands)
    mask = (frequency >= 2) & (frequency <= 55)
    average = np.log10(np.maximum(density[:, mask].mean(axis=1), 1e-30))[:, None]
    if not np.isfinite(bands).all() or not np.isfinite(average).all():
        raise ValueError("Nonfinite spectral feature")
    return bands, average


def feature_sets(geometry, bands, average):
    if geometry.shape != (1188, 15) or bands.shape != (1188, 5) or average.shape != (1188, 1):
        raise ValueError("Unexpected feature dimensions")
    beta = bands[:, [list(BANDS).index("beta")]]
    return {
        "relevant_band": beta,
        "geometry_relevant_band": np.column_stack((geometry, beta)),
        "average_psd": average,
        "geometry_average_psd": np.column_stack((geometry, average)),
        "geometry": geometry,
        "all_bands": bands,
        "geometry_all_bands": np.column_stack((geometry, bands)),
    }


def decode(x, labels, folds):
    pred = np.full(len(labels), -1, dtype=np.int8)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        model = make_estimator()
        if list(model.named_steps) != ["impute", "scale", "lda"]:
            raise ValueError("Decoder pipeline changed")
        model.fit(x[train], labels[train])
        pred[test] = model.predict(x[test])
    if not np.isin(pred, np.arange(6)).all():
        raise ValueError("Missing held-out predictions")
    return pred


def bootstrap_scores(labels, predictions, trial_ids):
    with np.load(SOURCE / "bootstrap.npz") as saved:
        sampled_trials = saved["trial_indices"].copy()
        old_geometry_scores = saved["macro_f1"].copy()
    if sampled_trials.shape != (2000, 198) or not np.isin(sampled_trials, np.arange(198)).all():
        raise ValueError("Unexpected frozen whole-trial bootstrap")
    rows = np.stack([np.flatnonzero(trial_ids == i) for i in range(198)])
    result = np.empty((len(sampled_trials), len(METHODS)))
    for index, sample in enumerate(sampled_trials):
        chosen = rows[sample].ravel()
        for j in range(len(METHODS)):
            result[index, j] = scores(labels[chosen], predictions[chosen, j])["macro_f1"]
    np.testing.assert_allclose(result[:, METHODS.index("geometry")],
                               old_geometry_scores, rtol=0, atol=1e-12)
    return result, sampled_trials


def analyze():
    source_hashes = {str(path.resolve()): sha256(path) for path in protected_files()}
    raw, geometry, labels, folds, trial_ids, original, metadata = load_inputs()
    bands, average = spectral_features(raw)
    features = feature_sets(geometry, bands, average)
    with threadpool_limits(limits=1):
        predictions = np.column_stack([decode(features[name], labels, folds) for name in METHODS])
    np.testing.assert_array_equal(predictions[:, METHODS.index("geometry")], original)
    bootstrap, sampled_trials = bootstrap_scores(labels, predictions, trial_ids)
    summary = []
    for j, name in enumerate(METHODS):
        metric = scores(labels, predictions[:, j])
        low, high = np.quantile(bootstrap[:, j], [.025, .975])
        summary.append({"method": name, "dimensions": features[name].shape[1],
                        "macro_f1": metric["macro_f1"], "ci_low": low, "ci_high": high,
                        "accuracy": metric["accuracy"], "n_trials": 198,
                        "n_windows": 1188})
    summary = pd.DataFrame(summary)
    original_score = scores(labels, original)["macro_f1"]
    np.testing.assert_allclose(summary.set_index("method").loc["geometry", "macro_f1"],
                               original_score, rtol=0, atol=1e-12)
    paired = []
    for added, baseline in (("geometry_relevant_band", "relevant_band"),
                            ("geometry_average_psd", "average_psd"),
                            ("geometry_all_bands", "all_bands")):
        delta = bootstrap[:, METHODS.index(added)] - bootstrap[:, METHODS.index(baseline)]
        observed = (summary.set_index("method").loc[added, "macro_f1"]
                    - summary.set_index("method").loc[baseline, "macro_f1"])
        low, high = np.quantile(delta, [.025, .975])
        paired.append({"added": added, "baseline": baseline,
                       "macro_f1_difference": observed, "ci_low": low, "ci_high": high})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tables").mkdir(exist_ok=True)
    config = {
        "recording": "monkeyT_session-y070316009-12_lfp-7",
        "task": "corrected six-stage decoding, 198 short-delay trials",
        "stage_anchors": ["TC_on", "TC_off", "SC_on", "SC_off", "GO", "GO"],
        "m": 3, "tau_ms": 3, "K_assumed": 1, "geometry_refit": False,
        "methods": list(METHODS), "spectral_input": "saved raw 300-ms windows, linearly detrended",
        "spectral_welch": {"fs_hz": 1000, "window": "hann", "nperseg": 300,
                           "noverlap": 150, "nfft": 10000, "detrend": False},
        "relevant_band": "log10 integrated beta PSD over 13-30 Hz, 1D",
        "average_psd": "log10 mean PSD density over 2-55 Hz, 1D",
        "all_bands": [dict(name=name, low_hz=low, high_hz=high) for name, low, high in BANDS_HZ],
        "decoder": "same five trial-grouped folds; training-only median imputation, StandardScaler, LDA(lsqr, shrinkage=auto)",
        "uncertainty": "2000 existing direction-stratified whole-trial bootstrap samples of held-out predictions; conditional on fitted models",
        "nominal_chance": 1 / 6, "source_hashes": source_hashes,
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "features.npz", **features, labels=labels, folds=folds,
                        trial_ids=trial_ids)
    np.savez_compressed(OUTPUT / "predictions.npz", predictions=predictions,
                        labels=labels, folds=folds, trial_ids=trial_ids)
    np.savez_compressed(OUTPUT / "bootstrap.npz", macro_f1=bootstrap,
                        trial_indices=sampled_trials)
    summary.to_csv(OUTPUT / "tables/summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(paired).to_csv(OUTPUT / "tables/paired_differences.csv",
                                 index=False, float_format="%.17g")
    spectral = metadata[["window_index", "trial_index", "stage", "stage_name", "direction"]].copy()
    for j, name in enumerate(BANDS):
        spectral[f"log10_{name}_power"] = bands[:, j]
    spectral["log10_average_psd"] = average[:, 0]
    spectral.to_csv(OUTPUT / "tables/spectral_features.csv", index=False, float_format="%.17g")
    heldout = metadata[["window_index", "trial_index", "stage", "stage_name", "direction",
                        "heldout_fold_zero_based"]].copy()
    for j, name in enumerate(METHODS):
        heldout[name] = predictions[:, j]
    heldout.to_csv(OUTPUT / "tables/heldout_predictions.csv", index=False)
    check = {str(path.resolve()): sha256(path) for path in protected_files()}
    if check != source_hashes:
        raise AssertionError("Frozen source changed")
    (OUTPUT / "validation.json").write_text(json.dumps({
        "passed": True, "source_hashes_unchanged": True, "geometry_predictions_exact": True,
        "geometry_bootstrap_reproduced": True, "n_trials": 198, "n_windows": 1188,
        "predictions_per_method": 1188, "n_methods": len(METHODS),
        "n_bootstrap": len(sampled_trials), "geometry_fitting_calls": 0}, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)
    return summary


def render(order, summary, filename, ylabel="Macro-F1"):
    selection = summary.set_index("method").loc[list(order)]
    if len(selection) != len(order) or not np.isfinite(selection[["macro_f1", "ci_low", "ci_high"]]).all().all():
        raise ValueError("Invalid saved summary")
    fig, ax = plt.subplots(figsize=(9.5 if len(order) == 7 else 7.5, 4.35), facecolor="white")
    x = np.arange(len(order), dtype=float)
    colors = [COLORS[method] for method in order]
    bars = ax.bar(x, selection.macro_f1.to_numpy(), width=.69, color=colors,
                  edgecolor="#54221D", linewidth=.55)
    low = selection.macro_f1.to_numpy() - selection.ci_low.to_numpy()
    high = selection.ci_high.to_numpy() - selection.macro_f1.to_numpy()
    ax.errorbar(x, selection.macro_f1, yerr=np.vstack((low, high)), fmt="none",
                ecolor="#352C2B", elinewidth=1.05, capsize=3, capthick=1.05, zorder=4)
    ax.axhline(1 / 6, color="#747C80", linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_ylim(0, max(.42, float(selection.ci_high.max()) + .10))
    ax.set_xlim(-.6, len(order) - .4)
    ax.set_xticks(x, [LABELS[name] for name in order], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("Six-stage decoding from one macaque LFP", fontsize=12,
                 fontweight="bold", loc="left", pad=9)
    ax.tick_params(axis="both", direction="out", length=3, width=.75, colors="#26343C")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#26343C")
    ax.spines["bottom"].set_color("#26343C")
    ax.grid(False)
    for rect, row in zip(bars, selection.itertuples()):
        ax.text(rect.get_x() + rect.get_width() / 2, row.ci_high + .012,
                f"{row.macro_f1:.3f}", ha="center", va="bottom", fontsize=8.5,
                color="#26343C")
    fig.subplots_adjust(left=.12, right=.98, top=.88, bottom=.23)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1
                                         or box.x1 > fig.bbox.width + 1
                                         or box.y1 > fig.bbox.height + 1):
            raise ValueError(f"Clipped plot text: {item.get_text()}")
    fig.savefig(filename, dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(filename) as saved:
        if not np.any(np.asarray(saved.convert("RGB")) < 240):
            raise ValueError("Blank comparison figure")


def plot_only():
    if not (OUTPUT / "validation.json").exists():
        raise ValueError("Analysis must complete before plotting")
    config = json.loads((OUTPUT / "config.json").read_text())
    if tuple(config["methods"]) != METHODS or config["geometry_refit"]:
        raise ValueError("Saved experiment configuration changed")
    summary_path = OUTPUT / "tables/summary.csv"
    summary_hash = sha256(summary_path)
    summary = pd.read_csv(summary_path)
    if summary.method.tolist() != list(METHODS):
        raise ValueError("Saved method ordering changed")
    if config["source_hashes"] != {str(path.resolve()): sha256(path) for path in protected_files()}:
        raise ValueError("Frozen source hashes changed")
    with np.load(OUTPUT / "predictions.npz") as saved_predictions:
        labels = saved_predictions["labels"]
        predictions = saved_predictions["predictions"]
    with np.load(OUTPUT / "bootstrap.npz") as saved_bootstrap:
        bootstrap = saved_bootstrap["macro_f1"]
    with np.load(OUTPUT / "features.npz") as saved_features:
        dimensions = [saved_features[name].shape[1] for name in METHODS]
    if predictions.shape != (1188, len(METHODS)) or bootstrap.shape != (2000, len(METHODS)):
        raise ValueError("Saved prediction or bootstrap shape changed")
    np.testing.assert_array_equal(summary.dimensions.to_numpy(), dimensions)
    for j, row in enumerate(summary.itertuples()):
        metric = scores(labels, predictions[:, j])
        np.testing.assert_allclose([row.macro_f1, row.accuracy],
                                   [metric["macro_f1"], metric["accuracy"]], rtol=0, atol=1e-12)
        np.testing.assert_allclose([row.ci_low, row.ci_high],
                                   np.quantile(bootstrap[:, j], [.025, .975]), rtol=0, atol=1e-12)
    render(MAIN, summary, OUTPUT / "stage_additive_five_methods.png", ylabel="F1")
    render(METHODS, summary, OUTPUT / "stage_additive_full_comparison.png")
    if summary_hash != sha256(summary_path):
        raise AssertionError("Plotting changed analysis scores")
    rows = ["# Corrected six-stage additive comparison", "",
            "Monkey T, session y070316009-12, channel 7; 198 short-delay trials, six corrected 300-ms stages each.",
            "", "| Feature set | Dimensions | Held-out macro-F1 | Conditional 95% interval |",
            "|---|---:|---:|---:|"]
    for row in summary.itertuples():
        rows.append(f"| {LABELS[row.method].replace(chr(10), ' ')} | {row.dimensions} | "
                    f"{row.macro_f1:.3f} | {row.ci_low:.3f}-{row.ci_high:.3f} |")
    rows += ["", "The [five-bar figure](stage_additive_five_methods.png) follows the requested layout; "
             "the [full seven-bar figure](stage_additive_full_comparison.png) also shows all-band powers "
             "and geometry added to all bands.",
             "The five-bar figure labels the vertical axis F1; its values are macro-averaged across the six stages.",
             "All methods use identical saved time windows and five trial-grouped folds. Torus features are the unchanged "
             "15D corrected fits; adding beta or average PSD gives 16D, and adding all five bands gives 20D. "
             "Spectral features use raw, linearly detrended "
             "windows; the torus features came from per-window-normalized, filtered windows. Beta is log10 integrated "
             "13-30 Hz power; average PSD is log10 mean 2-55 Hz density. Five band powers cover delta through low gamma. "
             "Hann/Welch uses 300 samples, 150 overlap, and 10,000-point FFT; zero padding does not improve "
             "the 300-ms frequency resolution.",
             "The dashed 1/6 line is nominal six-class chance, not a shuffled-label null. Whiskers are 95% "
             "percentile intervals from the same 2,000 direction-stratified whole-trial resamples of held-out "
             "predictions, conditional on fitted models. This is one channel in one recording; folds or windows "
             "are not independent biological replicates. No stars or pooled-animal inference are implied.",
             "Paired additive differences and their conditional intervals are saved in "
             "[paired_differences.csv](tables/paired_differences.csv). Beta, mean PSD, and the five band "
             "definitions follow the earlier macaque analysis. The geometry decoder reproduces the prior "
             "held-out predictions exactly. "
             "No geometric refit or Ripser run occurred. The beta and average-PSD features are not matched in "
             "dimensionality to the 15D geometry, so standalone scores are a representation comparison, not "
             "a controlled per-dimension efficiency test. The geometry-plus-all-bands check was added after "
             "the initial five-bar scores were inspected; its near-zero difference is descriptive, not a "
             "pre-registered hypothesis test. Low-frequency band estimates are limited by 300-ms windows.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(rows))
    (OUTPUT / "figure_manifest.json").write_text(json.dumps({
        "five_bar_sha256": sha256(OUTPUT / "stage_additive_five_methods.png"),
        "full_comparison_sha256": sha256(OUTPUT / "stage_additive_full_comparison.png"),
        "score_table_sha256": summary_hash}, indent=2) + "\n")
    print(f"Rendered comparisons in {OUTPUT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("analysis", "plot", "all"), default="all")
    args = parser.parse_args()
    if args.phase in ("analysis", "all"):
        analyze()
    if args.phase in ("plot", "all"):
        plot_only()


if __name__ == "__main__":
    main()
