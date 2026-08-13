#!/usr/bin/env python
"""Run reaching-condition decoding from macaque motor-cortex LFP features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from scipy import signal
from tqdm import tqdm

from motor_lfp_utils import (
    BANDS,
    CACHE_DIR,
    CONVERTED_DIR,
    DELAY_LABELS,
    DIRECTION_LABELS,
    EPOCHS,
    FEATURE_LABELS,
    FEATURE_ORDER,
    FS,
    PLOT_DIR,
    TABLE_DIR,
    compute_average_psd,
    compute_band_power,
    decode_features,
    ensure_dirs,
    extract_epoch_segments,
    load_torus_param_table,
    resolve_torus_params,
    target_values,
    torus_geometry_features,
    valid_target_mask,
    write_csv,
)


PAPER_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_red_scale",
    ["#f1d2ca", "#D85A30", "#7f1209"],
)

TARGETS = ["direction", "delay", "condition"]
SUMMARY_EPOCH = "movement"
SUMMARY_TARGET = "direction"
TARGET_TITLES = {
    "direction": "Reach Direction",
    "delay": "Delay Type",
    "condition": "Direction x Delay Condition",
}
EPOCH_LABELS = {
    "pre_go": "Pre-GO",
    "peri_go": "Peri-GO",
    "movement": "Movement",
    "post_go": "Post-GO",
}


def load_converted_files(max_lfps: int | None = None, monkeys: list[str] | None = None) -> list[Path]:
    paths = sorted(CONVERTED_DIR.glob("*.npz"))
    if monkeys:
        wanted = {f"monkey{monkey}" for monkey in monkeys}
        paths = [path for path in paths if any(path.stem.startswith(prefix) for prefix in wanted)]
    if max_lfps is not None:
        paths = paths[:max_lfps]
    return paths


def suffix_token(suffix: str | None) -> str:
    clean = str(suffix or "").strip().strip("_")
    return f"_{clean}" if clean else ""


def feature_cache_path(
    lfp_uid: str,
    epoch: str,
    tau: int,
    embedding_dim: int,
    cache_tag: str | None = None,
) -> Path:
    if cache_tag:
        embed = f"_{cache_tag}_tau{tau}_embed{embedding_dim}"
    elif tau == 20 and embedding_dim == 3:
        embed = ""
    elif tau == 20:
        embed = f"_embed{embedding_dim}"
    else:
        embed = f"_tau{tau}_embed{embedding_dim}"
    return CACHE_DIR / "features" / f"{lfp_uid}_{epoch}{embed}_features.npz"


def table_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def compute_epoch_features(
    path: Path,
    epoch: str,
    force: bool = False,
    torus_tau: int = 20,
    torus_embedding_dim: int = 3,
    torus_param_source: str = "cli_default",
    torus_param_id: str | None = None,
    cache_tag: str | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        lfp_uid = path.stem
        cache_path = feature_cache_path(lfp_uid, epoch, torus_tau, torus_embedding_dim, cache_tag=cache_tag)
        if cache_path.exists() and not force:
            with np.load(cache_path, allow_pickle=True) as cached:
                cached_tau = int(cached["torus_tau"]) if "torus_tau" in cached.files else None
                cached_dim = int(cached["torus_embedding_dim"]) if "torus_embedding_dim" in cached.files else None
                cache_matches = (
                    cached_tau in (None, int(torus_tau))
                    and cached_dim in (None, int(torus_embedding_dim))
                )
                if not cache_matches:
                    raise ValueError(
                        f"Feature cache parameter mismatch for {lfp_uid} {epoch}: "
                        f"cache has tau={cached_tau}, dim={cached_dim}; requested tau={torus_tau}, dim={torus_embedding_dim}"
                    )
                keep = cached["keep"]
                band_power = cached["band_power"]
                torus = cached["torus_geometry_15"]
                if "average_psd" in cached.files:
                    average_psd = cached["average_psd"]
                else:
                    segments, extracted_keep = extract_epoch_segments(data, epoch)
                    if not np.array_equal(extracted_keep, keep):
                        raise ValueError(f"Cached keep indices no longer match extracted segments for {lfp_uid} {epoch}")
                    average_psd = compute_average_psd(segments, fs=FS)
                    np.savez(
                        cache_path,
                        keep=keep,
                        band_power=band_power,
                        average_psd=average_psd,
                        torus_geometry_15=torus,
                        torus_tau=np.asarray(torus_tau, dtype=int),
                        torus_embedding_dim=np.asarray(torus_embedding_dim, dtype=int),
                        torus_param_source=np.asarray(torus_param_source),
                        torus_param_id=np.asarray(torus_param_id or f"tau{torus_tau}_embed{torus_embedding_dim}"),
                    )
                return {
                    "lfp_uid": lfp_uid,
                    "keep": keep,
                    "band_power": band_power,
                    "average_psd": average_psd,
                    "torus_geometry_15": torus,
                    "torus_tau": int(torus_tau),
                    "torus_embedding_dim": int(torus_embedding_dim),
                    "torus_param_source": torus_param_source,
                    "torus_param_id": torus_param_id or f"tau{torus_tau}_embed{torus_embedding_dim}",
                    "n_segments": int(band_power.shape[0]),
                }

        segments, keep = extract_epoch_segments(data, epoch)
        if len(segments) == 0:
            band_power = np.empty((0, len(BANDS)), dtype=float)
            average_psd = np.empty((0, 1), dtype=float)
            torus = np.empty((0, 15), dtype=float)
        else:
            band_power = compute_band_power(segments, fs=FS)
            average_psd = compute_average_psd(segments, fs=FS)
            torus = np.vstack(
                [
                    torus_geometry_features(seg, fs=FS, tau=torus_tau, embedding_dim=torus_embedding_dim)
                    for seg in segments
                ]
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            keep=keep,
            band_power=band_power,
            average_psd=average_psd,
            torus_geometry_15=torus,
            torus_tau=np.asarray(torus_tau, dtype=int),
            torus_embedding_dim=np.asarray(torus_embedding_dim, dtype=int),
            torus_param_source=np.asarray(torus_param_source),
            torus_param_id=np.asarray(torus_param_id or f"tau{torus_tau}_embed{torus_embedding_dim}"),
        )
        return {
            "lfp_uid": lfp_uid,
            "keep": keep,
            "band_power": band_power,
            "average_psd": average_psd,
            "torus_geometry_15": torus,
            "torus_tau": int(torus_tau),
            "torus_embedding_dim": int(torus_embedding_dim),
            "torus_param_source": torus_param_source,
            "torus_param_id": torus_param_id or f"tau{torus_tau}_embed{torus_embedding_dim}",
            "n_segments": int(len(keep)),
        }


def feature_matrix(feature_name: str, band_power: np.ndarray, average_psd: np.ndarray, torus: np.ndarray) -> np.ndarray:
    if feature_name in BANDS:
        return band_power[:, [list(BANDS).index(feature_name)]]
    if feature_name == "average_psd":
        return average_psd
    if feature_name == "all_band_power":
        return band_power
    if feature_name == "torus_geometry_15":
        return torus
    raise ValueError(f"Unknown feature set: {feature_name}")


def metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    return {
        "monkey": str(data["monkey"]),
        "session_id": str(data["session_id"]),
        "lfp_id": str(data["lfp_id"]),
    }


def decode_one_file(
    path: Path,
    epochs: list[str],
    targets: list[str],
    force_features: bool = False,
    torus_tau: int = 20,
    torus_embedding_dim: int = 3,
    torus_param_table: dict[str, dict[str, object]] | None = None,
    cache_tag: str | None = None,
    standardize: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    confusion_records: list[dict[str, object]] = []
    with np.load(path, allow_pickle=True) as data:
        metadata = metadata_from_npz(data)
        torus_params = resolve_torus_params(path.stem, torus_param_table, torus_tau, torus_embedding_dim)
        resolved_tau = int(torus_params["torus_tau"])
        resolved_dim = int(torus_params["torus_embedding_dim"])
        param_source = str(torus_params.get("torus_param_source", "cli_default"))
        param_id = str(torus_params.get("torus_param_id", f"tau{resolved_tau}_embed{resolved_dim}"))
        for epoch in epochs:
            features = compute_epoch_features(
                path,
                epoch=epoch,
                force=force_features,
                torus_tau=resolved_tau,
                torus_embedding_dim=resolved_dim,
                torus_param_source=param_source,
                torus_param_id=param_id,
                cache_tag=cache_tag,
            )
            keep = features["keep"]
            if len(keep) == 0:
                continue
            band_power = features["band_power"]
            average_psd = features["average_psd"]
            torus = features["torus_geometry_15"]
            for target in targets:
                labels, classes = target_values(data, target, keep)
                mask = valid_target_mask(data, target, keep)
                if mask.sum() != len(labels):
                    band_target = band_power[mask]
                    average_target = average_psd[mask]
                    torus_target = torus[mask]
                else:
                    band_target = band_power
                    average_target = average_psd
                    torus_target = torus

                for feature_name in FEATURE_ORDER:
                    x = feature_matrix(feature_name, band_target, average_target, torus_target)
                    result = decode_features(x, labels, classes=classes, standardize=standardize)
                    row = {
                        "lfp_uid": path.stem,
                        **metadata,
                        "epoch": epoch,
                        "target": target,
                        "feature_set": feature_name,
                        "status": result["status"],
                        "accuracy": result["accuracy"],
                        "f1": result["f1"],
                        "n_classes": result["n_classes"],
                        "n_trials_balanced": result["n_trials_balanced"],
                        "per_class_n": result["per_class_n"],
                        "torus_tau": resolved_tau,
                        "torus_tau_ms": float(resolved_tau * 1000.0 / FS),
                        "torus_embedding_dim": resolved_dim,
                        "torus_param_source": param_source,
                        "torus_param_id": param_id,
                    }
                    rows.append(row)
                    if result["status"] == "ok":
                        confusion_records.append(
                            {
                                "lfp_uid": path.stem,
                                **metadata,
                                "epoch": epoch,
                                "target": target,
                                "feature_set": feature_name,
                                "torus_tau": resolved_tau,
                                "torus_embedding_dim": resolved_dim,
                                "torus_param_source": param_source,
                                "torus_param_id": param_id,
                                "class_labels": result["class_labels"],
                                "confusion": result["confusion"],
                            }
                        )
    return rows, confusion_records


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    valid = scores[(scores["status"] == "ok") & scores["f1"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    summary = (
        valid.groupby(["target", "epoch", "feature_set"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            n_lfps=("lfp_uid", "nunique"),
            mean_balanced_trials=("n_trials_balanced", "mean"),
        )
        .sort_values(["target", "epoch", "feature_set"])
    )
    return summary


def split_score_tables(scores: pd.DataFrame, suffix: str | None = None) -> None:
    write_csv(scores[scores["feature_set"].isin(BANDS.keys())], table_path("single_band_decode_scores.csv", suffix))
    write_csv(scores[scores["feature_set"] == "average_psd"], table_path("average_psd_decode_scores.csv", suffix))
    write_csv(scores[scores["feature_set"] == "all_band_power"], table_path("multiband_decode_scores.csv", suffix))
    write_csv(scores[scores["feature_set"] == "torus_geometry_15"], table_path("torus_decode_scores.csv", suffix))
    write_csv(scores, table_path("all_feature_decode_scores.csv", suffix))


def plot_bar(summary: pd.DataFrame, target: str, epoch: str, metric: str, suffix: str | None = None) -> None:
    sub = summary[(summary["target"] == target) & (summary["epoch"] == epoch)].set_index("feature_set").reindex(FEATURE_ORDER)
    sub = sub.dropna(subset=[f"mean_{metric}"])
    if sub.empty:
        return

    means = sub[f"mean_{metric}"].to_numpy(float)
    stds = sub[f"std_{metric}"].fillna(0.0).to_numpy(float)
    norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = PAPER_RED_CMAP(norm(means))

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(sub))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#6f1009",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "ecolor": "#3a0a06", "capthick": 1.1},
    )
    chance = {"direction": 1 / 6, "delay": 1 / 2, "condition": 1 / 12}.get(target)
    if chance is not None and metric == "accuracy":
        ax.axhline(chance, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, [FEATURE_LABELS[idx] for idx in sub.index])
    ax.set_ylim(0, min(1.0, max(0.55, float(np.nanmax(means + stds)) + 0.08)))
    ax.set_ylabel("F1" if metric == "f1" else "Decoding accuracy")
    pretty_target = TARGET_TITLES.get(target, target.replace("_", " ").title())
    pretty_epoch = EPOCH_LABELS.get(epoch, epoch.replace("_", " ").title())
    ax.set_title(f"{pretty_target} Decoding From {pretty_epoch} LFP", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.014, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = plot_path(f"{target}_{epoch}_feature_{metric}_barplot.png", suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_heatmap(summary: pd.DataFrame, target: str, metric: str, suffix: str | None = None) -> None:
    sub = summary[summary["target"] == target].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="epoch", columns="feature_set", values=f"mean_{metric}").reindex(index=list(EPOCHS), columns=FEATURE_ORDER)
    if pivot.dropna(how="all").empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    im = ax.imshow(pivot.to_numpy(float), vmin=0.0, vmax=1.0, cmap=PAPER_RED_CMAP, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.58 else "#1f1715", fontsize=9)
    ax.set_xticks(np.arange(len(FEATURE_ORDER)), [FEATURE_LABELS[idx].replace("\n", " ") for idx in FEATURE_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(EPOCHS)), [EPOCH_LABELS.get(epoch, epoch.replace("_", " ").title()) for epoch in EPOCHS])
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Task epoch")
    pretty_target = TARGET_TITLES.get(target, target.replace("_", " ").title())
    ax.set_title(f"{pretty_target} Decoding Across Task Epochs", fontweight="bold")
    fig.colorbar(im, ax=ax, label="F1" if metric == "f1" else "Decoding accuracy")
    fig.tight_layout()
    out = plot_path(f"{target}_epoch_feature_{metric}_heatmap.png", suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def labels_for_target(target: str) -> list[Any]:
    if target == "direction":
        return DIRECTION_LABELS
    if target == "delay":
        return DELAY_LABELS
    return [f"D{d}_{delay}" for delay in DELAY_LABELS for d in DIRECTION_LABELS]


def plot_mean_confusion(confusions: list[dict[str, object]], target: str, epoch: str, feature_set: str, suffix: str | None = None) -> None:
    labels = labels_for_target(target)
    selected = [
        rec
        for rec in confusions
        if rec["target"] == target
        and rec["epoch"] == epoch
        and rec["feature_set"] == feature_set
        and list(rec["class_labels"]) == labels
    ]
    if not selected:
        return
    mat = np.nanmean(np.stack([rec["confusion"] for rec in selected]), axis=0)
    fig, ax = plt.subplots(figsize=(5.7, 5.1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=PAPER_RED_CMAP)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] > 0.58 else "#1f1715", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{FEATURE_LABELS[feature_set].replace(chr(10), ' ')}: Mean {target.title()} Confusion", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Row-normalized accuracy")
    fig.tight_layout()
    out = plot_path(f"{target}_{epoch}_{feature_set}_mean_confusion.png", suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame(mat, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    write_csv(table.reset_index(names="true_label"), table_path(f"{target}_{epoch}_{feature_set}_mean_confusion.csv", suffix))


def plot_task_sanity(paths: list[Path]) -> None:
    if not paths:
        return
    chosen = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            dirs = set(data["direction"].astype(int).tolist())
            delays = set(data["delay_label"].astype(str).tolist())
            if set(DIRECTION_LABELS).issubset(dirs) and {"short", "long"}.issubset(delays):
                chosen = path
                break
    if chosen is None:
        chosen = paths[0]

    with np.load(chosen, allow_pickle=True) as data:
        lfp = data["lfp"].astype(float)
        direction = data["direction"].astype(int)
        delay = data["delay_label"].astype(str)
        movement_onset = data["movement_onset_ms"].astype(float)
        go_sample = int(data["go_sample"])
        fs = int(data["sampling_rate_hz"])
        start = go_sample - 2000
        end = go_sample + 2000
        trials = lfp[: min(120, len(lfp)), start:end]
        freqs, times, sxx = signal.spectrogram(trials, fs=fs, nperseg=256, noverlap=220, axis=1)
        sxx_mean = 10 * np.log10(np.nanmean(sxx, axis=0) + 1e-12)

    counts = pd.DataFrame(0, index=DIRECTION_LABELS, columns=DELAY_LABELS, dtype=int)
    for d in DIRECTION_LABELS:
        for del_label in DELAY_LABELS:
            counts.loc[d, del_label] = int(np.sum((direction == d) & (delay == del_label)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [0.9, 1.4, 0.9]})
    im0 = axes[0].imshow(counts.to_numpy(), cmap=PAPER_RED_CMAP)
    axes[0].set_xticks(np.arange(len(DELAY_LABELS)), DELAY_LABELS)
    axes[0].set_yticks(np.arange(len(DIRECTION_LABELS)), DIRECTION_LABELS)
    axes[0].set_xlabel("Delay")
    axes[0].set_ylabel("Reach direction")
    axes[0].set_title("Trial Counts", fontweight="bold")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            axes[0].text(j, i, str(counts.iat[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    im1 = axes[1].pcolormesh(times - 2.0, freqs, sxx_mean, shading="auto", cmap="magma")
    axes[1].axvline(0, color="white", lw=1.2)
    axes[1].set_ylim(0, 80)
    axes[1].set_xlabel("Time from GO (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("LFP Spectrogram", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], label="Power (dB)")

    finite_onsets = movement_onset[np.isfinite(movement_onset)]
    axes[2].hist(finite_onsets, bins=25, color="#8b0000", alpha=0.82, edgecolor="white")
    axes[2].axvline(0, color="black", lw=1.0, alpha=0.5)
    axes[2].set_xlabel("Movement onset from GO (ms)")
    axes[2].set_ylabel("Trials")
    axes[2].set_title("Movement Onsets", fontweight="bold")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(chosen.stem, fontweight="bold")
    fig.tight_layout()
    out = PLOT_DIR / "summary" / "task_sanity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(
    summary: pd.DataFrame,
    suffix: str | None = None,
    torus_embedding_dim: int = 3,
    torus_params_csv: str | None = None,
) -> None:
    path = table_path("rebuttal_interpretation.md", suffix)
    if summary.empty:
        path.write_text("No valid motor-cortex LFP decoding results were available.\n")
        return
    sub = summary[(summary["target"] == SUMMARY_TARGET) & (summary["epoch"] == SUMMARY_EPOCH)].set_index("feature_set")
    lines = [
        "# Motor-Cortex LFP Reaching Interpretation",
        "",
        "This analysis tests the reviewer-suggested macaque reaching setting directly, using the published motor-cortex LFP dataset rather than adding an unrelated grid-cell discussion.",
        (
            f"The torus-feature row uses per-LFP tau and embedding dimension from {torus_params_csv}."
            if torus_params_csv
            else f"The torus-feature row uses lag-embedding dimension {torus_embedding_dim}."
        ),
        "",
    ]
    if "torus_geometry_15" in sub.index:
        torus = sub.loc["torus_geometry_15"]
        lines.append(
            f"For movement-aligned reach-direction decoding, torus geometry features reached F1={torus['mean_f1']:.3f} "
            f"+/- {torus['std_f1']:.3f} across {int(torus['n_lfps'])} unique LFP recordings."
        )
    if "all_band_power" in sub.index:
        band = sub.loc["all_band_power"]
        lines.append(
            f"The multi-band spectral baseline reached F1={band['mean_f1']:.3f} +/- {band['std_f1']:.3f}, "
            "with single-band baselines reported separately in the accompanying CSV and bar plots."
        )
    lines.extend(
        [
            "",
            "Interpretation for the rebuttal should emphasize that this is a motor-cortex reaching analysis with the dataset's documented six reach directions and short/long delays. The result is intended as an empirical stress test of whether delay-geometry features carry task information beyond conventional spectral power, not as a claim about grid-cell attractor topology.",
            "",
            "Large converted arrays and feature caches are stored locally under git-ignored folders; the committed artifacts are scripts, notebooks, compact tables, and final plots.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--max-lfps", type=int, default=None)
    parser.add_argument("--monkeys", nargs="+", default=None, choices=["T", "M"])
    parser.add_argument("--epochs", nargs="+", default=list(EPOCHS), choices=list(EPOCHS))
    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    parser.add_argument("--torus-tau", type=int, default=20)
    parser.add_argument("--torus-embedding-dim", type=int, default=3)
    parser.add_argument("--torus-params-csv", default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    paths = load_converted_files(max_lfps=args.max_lfps, monkeys=args.monkeys)
    if not paths:
        raise FileNotFoundError(f"No converted .npz files found in {CONVERTED_DIR}; run convert_motor_lfp.py first.")
    torus_param_table = load_torus_param_table(args.torus_params_csv)
    cache_tag = suffix_token(args.output_suffix).strip("_") if torus_param_table else None

    rows: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    for path in tqdm(paths, desc="Decoding LFPs"):
        file_rows, file_confusions = decode_one_file(
            path,
            epochs=args.epochs,
            targets=args.targets,
            force_features=args.force_features,
            torus_tau=args.torus_tau,
            torus_embedding_dim=args.torus_embedding_dim,
            torus_param_table=torus_param_table,
            cache_tag=cache_tag,
            standardize=not args.no_standardize,
        )
        rows.extend(file_rows)
        confusions.extend(file_confusions)

    scores = pd.DataFrame(rows)
    split_score_tables(scores, suffix=args.output_suffix)
    summary = summarize_scores(scores)
    write_csv(summary, table_path("summary_decode_scores.csv", args.output_suffix))

    conf_path = CACHE_DIR / f"confusions{suffix_token(args.output_suffix)}.json"
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for rec in confusions:
        serializable.append({**{k: v for k, v in rec.items() if k != "confusion"}, "confusion": np.asarray(rec["confusion"]).tolist()})
    conf_path.write_text(json.dumps(serializable, indent=2))

    plot_task_sanity(paths)
    for target in args.targets:
        for metric in ("f1", "accuracy"):
            plot_epoch_heatmap(summary, target=target, metric=metric, suffix=args.output_suffix)
    for metric in ("f1", "accuracy"):
        plot_bar(summary, target=SUMMARY_TARGET, epoch=SUMMARY_EPOCH, metric=metric, suffix=args.output_suffix)
    for feature_set in ("torus_geometry_15", "all_band_power", "beta", "low_gamma"):
        plot_mean_confusion(confusions, target=SUMMARY_TARGET, epoch=SUMMARY_EPOCH, feature_set=feature_set, suffix=args.output_suffix)
    write_interpretation(
        summary,
        suffix=args.output_suffix,
        torus_embedding_dim=args.torus_embedding_dim,
        torus_params_csv=args.torus_params_csv,
    )

    print(f"Decoded {scores['lfp_uid'].nunique()} unique LFP recordings.")
    print(f"Wrote {table_path('summary_decode_scores.csv', args.output_suffix)}")
    print(f"Wrote plots under {PLOT_DIR / 'summary'}")


if __name__ == "__main__":
    main()
