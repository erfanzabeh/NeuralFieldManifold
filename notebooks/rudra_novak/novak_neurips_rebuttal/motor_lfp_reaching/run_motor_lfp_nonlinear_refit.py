#!/usr/bin/env python
"""Rerun movement-direction decoding with nonlinear torus fits."""

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
from joblib import Parallel, delayed
from matplotlib import colors as mcolors
from scipy import stats
from tqdm import tqdm

from motor_lfp_utils import (
    BANDS,
    CACHE_DIR,
    CONVERTED_DIR,
    DIRECTION_LABELS,
    FEATURE_LABELS,
    FS,
    PLOT_DIR,
    RANDOM_SEED,
    TABLE_DIR,
    compute_average_psd,
    compute_band_power,
    decode_features,
    ensure_dirs,
    extract_epoch_segments,
    load_torus_param_table,
    nonlinear_torus_geometry_features,
    resolve_torus_params,
    target_values,
    valid_target_mask,
    write_csv,
)


EPOCH = "movement"
TARGET = "direction"
AVERAGE_PSD_FEATURE = "average_psd"
TORUS_FEATURE = "torus_nonlinear_15"
SINGLE_BAND_FEATURES = list(BANDS.keys())
FEATURE_ORDER = [*SINGLE_BAND_FEATURES, AVERAGE_PSD_FEATURE, "all_band_power", TORUS_FEATURE]
FEATURE_LABELS_REFIT = {
    **FEATURE_LABELS,
    AVERAGE_PSD_FEATURE: "Average\nPSD",
    TORUS_FEATURE: "Torus\nfeatures",
}
PAPER_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_red_scale",
    ["#f1d2ca", "#D85A30", "#7f1209"],
)
REVIEWER_F1_YLIM = 0.30


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


def table_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return TABLE_DIR / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def plot_path(name: str, suffix: str | None = None) -> Path:
    path = Path(name)
    return PLOT_DIR / "summary" / f"{path.stem}{suffix_token(suffix)}{path.suffix}"


def cache_path(lfp_uid: str, n_points: int, max_nfev: int, tau: int, embedding_dim: int) -> Path:
    embed = "" if embedding_dim == 3 else f"_embed{embedding_dim}"
    return CACHE_DIR / "nonlinear_refit" / f"{lfp_uid}_{EPOCH}_tau{tau}{embed}_pts{n_points}_nfev{max_nfev}.npz"


def safe_fit(
    segment: np.ndarray,
    seed: int,
    tau: int,
    embedding_dim: int,
    n_points: int,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        return nonlinear_torus_geometry_features(
            segment,
            fs=FS,
            tau=tau,
            embedding_dim=embedding_dim,
            n_points=n_points,
            max_nfev=max_nfev,
            seed=seed,
        )
    except Exception as exc:
        return np.full(15, np.nan, dtype=float), {"success": False, "reason": repr(exc), "cost": np.nan, "nfev": 0}


def compute_features_for_lfp(
    path: Path,
    force: bool,
    n_jobs: int,
    tau: int,
    embedding_dim: int,
    torus_param_source: str,
    torus_param_id: str,
    n_points: int,
    max_nfev: int,
) -> dict[str, Any]:
    out_path = cache_path(path.stem, n_points=n_points, max_nfev=max_nfev, tau=tau, embedding_dim=embedding_dim)
    if out_path.exists() and not force:
        with np.load(out_path, allow_pickle=True) as cached:
            keep = cached["keep"]
            band_power = cached["band_power"]
            torus = cached["torus"]
            fit_success = cached["fit_success"].astype(bool)
            fit_cost = cached["fit_cost"]
            fit_nfev = cached["fit_nfev"]
            cached_tau = int(cached["tau"]) if "tau" in cached.files else None
            cached_dim = int(cached["embedding_dim"]) if "embedding_dim" in cached.files else None
            if cached_tau not in (None, int(tau)) or cached_dim not in (None, int(embedding_dim)):
                raise ValueError(
                    f"Nonlinear cache parameter mismatch for {path.stem}: "
                    f"cache has tau={cached_tau}, dim={cached_dim}; requested tau={tau}, dim={embedding_dim}"
                )
            if "average_psd" in cached.files:
                average_psd = cached["average_psd"]
            else:
                with np.load(path, allow_pickle=True) as data:
                    segments, extracted_keep = extract_epoch_segments(data, EPOCH)
                if not np.array_equal(extracted_keep, keep):
                    raise ValueError(f"Cached keep indices no longer match extracted segments for {path.stem}")
                average_psd = compute_average_psd(segments, fs=FS)
                np.savez(
                    out_path,
                    keep=keep,
                    band_power=band_power,
                    average_psd=average_psd,
                    torus=torus,
                    fit_success=fit_success,
                    fit_cost=fit_cost,
                    fit_nfev=fit_nfev,
                    tau=np.asarray(tau, dtype=int),
                    embedding_dim=np.asarray(embedding_dim, dtype=int),
                    torus_param_source=np.asarray(torus_param_source),
                    torus_param_id=np.asarray(torus_param_id),
                    n_points=np.asarray(n_points, dtype=int),
                    max_nfev=np.asarray(max_nfev, dtype=int),
                )
            return {
                "lfp_uid": path.stem,
                "keep": keep,
                "band_power": band_power,
                "average_psd": average_psd,
                "torus": torus,
                "fit_success": fit_success,
                "fit_cost": fit_cost,
                "fit_nfev": fit_nfev,
                "torus_tau": int(tau),
                "torus_embedding_dim": int(embedding_dim),
                "torus_param_source": torus_param_source,
                "torus_param_id": torus_param_id,
            }

    with np.load(path, allow_pickle=True) as data:
        segments, keep = extract_epoch_segments(data, EPOCH)

    if len(segments) == 0:
        band_power = np.empty((0, len(BANDS)), dtype=float)
        average_psd = np.empty((0, 1), dtype=float)
        torus = np.empty((0, 15), dtype=float)
        success = np.array([], dtype=bool)
        costs = np.array([], dtype=float)
        nfev = np.array([], dtype=int)
    else:
        band_power = compute_band_power(segments, fs=FS)
        average_psd = compute_average_psd(segments, fs=FS)
        fit_results = Parallel(n_jobs=n_jobs)(
            delayed(safe_fit)(
                segment,
                RANDOM_SEED + i,
                tau,
                embedding_dim,
                n_points,
                max_nfev,
            )
            for i, segment in enumerate(segments)
        )
        torus = np.vstack([features for features, _meta in fit_results])
        success = np.asarray([bool(meta.get("success", False)) for _features, meta in fit_results], dtype=bool)
        costs = np.asarray([float(meta.get("cost", np.nan)) for _features, meta in fit_results], dtype=float)
        nfev = np.asarray([int(meta.get("nfev", 0)) for _features, meta in fit_results], dtype=int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        keep=keep,
        band_power=band_power,
        average_psd=average_psd,
        torus=torus,
        fit_success=success,
        fit_cost=costs,
        fit_nfev=nfev,
        tau=np.asarray(tau, dtype=int),
        embedding_dim=np.asarray(embedding_dim, dtype=int),
        torus_param_source=np.asarray(torus_param_source),
        torus_param_id=np.asarray(torus_param_id),
        n_points=np.asarray(n_points, dtype=int),
        max_nfev=np.asarray(max_nfev, dtype=int),
    )
    return {
        "lfp_uid": path.stem,
        "keep": keep,
        "band_power": band_power,
        "average_psd": average_psd,
        "torus": torus,
        "fit_success": success,
        "fit_cost": costs,
        "fit_nfev": nfev,
        "torus_tau": int(tau),
        "torus_embedding_dim": int(embedding_dim),
        "torus_param_source": torus_param_source,
        "torus_param_id": torus_param_id,
    }


def feature_matrix(feature_name: str, band_power: np.ndarray, average_psd: np.ndarray, torus: np.ndarray) -> np.ndarray:
    if feature_name in BANDS:
        return band_power[:, [list(BANDS).index(feature_name)]]
    if feature_name == AVERAGE_PSD_FEATURE:
        return average_psd
    if feature_name == "all_band_power":
        return band_power
    if feature_name == TORUS_FEATURE:
        return torus
    raise ValueError(f"Unknown feature set: {feature_name}")


def metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    return {
        "monkey": str(data["monkey"]),
        "session_id": str(data["session_id"]),
        "lfp_id": str(data["lfp_id"]),
    }


def decode_lfp(path: Path, features: dict[str, Any]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    with np.load(path, allow_pickle=True) as data:
        metadata = metadata_from_npz(data)
        keep = features["keep"]
        labels, classes = target_values(data, TARGET, keep)
        mask = valid_target_mask(data, TARGET, keep)
        band_power = features["band_power"][mask]
        average_psd = features["average_psd"][mask]
        torus = features["torus"][mask]
        success = features["fit_success"][mask]
        torus_tau = int(features["torus_tau"])
        torus_embedding_dim = int(features["torus_embedding_dim"])
        torus_param_source = str(features["torus_param_source"])
        torus_param_id = str(features["torus_param_id"])

    for feature_name in FEATURE_ORDER:
        x = feature_matrix(feature_name, band_power, average_psd, torus)
        result = decode_features(x, labels, classes=classes, standardize=True)
        row = {
            "lfp_uid": path.stem,
            **metadata,
            "epoch": EPOCH,
            "target": TARGET,
            "feature_set": feature_name,
            "status": result["status"],
            "accuracy": result["accuracy"],
            "f1": result["f1"],
            "n_classes": result["n_classes"],
            "n_trials_balanced": result["n_trials_balanced"],
            "per_class_n": result["per_class_n"],
            "torus_fit_success_fraction": float(np.mean(success)) if len(success) else np.nan,
            "torus_tau": torus_tau,
            "torus_tau_ms": float(torus_tau * 1000.0 / FS),
            "torus_embedding_dim": torus_embedding_dim,
            "torus_param_source": torus_param_source,
            "torus_param_id": torus_param_id,
        }
        rows.append(row)
        if result["status"] == "ok":
            confusions.append(
                {
                    "lfp_uid": path.stem,
                    **metadata,
                    "epoch": EPOCH,
                    "target": TARGET,
                    "feature_set": feature_name,
                    "torus_tau": torus_tau,
                    "torus_embedding_dim": torus_embedding_dim,
                    "torus_param_source": torus_param_source,
                    "torus_param_id": torus_param_id,
                    "class_labels": result["class_labels"],
                    "confusion": result["confusion"],
                }
            )
    return rows, confusions


def summarize(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    scored = scores.merge(manifest[["lfp_uid", "has_all_6_directions"]], on="lfp_uid", how="left")
    records = []
    for analysis_set, subset in [
        ("all_unique_lfps", scored),
        ("full_6_direction_lfps", scored[scored["has_all_6_directions"].astype(bool)]),
    ]:
        valid = subset[(subset["status"] == "ok") & subset["f1"].notna()]
        if valid.empty:
            continue
        grouped = (
            valid.groupby("feature_set", as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_f1=("f1", "mean"),
                std_f1=("f1", "std"),
                n_lfps=("lfp_uid", "nunique"),
                mean_balanced_trials=("n_trials_balanced", "mean"),
                mean_torus_fit_success=("torus_fit_success_fraction", "mean"),
            )
        )
        grouped.insert(0, "analysis_set", analysis_set)
        records.append(grouped)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def plot_bar(summary: pd.DataFrame, analysis_set: str, metric: str, suffix: str | None = None) -> None:
    sub = summary[summary["analysis_set"] == analysis_set].set_index("feature_set").reindex(FEATURE_ORDER)
    sub = sub.dropna(subset=[f"mean_{metric}"])
    if sub.empty:
        return
    means = sub[f"mean_{metric}"].to_numpy(float)
    stds = sub[f"std_{metric}"].fillna(0.0).to_numpy(float)
    norm = mcolors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = PAPER_RED_CMAP(norm(means))

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
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
    if metric == "accuracy":
        ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(x, [FEATURE_LABELS_REFIT[idx] for idx in sub.index])
    if metric == "f1" and analysis_set == "full_6_direction_lfps":
        ax.set_ylim(0, REVIEWER_F1_YLIM)
    else:
        ax.set_ylim(0, min(1.0, max(0.55, float(np.nanmax(means + stds)) + 0.08)))
    ax.set_ylabel("F1" if metric == "f1" else "Decoding accuracy")
    title_set = "All Unique LFPs" if analysis_set == "all_unique_lfps" else "Full 6-Direction LFPs"
    ax.set_title(f"Reach Direction Decoding From Movement LFP ({title_set})", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.014, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = plot_path(f"direction_movement_feature_{metric}_barplot_nonlinear_torus_{analysis_set}.png", suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mean_confusion(confusions: list[dict[str, object]], manifest: pd.DataFrame, suffix: str | None = None) -> None:
    full6 = set(manifest.loc[manifest["has_all_6_directions"].astype(bool), "lfp_uid"])
    selected = [
        rec
        for rec in confusions
        if rec["feature_set"] == TORUS_FEATURE
        and rec["lfp_uid"] in full6
        and list(rec["class_labels"]) == DIRECTION_LABELS
    ]
    if not selected:
        return
    mat = np.nanmean(np.stack([rec["confusion"] for rec in selected]), axis=0)
    fig, ax = plt.subplots(figsize=(5.7, 5.1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=PAPER_RED_CMAP)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] > 0.58 else "#1f1715", fontsize=10)
    ax.set_xticks(np.arange(len(DIRECTION_LABELS)), DIRECTION_LABELS)
    ax.set_yticks(np.arange(len(DIRECTION_LABELS)), DIRECTION_LABELS)
    ax.set_xlabel("Predicted direction")
    ax.set_ylabel("True direction")
    ax.set_title("Torus Features: Mean Direction Confusion", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Row-normalized accuracy")
    fig.tight_layout()
    out = plot_path("direction_movement_torus_nonlinear_mean_confusion.png", suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame(mat, index=[f"true_{x}" for x in DIRECTION_LABELS], columns=[f"pred_{x}" for x in DIRECTION_LABELS])
    write_csv(table.reset_index(names="true_label"), table_path("direction_movement_torus_nonlinear_mean_confusion.csv", suffix))


def star_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def holm_correct(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()


def feature_display_name(feature_set: str) -> str:
    return FEATURE_LABELS_REFIT.get(feature_set, feature_set).replace("\n", " ")


def build_best_band_comparison(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    min_lfps: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = scores.merge(
        manifest[["lfp_uid", "has_all_6_directions"]],
        on="lfp_uid",
        how="left",
    )
    valid = scored[
        scored["has_all_6_directions"].astype(bool)
        & (scored["status"] == "ok")
        & scored["f1"].notna()
    ].copy()

    comparison_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    skip_rows: list[dict[str, object]] = []

    def add_group(
        analysis_level: str,
        analysis_label: str,
        subset: pd.DataFrame,
        monkey: str | None = None,
        session_id: str | None = None,
    ) -> None:
        n_lfps = int(subset["lfp_uid"].nunique())
        if n_lfps < min_lfps:
            skip_rows.append(
                {
                    "analysis_level": analysis_level,
                    "analysis_label": analysis_label,
                    "monkey": monkey,
                    "session_id": session_id,
                    "n_lfps": n_lfps,
                    "reason": f"fewer than {min_lfps} full-six-direction LFPs",
                }
            )
            return

        band_means = subset[subset["feature_set"].isin(SINGLE_BAND_FEATURES)].groupby("feature_set")["f1"].mean()
        if band_means.empty:
            skip_rows.append(
                {
                    "analysis_level": analysis_level,
                    "analysis_label": analysis_label,
                    "monkey": monkey,
                    "session_id": session_id,
                    "n_lfps": n_lfps,
                    "reason": "no single-band scores",
                }
            )
            return
        ordered_means = band_means.reindex(SINGLE_BAND_FEATURES)
        best_band = str(ordered_means.idxmax())
        selection_rows.append(
            {
                "analysis_level": analysis_level,
                "analysis_label": analysis_label,
                "monkey": monkey,
                "session_id": session_id,
                "n_lfps": n_lfps,
                "selected_band": best_band,
                "selected_band_label": feature_display_name(best_band),
                "selected_band_mean_f1": float(ordered_means.loc[best_band]),
            }
        )

        needed = [TORUS_FEATURE, AVERAGE_PSD_FEATURE, best_band]
        wide = (
            subset[subset["feature_set"].isin(needed)]
            .pivot_table(
                index=["lfp_uid", "monkey", "session_id", "lfp_id"],
                columns="feature_set",
                values="f1",
                aggfunc="mean",
            )
            .dropna(subset=needed)
            .reset_index()
        )
        if len(wide) < min_lfps:
            skip_rows.append(
                {
                    "analysis_level": analysis_level,
                    "analysis_label": analysis_label,
                    "monkey": monkey,
                    "session_id": session_id,
                    "n_lfps": int(len(wide)),
                    "reason": f"fewer than {min_lfps} matched LFPs after requiring all comparison features",
                }
            )
            return

        role_map = [
            ("torus_features", "Torus features", TORUS_FEATURE),
            ("average_psd", "Average PSD", AVERAGE_PSD_FEATURE),
            ("best_single_band", "Best single band", best_band),
        ]
        for _, row in wide.iterrows():
            for role, role_label, feature_set in role_map:
                comparison_rows.append(
                    {
                        "analysis_level": analysis_level,
                        "analysis_label": analysis_label,
                        "monkey": row["monkey"],
                        "session_id": row["session_id"],
                        "lfp_uid": row["lfp_uid"],
                        "lfp_id": row["lfp_id"],
                        "feature_role": role,
                        "feature_label": role_label,
                        "feature_set": feature_set,
                        "selected_band": best_band,
                        "selected_band_label": feature_display_name(best_band),
                        "f1": float(row[feature_set]),
                    }
                )

    add_group("full_6_direction_lfps", "Full 6-direction LFPs", valid)
    for monkey, subset in valid.groupby("monkey", sort=True):
        add_group("monkey", f"Monkey {monkey}", subset, monkey=str(monkey))
    for (monkey, session_id), subset in valid.groupby(["monkey", "session_id"], sort=True):
        add_group("session", f"Monkey {monkey} session {session_id}", subset, monkey=str(monkey), session_id=str(session_id))

    comparison = pd.DataFrame(comparison_rows)
    selection = pd.DataFrame(selection_rows)
    skips = pd.DataFrame(skip_rows)
    significance = compute_significance_table(comparison) if not comparison.empty else pd.DataFrame()
    return comparison, selection, significance, skips


def compute_significance_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [
        ("torus_features", "average_psd"),
        ("torus_features", "best_single_band"),
        ("best_single_band", "average_psd"),
    ]
    for (analysis_level, analysis_label), subset in comparison.groupby(["analysis_level", "analysis_label"], sort=False):
        wide = subset.pivot_table(index="lfp_uid", columns="feature_role", values="f1", aggfunc="mean")
        group_rows = []
        for left, right in pairs:
            paired = wide[[left, right]].dropna()
            statistic = np.nan
            p_value = np.nan
            if len(paired) >= 5:
                try:
                    statistic, p_value = stats.wilcoxon(paired[left], paired[right], zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    statistic, p_value = np.nan, 1.0
            group_rows.append(
                {
                    "analysis_level": analysis_level,
                    "analysis_label": analysis_label,
                    "comparison": f"{left} vs {right}",
                    "feature_left": left,
                    "feature_right": right,
                    "n_pairs": int(len(paired)),
                    "mean_left": float(paired[left].mean()) if len(paired) else np.nan,
                    "mean_right": float(paired[right].mean()) if len(paired) else np.nan,
                    "mean_difference_left_minus_right": float((paired[left] - paired[right]).mean()) if len(paired) else np.nan,
                    "wilcoxon_statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_uncorrected": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
        corrected = holm_correct([row["p_uncorrected"] for row in group_rows])
        for row, p_holm in zip(group_rows, corrected):
            row["p_holm"] = p_holm
            row["significance"] = star_label(p_holm)
            rows.append(row)
    return pd.DataFrame(rows)


def add_significance_bracket(ax: plt.Axes, x1: int, x2: int, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#1f1715", lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=10, fontweight="bold")


def plot_comparison_axis(
    ax: plt.Axes,
    subset: pd.DataFrame,
    significance: pd.DataFrame,
    title: str,
    selected_band_label: str,
    fixed_ylim: float | None = None,
) -> None:
    roles = ["torus_features", "average_psd", "best_single_band"]
    labels = ["Torus\nfeatures", "Average\nPSD", f"Best band\n({selected_band_label})"]
    colors = ["#8b0000", "#6f6f6f", "#D85A30"]
    rng = np.random.default_rng(RANDOM_SEED)
    values = [subset.loc[subset["feature_role"] == role, "f1"].to_numpy(float) for role in roles]

    box = ax.boxplot(values, positions=np.arange(1, 4), widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#201715")
        patch.set_linewidth(0.8)
    for item in box["whiskers"] + box["caps"] + box["medians"]:
        item.set_color("#201715")
        item.set_linewidth(1.0)

    if fixed_ylim is None:
        for i, vals in enumerate(values, start=1):
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=13, color="#201715", alpha=0.38, linewidths=0)

    ax.axhline(1 / 6, ls="--", color="black", alpha=0.35, lw=1)
    ax.set_xticks(np.arange(1, 4), labels)
    ax.set_ylabel("F1")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, fixed_ylim if fixed_ylim is not None else 1.0)
    ax.grid(axis="y", color="#d8d0cc", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fixed_ylim is not None:
        height = fixed_ylim * 0.010
        text_offset = fixed_ylim * 0.002
        text_size = 8
        fixed_y = {
            "torus_features vs average_psd": fixed_ylim * 0.860,
            "torus_features vs best_single_band": fixed_ylim * 0.910,
            "best_single_band vs average_psd": fixed_ylim * 0.955,
        }
    else:
        y_base = max([np.nanmax(vals) if len(vals) else 0.0 for vals in values] + [1 / 6]) + 0.055
        step = 0.075
        height = 0.022
        text_offset = 0.0
        text_size = 10
        fixed_y = {}
    pair_positions = {
        "torus_features vs average_psd": (1, 2),
        "torus_features vs best_single_band": (1, 3),
        "best_single_band vs average_psd": (3, 2),
    }
    sig_lookup = significance.set_index("comparison") if not significance.empty else pd.DataFrame()
    for k, (comparison, (x1, x2)) in enumerate(pair_positions.items()):
        text = "n/a"
        if not sig_lookup.empty and comparison in sig_lookup.index:
            text = str(sig_lookup.loc[comparison, "significance"])
        if fixed_ylim is not None:
            y = fixed_y[comparison]
        else:
            y = y_base + k * step
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#1f1715", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2, y + height + text_offset, text, ha="center", va="bottom", fontsize=text_size, fontweight="bold")
    if fixed_ylim is None:
        ax.set_ylim(0, min(1.15, y_base + len(pair_positions) * step + 0.08))
    else:
        ax.set_ylim(0, fixed_ylim)


def plot_best_band_comparisons(
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    significance: pd.DataFrame,
    suffix: str | None = None,
) -> None:
    if comparison.empty or selection.empty:
        return
    (PLOT_DIR / "summary").mkdir(parents=True, exist_ok=True)

    main_label = "Full 6-direction LFPs"
    main = comparison[
        (comparison["analysis_level"] == "full_6_direction_lfps")
        & (comparison["analysis_label"] == main_label)
    ]
    main_sig = significance[
        (significance["analysis_level"] == "full_6_direction_lfps")
        & (significance["analysis_label"] == main_label)
    ]
    main_sel = selection[
        (selection["analysis_level"] == "full_6_direction_lfps")
        & (selection["analysis_label"] == main_label)
    ]
    if not main.empty and not main_sel.empty:
        fig, ax = plt.subplots(figsize=(4.2, 5.1))
        plot_comparison_axis(
            ax,
            main,
            main_sig,
            "Reach Direction Decoding",
            str(main_sel.iloc[0]["selected_band_label"]),
            fixed_ylim=REVIEWER_F1_YLIM,
        )
        fig.tight_layout()
        fig.savefig(plot_path("direction_movement_torus_avgpsd_bestband_f1_boxplot.png", suffix), dpi=240, bbox_inches="tight")
        plt.close(fig)

    monkey_labels = selection.loc[selection["analysis_level"] == "monkey", "analysis_label"].tolist()
    if monkey_labels:
        fig, axes = plt.subplots(1, len(monkey_labels), figsize=(4.2 * len(monkey_labels), 5.1), squeeze=False)
        for ax, label in zip(axes.ravel(), monkey_labels):
            sub = comparison[(comparison["analysis_level"] == "monkey") & (comparison["analysis_label"] == label)]
            sig = significance[(significance["analysis_level"] == "monkey") & (significance["analysis_label"] == label)]
            sel = selection[(selection["analysis_level"] == "monkey") & (selection["analysis_label"] == label)].iloc[0]
            plot_comparison_axis(ax, sub, sig, label, str(sel["selected_band_label"]))
        fig.tight_layout()
        fig.savefig(plot_path("direction_movement_torus_avgpsd_bestband_f1_boxplot_by_monkey.png", suffix), dpi=240, bbox_inches="tight")
        plt.close(fig)

    session_dir = PLOT_DIR / "summary" / f"per_session_bestband{suffix_token(suffix)}"
    session_dir.mkdir(parents=True, exist_ok=True)
    for _, sel in selection[selection["analysis_level"] == "session"].iterrows():
        label = str(sel["analysis_label"])
        sub = comparison[(comparison["analysis_level"] == "session") & (comparison["analysis_label"] == label)]
        sig = significance[(significance["analysis_level"] == "session") & (significance["analysis_label"] == label)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 5.1))
        plot_comparison_axis(ax, sub, sig, label, str(sel["selected_band_label"]))
        slug = str(label).replace(" ", "_").replace("/", "-")
        fig.tight_layout()
        fig.savefig(session_dir / f"{slug}_torus_avgpsd_bestband_f1_boxplot.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def write_interpretation(
    summary: pd.DataFrame,
    suffix: str | None = None,
    embedding_dim: int = 3,
    torus_params_csv: str | None = None,
) -> None:
    path = table_path("nonlinear_refit_interpretation.md", suffix)
    if summary.empty:
        path.write_text("No nonlinear torus refit results were available.\n")
        return
    lines = [
        "# Torus Feature Refit Interpretation",
        "",
        "This rerun replaces the earlier PCA torus proxy with torus features from an elliptical torus least-squares fit for movement-aligned LFP epochs.",
        (
            f"The delay cloud uses per-LFP tau and embedding dimension from {torus_params_csv}; dimensions above 3 are projected to their leading three principal coordinates before applying the same 3D elliptical torus fit."
            if torus_params_csv
            else f"The delay cloud uses lag-embedding dimension {embedding_dim}; dimensions above 3 are projected to their leading three principal coordinates before applying the same 3D elliptical torus fit."
        ),
        "",
    ]
    for analysis_set in ["all_unique_lfps", "full_6_direction_lfps"]:
        sub = summary[summary["analysis_set"] == analysis_set].set_index("feature_set")
        if TORUS_FEATURE not in sub.index:
            continue
        torus = sub.loc[TORUS_FEATURE]
        average_psd = sub.loc[AVERAGE_PSD_FEATURE] if AVERAGE_PSD_FEATURE in sub.index else None
        band = sub.loc["all_band_power"] if "all_band_power" in sub.index else None
        name = "all unique LFPs" if analysis_set == "all_unique_lfps" else "full six-direction LFPs"
        lines.append(
            f"For {name}, torus features reached F1={torus['mean_f1']:.3f} +/- {torus['std_f1']:.3f} "
            f"across {int(torus['n_lfps'])} LFPs."
        )
        if average_psd is not None:
            lines.append(f"The average-PSD baseline reached F1={average_psd['mean_f1']:.3f} +/- {average_psd['std_f1']:.3f}.")
        if band is not None:
            lines.append(f"The all-band spectral baseline reached F1={band['mean_f1']:.3f} +/- {band['std_f1']:.3f}.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-lfps", type=int, default=None)
    parser.add_argument("--monkeys", nargs="+", default=None, choices=["T", "M"])
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--torus-params-csv", default=None)
    parser.add_argument("--n-torus-points", type=int, default=300)
    parser.add_argument("--max-nfev", type=int, default=1200)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    ensure_dirs()
    paths = load_converted_files(max_lfps=args.max_lfps, monkeys=args.monkeys)
    if not paths:
        raise FileNotFoundError(f"No converted .npz files found in {CONVERTED_DIR}; run convert_motor_lfp.py first.")
    manifest = pd.read_csv(TABLE_DIR / "lfp_manifest.csv")
    torus_param_table = load_torus_param_table(args.torus_params_csv)

    rows: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    for path in tqdm(paths, desc="Nonlinear torus refit"):
        torus_params = resolve_torus_params(path.stem, torus_param_table, args.tau, args.embedding_dim)
        resolved_tau = int(torus_params["torus_tau"])
        resolved_dim = int(torus_params["torus_embedding_dim"])
        param_source = str(torus_params.get("torus_param_source", "cli_default"))
        param_id = str(torus_params.get("torus_param_id", f"tau{resolved_tau}_embed{resolved_dim}"))
        features = compute_features_for_lfp(
            path,
            force=args.force,
            n_jobs=args.n_jobs,
            tau=resolved_tau,
            embedding_dim=resolved_dim,
            torus_param_source=param_source,
            torus_param_id=param_id,
            n_points=args.n_torus_points,
            max_nfev=args.max_nfev,
        )
        file_rows, file_confusions = decode_lfp(path, features)
        rows.extend(file_rows)
        confusions.extend(file_confusions)

    scores = pd.DataFrame(rows)
    write_csv(scores, table_path("nonlinear_refit_direction_movement_scores.csv", args.output_suffix))
    summary = summarize(scores, manifest)
    write_csv(summary, table_path("nonlinear_refit_direction_movement_summary.csv", args.output_suffix))

    serializable = []
    for rec in confusions:
        serializable.append({**{k: v for k, v in rec.items() if k != "confusion"}, "confusion": np.asarray(rec["confusion"]).tolist()})
    conf_path = CACHE_DIR / f"nonlinear_refit_confusions{suffix_token(args.output_suffix)}.json"
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(json.dumps(serializable, indent=2))

    for analysis_set in ["all_unique_lfps", "full_6_direction_lfps"]:
        for metric in ["f1", "accuracy"]:
            plot_bar(summary, analysis_set=analysis_set, metric=metric, suffix=args.output_suffix)
    plot_mean_confusion(confusions, manifest, suffix=args.output_suffix)
    comparison, selection, significance, skips = build_best_band_comparison(scores, manifest)
    write_csv(comparison, table_path("torus_avgpsd_bestband_f1_scores.csv", args.output_suffix))
    write_csv(selection, table_path("relevant_band_selection.csv", args.output_suffix))
    write_csv(significance, table_path("torus_avgpsd_bestband_significance.csv", args.output_suffix))
    write_csv(skips, table_path("torus_avgpsd_bestband_session_skips.csv", args.output_suffix))
    plot_best_band_comparisons(comparison, selection, significance, suffix=args.output_suffix)
    write_interpretation(
        summary,
        suffix=args.output_suffix,
        embedding_dim=args.embedding_dim,
        torus_params_csv=args.torus_params_csv,
    )

    print(f"Decoded {scores['lfp_uid'].nunique()} unique LFP recordings with nonlinear torus fits.")
    print(f"Wrote {table_path('nonlinear_refit_direction_movement_summary.csv', args.output_suffix)}")
    print(f"Wrote nonlinear refit plots under {PLOT_DIR / 'summary'}")


if __name__ == "__main__":
    main()
