from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"
DATA_PATH = UNIT_DIR.parent.parent / "data" / "monkey_lfp.npz"
SOURCE_BETTI = Path("/home/dev/NeuralFieldManifold/notebooks/rudra_novak/betti_sweep.npy")
DIAGRAM_CACHE = CACHE_DIR / "full_trace_ripser_dgms.npz"
SHUFFLE_BETTI_SWEEP_DIR = UNIT_DIR.parent / "nulls" / "cache" / "betti_sweeps"
BETTI_TAU = 40
BETTI_WINDOW = 2000
BETTI_N_SUBSAMPLE = 300
BETTI_SEED = 42
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_REPEATS = 5000

SHUFFLES = [
    ("fourier_phase_shuffle", "Fourier phase shuffle", "#2f6f9f"),
    ("iaaft_shuffle", "IAAFT shuffle", "#4b8f5a"),
    ("aperiodic_1f", "Aperiodic 1/f shuffle", "#d1892f"),
    ("envelope_phase_reset", "Envelope phase reset", "#7e62a3"),
    ("one_mode_ablation", "One-mode ablation", "#5c677d"),
]


def lag_embed(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("Trace is too short for requested lag embedding")
    return np.column_stack([x[i * tau:i * tau + n] for i in range(dim)])


def normalize_cloud(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / (norms + 1e-12)


def import_ripser():
    deps = os.environ.get("REF1_RIPSER_DEPS")
    if deps:
        sys.path.append(deps)
    from ripser import ripser

    return ripser


def load_trace() -> tuple[np.ndarray, float, int]:
    data = np.load(DATA_PATH)
    x = np.asarray(data["xs"], dtype=np.float64)
    fs = float(np.asarray(data["Fs"]).item())
    tau = int(np.asarray(data["tau"]).item()) if "tau" in data.files else 40
    return x, fs, tau


def notebook_window_starts(n_trace: int, n_samples: int) -> np.ndarray:
    max_start = n_trace - (BETTI_WINDOW + 3 * BETTI_TAU)
    if max_start <= 0:
        raise ValueError("Trace is too short for the notebook Betti sweep")
    rng = np.random.default_rng(BETTI_SEED)
    return rng.integers(0, max_start, size=n_samples).astype(int)


def bootstrap_block_len(n_trace: int, n_samples: int) -> int:
    max_start = n_trace - (BETTI_WINDOW + 3 * BETTI_TAU)
    mean_start_spacing = max_start / n_samples
    return max(1, int(np.ceil(BETTI_WINDOW / mean_start_spacing)))


def block_bootstrap_percent_sems(
    labels: np.ndarray,
    starts: np.ndarray,
    categories: list[str],
    n_trace: int,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, float]:
    labels = np.asarray(labels)
    starts = np.asarray(starts, dtype=int)
    order = np.argsort(starts)
    labels = labels[order]
    n = len(labels)
    block_len = bootstrap_block_len(n_trace=n_trace, n_samples=n)
    rng = np.random.default_rng(seed)
    out = {category: [] for category in categories}

    for _ in range(repeats):
        sample_idx: list[int] = []
        while len(sample_idx) < n:
            start = int(rng.integers(0, n))
            sample_idx.extend((np.arange(start, start + block_len) % n).tolist())
        sample = labels[np.asarray(sample_idx[:n], dtype=int)]
        for category in categories:
            out[category].append(100.0 * float(np.mean(sample == category)))

    return {category: float(np.std(values, ddof=1)) for category, values in out.items()}


def load_or_compute_diagrams(recompute: bool) -> list[np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DIAGRAM_CACHE.exists() and not recompute:
        data = np.load(DIAGRAM_CACHE)
        return [np.asarray(data[f"dgm_{dim}"], dtype=np.float64) for dim in range(3)]

    ripser = import_ripser()
    x, fs, tau_from_file = load_trace()
    tau = 40
    pts = lag_embed(x, dim=3, tau=tau)
    idx = np.linspace(0, len(pts) - 1, 500, dtype=int)
    cloud = normalize_cloud(pts[idx])
    result = ripser(cloud, maxdim=2)
    dgms = [np.asarray(dgm, dtype=np.float64) for dgm in result["dgms"][:3]]
    metadata = {
        "source": str(DATA_PATH),
        "fs": fs,
        "tau_from_file": tau_from_file,
        "tau_used_for_barcode": tau,
        "embedding_dim": 3,
        "n_cloud_points": int(len(cloud)),
        "ripser_maxdim": 2,
    }
    np.savez_compressed(
        DIAGRAM_CACHE,
        dgm_0=dgms[0],
        dgm_1=dgms[1],
        dgm_2=dgms[2],
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return dgms


def classify_rows(
    betti: np.ndarray,
    starts: np.ndarray,
    n_trace: int,
) -> tuple[list[dict[str, float | str]], dict[tuple[int, int, int], int]]:
    full = np.all(betti == np.array([1, 2, 1]), axis=1)
    partial = np.all(betti == np.array([1, 1, 1]), axis=1)
    other = ~(full | partial)
    n = len(betti)
    compact_labels = np.where(full, "121", np.where(partial, "111", "Other"))
    compact_sems = block_bootstrap_percent_sems(
        compact_labels,
        starts=starts,
        categories=["121", "111", "Other"],
        n_trace=n_trace,
    )
    rows = [
        {
            "category": "1,2,1",
            "count": int(full.sum()),
            "n": n,
            "percent": 100.0 * float(full.mean()),
            "sem_percent": compact_sems["121"],
        },
        {
            "category": "1,1,1",
            "count": int(partial.sum()),
            "n": n,
            "percent": 100.0 * float(partial.mean()),
            "sem_percent": compact_sems["111"],
        },
        {
            "category": "Others",
            "count": int(other.sum()),
            "n": n,
            "percent": 100.0 * float(other.mean()),
            "sem_percent": compact_sems["Other"],
        },
    ]
    values, counts = np.unique(betti, axis=0, return_counts=True)
    expanded = {tuple(map(int, v)): int(c) for v, c in zip(values, counts)}
    assert sum(int(row["count"]) for row in rows) == n
    return rows, expanded


def write_csv(rows: list[dict[str, float | str]], expanded: dict[tuple[int, int, int], int]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with (PLOTS_DIR / "ref1_real_betti_compact_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "count", "n", "percent", "sem_percent"])
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "percent": round(float(row["percent"]), 1),
                "sem_percent": round(float(row["sem_percent"]), 1),
            }
            for row in rows
        )
    with (PLOTS_DIR / "ref1_real_betti_expanded_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta0", "beta1", "beta2", "count"])
        for betti, count in sorted(expanded.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([*betti, count])


def betti_label(signature: tuple[int, int, int]) -> str:
    return "".join(str(v) for v in signature)


def expand_betti_counts(betti: np.ndarray) -> dict[tuple[int, int, int], int]:
    values, counts = np.unique(np.asarray(betti, dtype=int), axis=0, return_counts=True)
    return {tuple(map(int, value)): int(count) for value, count in zip(values, counts)}


def load_shuffle_betti_sweeps() -> dict[str, dict[str, np.ndarray | int]]:
    out = {}
    for stem, label, _ in SHUFFLES:
        path = SHUFFLE_BETTI_SWEEP_DIR / f"{stem}_betti_sweep.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing shuffle Betti sweep cache: {path}")
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        out[label] = {
            "betti": np.asarray(data["betti_sweep"], dtype=int),
            "starts": np.asarray(data["starts"], dtype=int),
            "n_trace": int(metadata["n_trace_samples"]),
        }
    return out


def write_grouped_betti_csv(
    real_betti: np.ndarray,
    real_starts: np.ndarray,
    real_n_trace: int,
    shuffle_sweeps: dict[str, dict[str, np.ndarray | int]],
) -> tuple[list[str], list[dict[str, float | int | str]]]:
    expanded = expand_betti_counts(real_betti)
    shuffle_expanded = {
        label: expand_betti_counts(np.asarray(sweep["betti"], dtype=int))
        for label, sweep in shuffle_sweeps.items()
    }
    real_total = sum(expanded.values())
    shuffle_totals = {label: sum(counts.values()) for label, counts in shuffle_expanded.items()}
    total_counts: dict[tuple[int, int, int], int] = dict(expanded)
    for counts in shuffle_expanded.values():
        for signature, count in counts.items():
            total_counts[signature] = total_counts.get(signature, 0) + count

    all_signatures = set(total_counts)
    ordered_signatures = sorted(
        all_signatures,
        key=lambda signature: (-total_counts.get(signature, 0), signature),
    )
    signature_labels = [betti_label(signature) for signature in ordered_signatures]
    real_signature_labels = np.asarray([betti_label(tuple(map(int, row))) for row in real_betti])
    real_sems = block_bootstrap_percent_sems(
        real_signature_labels,
        starts=real_starts,
        categories=signature_labels,
        n_trace=real_n_trace,
    )
    shuffle_sems = {}
    for label, sweep in shuffle_sweeps.items():
        shuffle_signature_labels = np.asarray([
            betti_label(tuple(map(int, row)))
            for row in np.asarray(sweep["betti"], dtype=int)
        ])
        shuffle_sems[label] = block_bootstrap_percent_sems(
            shuffle_signature_labels,
            starts=np.asarray(sweep["starts"], dtype=int),
            categories=signature_labels,
            n_trace=int(sweep["n_trace"]),
        )

    rows: list[dict[str, float | int | str]] = []
    for sort_index, signature in enumerate(ordered_signatures):
        signature_label = betti_label(signature)
        real_count = int(expanded.get(signature, 0))
        rows.append(
            {
                "signature_sort_index": sort_index,
                "signature": signature_label,
                "condition": "Real",
                "count": real_count,
                "n": real_total,
                "percent": round(100.0 * real_count / real_total, 1),
                "sem_percent": round(real_sems[signature_label], 1),
                "color": "#8B0000",
            }
        )
        for _, label, color in SHUFFLES:
            count = int(shuffle_expanded[label].get(signature, 0))
            n = int(shuffle_totals[label])
            rows.append(
                {
                    "signature_sort_index": sort_index,
                    "signature": signature_label,
                    "condition": label,
                    "count": count,
                    "n": n,
                    "percent": round(100.0 * count / n, 1),
                    "sem_percent": round(shuffle_sems[label][signature_label], 1),
                    "color": color,
                }
            )

    with (PLOTS_DIR / "ref1_real_vs_shuffle_betti_grouped_bars.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["signature_sort_index", "signature", "condition", "count", "n", "percent", "sem_percent", "color"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return signature_labels, rows


def compact_cells(betti: np.ndarray, starts: np.ndarray, n_trace: int) -> dict[str, str]:
    betti = np.asarray(betti, dtype=int)
    full = np.all(betti == np.array([1, 2, 1]), axis=1)
    partial = np.all(betti == np.array([1, 1, 1]), axis=1)
    compact_labels = np.where(full, "1,2,1", np.where(partial, "1,1,1", "Others"))
    categories = ["1,2,1", "1,1,1", "Others"]
    sems = block_bootstrap_percent_sems(compact_labels, starts=starts, categories=categories, n_trace=n_trace)
    out = {}
    for category in categories:
        percent = 100.0 * float(np.mean(compact_labels == category))
        out[category] = f"{percent:.1f} +/- {sems[category]:.1f}"
    return out


def write_table2_compact_md(
    real_betti: np.ndarray,
    real_starts: np.ndarray,
    real_n_trace: int,
    shuffle_sweeps: dict[str, dict[str, np.ndarray | int]],
) -> None:
    condition_cells = {
        "Real": compact_cells(real_betti, real_starts, real_n_trace),
    }
    for _, label, _ in SHUFFLES:
        sweep = shuffle_sweeps[label]
        condition_cells[label] = compact_cells(
            np.asarray(sweep["betti"], dtype=int),
            np.asarray(sweep["starts"], dtype=int),
            int(sweep["n_trace"]),
        )

    lines = [
        "| Condition | 1,2,1 | 1,1,1 | Others |",
        "|---|---:|---:|---:|",
    ]
    for condition in ["Real"] + [label for _, label, _ in SHUFFLES]:
        cells = condition_cells[condition]
        lines.append(
            f"| {condition} | {cells['1,2,1']} | {cells['1,1,1']} | {cells['Others']} |"
        )
    block_len = bootstrap_block_len(real_n_trace, len(real_betti))
    lines.extend(
        [
            "",
            f"Values are percentages of the 200 sampled windows assigned to each Betti category, reported as percent +/- SEM from {BOOTSTRAP_REPEATS} time-block bootstrap resamples.",
            f"For each bootstrap resample, windows were sorted by start time and resampled in contiguous blocks of {block_len} sampled windows, approximately one 2 s analysis window.",
            "This is used because nearby LFP windows come from the same autocorrelated recording, so adjacent windows should vary together rather than be treated as independent samples.",
        ]
    )
    (UNIT_DIR / "table2_compact.md").write_text("\n".join(lines) + "\n")


def plot_barcode_panel(fig: plt.Figure, grid_cell, dgms: list[np.ndarray]) -> None:
    barcode_grid = grid_cell.subgridspec(3, 1, hspace=0.12)
    axes = [fig.add_subplot(barcode_grid[dim, 0]) for dim in range(3)]
    annotations = ["1 Continuous\ncomponent", "2 loop", "1 cavity"]
    finite_deaths = [dgm[np.isfinite(dgm[:, 1]), 1] for dgm in dgms]
    max_radius = max(float(deaths.max()) for deaths in finite_deaths if len(deaths) > 0) * 1.05

    for dim, ax in enumerate(axes):
        dgm = dgms[dim]
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        finite = np.isfinite(deaths)
        lifetimes = np.where(finite, deaths - births, np.inf)
        top_idx = np.argsort(lifetimes)[::-1][:30]
        top_idx = top_idx[np.argsort(births[top_idx])]
        max_death = deaths[finite].max() if finite.any() else 1.0
        end_cap = max_death * 1.05

        for i, idx in enumerate(top_idx):
            birth, death = births[idx], deaths[idx]
            if np.isfinite(death):
                ax.plot([birth, death], [i, i], color="#8B0000", lw=1.7, solid_capstyle="butt")
            else:
                ax.plot([birth, end_cap], [i, i], color="#8B0000", lw=1.7, solid_capstyle="butt")
                ax.scatter([end_cap], [i], marker="v", color="black", s=18, zorder=5)

        ax.set_ylim(-1, len(top_idx) + 1)
        ax.set_xlim(0.0, max_radius)
        ax.set_yticks(np.arange(0, len(top_idx) + 1, 10))
        ax.set_ylabel(rf"$H_{dim}$", fontsize=10, fontweight="bold", rotation=0, labelpad=14)
        ax.text(0.97, 0.52, annotations[dim], transform=ax.transAxes, ha="right", va="center", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        if dim < 2:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Radius", fontsize=9)


def plot_betti_space_panel(ax: plt.Axes, betti: np.ndarray) -> None:
    bs = np.asarray(betti, dtype=int)
    n_total = len(bs)
    rng_plot = np.random.default_rng(0)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    main_keys = [(1, 1, 1), (1, 2, 1)]
    other_mask = np.ones(len(bs), dtype=bool)
    for sig in main_keys:
        other_mask &= ~np.all(bs == list(sig), axis=1)

    if other_mask.sum() > 0:
        n_other = int(other_mask.sum())
        jittered = bs[other_mask].astype(float) + rng_plot.normal(0, 0.12, (n_other, 3))
        ax.scatter(
            jittered[:, 1],
            jittered[:, 2],
            jittered[:, 0],
            c="#9869698F",
            s=18,
            alpha=0.3,
            edgecolors="white",
            linewidths=0.25,
            label=f"Other: {100 * n_other / n_total:.0f}%",
        )

    mask_111 = np.all(bs == [1, 1, 1], axis=1)
    count_111 = int(mask_111.sum())
    if count_111 > 0:
        jittered = bs[mask_111].astype(float) + rng_plot.normal(0, 0.12, (count_111, 3))
        ax.scatter(
            jittered[:, 1],
            jittered[:, 2],
            jittered[:, 0],
            c="#B860608F",
            s=30,
            alpha=0.5,
            edgecolors="white",
            linewidths=0.25,
            label=f"β=(1,1,1): {100 * count_111 / n_total:.0f}%",
        )

    mask_121 = np.all(bs == [1, 2, 1], axis=1)
    count_121 = int(mask_121.sum())
    if count_121 > 0:
        jittered = bs[mask_121].astype(float) + rng_plot.normal(0, 0.12, (count_121, 3))
        ax.scatter(
            jittered[:, 1],
            jittered[:, 2],
            jittered[:, 0],
            c="#8B0000",
            s=40,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            label=f"β=(1,2,1): {100 * count_121 / n_total:.0f}%",
        )

    ax.set_xlabel(r"$\beta_1$", fontsize=9, labelpad=5)
    ax.set_ylabel(r"$\beta_2$", fontsize=9, labelpad=5)
    ax.set_zlabel(r"$\beta_0$", fontsize=9, labelpad=5)
    ax.set_xticks(range(0, 4))
    ax.set_yticks(range(0, 3))
    ax.set_zticks([0, 1, 2])
    ax.set_xlim(1.5, 3.5)
    ax.set_ylim(1, 2.5)
    ax.set_zlim(0, 1.5)
    ax.view_init(elev=20, azim=225)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8, pad=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8, framealpha=0.8)


def plot_expanded_bar_panel(ax: plt.Axes, expanded: dict[tuple[int, int, int], int]) -> None:
    ordered = sorted(expanded.items(), key=lambda item: (-item[1], item[0]))
    total = sum(expanded.values())
    labels = [betti_label(signature) for signature, _ in ordered]
    counts = [count for _, count in ordered]
    percents = [100.0 * count / total for count in counts]
    colors = [
        "darkred" if label == "121" else "#d9a0a0" if label == "111" else "#cfc3c3"
        for label in labels
    ]

    ax.bar(labels, percents, color=colors, edgecolor="none")
    ax.set_ylabel("Windows (%)", fontsize=9)
    ax.set_xlabel(r"Betti signature $\beta_0\beta_1\beta_2$", fontsize=9)
    ax.set_ylim(0, max(55, max(percents) + 8))
    ax.tick_params(axis="both", labelsize=8)
    for i, (percent, count) in enumerate(zip(percents, counts)):
        ax.text(i, percent + 1.2, f"{percent:.1f}%\n{count}", ha="center", va="bottom", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_real_vs_shuffle_grouped_bar(signature_labels: list[str], rows: list[dict[str, float | int | str]]) -> None:
    conditions = [("Real", "#8B0000")] + [(label, color) for _, label, color in SHUFFLES]
    values = {
        (str(row["signature"]), str(row["condition"])): float(row["percent"])
        for row in rows
    }
    sems = {
        (str(row["signature"]), str(row["condition"])): float(row["sem_percent"])
        for row in rows
    }
    counts = {
        (str(row["signature"]), str(row["condition"])): int(row["count"])
        for row in rows
    }

    x = np.arange(len(signature_labels), dtype=float)
    width = min(0.12, 0.78 / len(conditions))
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    for offset, (condition, color) in zip(offsets, conditions):
        y = [values[(signature, condition)] for signature in signature_labels]
        yerr = [sems[(signature, condition)] for signature in signature_labels]
        bars = ax.bar(
            x + offset,
            y,
            yerr=yerr,
            capsize=2.0,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "0.25"},
            width=width,
            color=color,
            edgecolor="none",
            label=condition,
        )
        for signature, bar, percent, err in zip(signature_labels, bars, y, yerr):
            count = counts[(signature, condition)]
            if count == 0:
                continue
            label = f"{percent:.1f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                percent + err + 1.6,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(signature_labels)
    ax.set_xlabel(r"Betti numbers $(\beta_0,\beta_1,\beta_2)$")
    ax.set_ylabel("Windows (%)")
    ax.set_ylim(0, max(55, max(values[key] + sems[key] for key in values) + 10))
    ax.set_title("Betti numbers in real data and shuffles", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ref1_real_vs_shuffle_betti_grouped_bar.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def plot_ref1_1x3(dgms: list[np.ndarray], betti: np.ndarray, expanded: dict[tuple[int, int, int], int]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.2, 4.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.35, 1.0], wspace=0.22)

    plot_barcode_panel(fig, grid[0, 0], dgms)
    ax_scatter = fig.add_subplot(grid[0, 1], projection="3d")
    plot_betti_space_panel(ax_scatter, betti)
    ax_bar = fig.add_subplot(grid[0, 2])
    plot_expanded_bar_panel(ax_bar, expanded)

    fig.suptitle("Topology Assessment", fontweight="bold", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.17, top=0.84)
    fig.savefig(PLOTS_DIR / "ref1_1x3_regenerated_components.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Ref1 topology-assessment source components.")
    parser.add_argument("--recompute-diagram", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    betti = np.asarray(np.load(SOURCE_BETTI), dtype=int)
    real_trace, _, _ = load_trace()
    real_starts = notebook_window_starts(len(real_trace), len(betti))
    np.save(CACHE_DIR / "betti_sweep_real.npy", betti)
    rows, expanded = classify_rows(betti, starts=real_starts, n_trace=len(real_trace))
    write_csv(rows, expanded)
    dgms = load_or_compute_diagrams(recompute=args.recompute_diagram)
    plot_ref1_1x3(dgms, betti, expanded)
    shuffle_sweeps = load_shuffle_betti_sweeps()
    signature_labels, grouped_rows = write_grouped_betti_csv(betti, real_starts, len(real_trace), shuffle_sweeps)
    write_table2_compact_md(betti, real_starts, len(real_trace), shuffle_sweeps)
    plot_real_vs_shuffle_grouped_bar(signature_labels, grouped_rows)


if __name__ == "__main__":
    main()
