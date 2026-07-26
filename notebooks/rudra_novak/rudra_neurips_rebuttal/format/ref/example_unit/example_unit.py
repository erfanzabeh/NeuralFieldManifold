from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"
CACHE_PATH = CACHE_DIR / "example_unit.pkl"
PLOT_PATH = PLOTS_DIR / "example_unit.pdf"


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.figsize"] = [3.0, 3.0]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False


def compute_example() -> dict[str, np.ndarray | dict[str, float]]:
    x = np.linspace(0.0, 1.0, 40)
    condition_a = np.sin(2 * np.pi * x)
    condition_b = np.sin(2 * np.pi * x + 0.6) + 0.2
    return {
        "x": x,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "summary": {
            "condition_a_mean": float(condition_a.mean()),
            "condition_b_mean": float(condition_b.mean()),
        },
    }


def load_or_compute(recompute: bool) -> dict[str, np.ndarray | dict[str, float]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if recompute and CACHE_PATH.exists():
        CACHE_PATH.unlink()
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as handle:
            return pickle.load(handle)
    result = compute_example()
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(result, handle)
    return result


def plot_example(result: dict[str, np.ndarray | dict[str, float]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot(result["x"], result["condition_a"], color="darkred", label="Condition A")
    ax.plot(result["x"], result["condition_b"], color="midnightblue", label="Condition B")
    ax.set_xlabel("Normalized sample")
    ax.set_ylabel("Synthetic measurement")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ymin = min(float(np.min(result["condition_a"])), float(np.min(result["condition_b"])))
    ymax = max(float(np.max(result["condition_a"])), float(np.max(result["condition_b"])))
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_yticks([round(ymin, 1), round(ymax, 1)])
    ax.legend(loc="best", fontsize=8, frameon=False)
    sns.despine(ax=ax, trim=True, offset=10)
    fig.savefig(PLOT_PATH, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the example compact unit.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Delete the existing cache and recompute this unit before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = load_or_compute(recompute=args.recompute)
    plot_example(result)


if __name__ == "__main__":
    main()
