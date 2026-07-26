from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

from NeuralFieldManifold.embedders import embed
from NeuralFieldManifold.fits.two_torus import two_torus_fit


UNIT_DIR = Path(__file__).resolve().parent
REBUTTAL_DIR = UNIT_DIR.parent.parent
DATA_PATH = REBUTTAL_DIR / "data" / "monkey_15_min.npz"
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"


def envelope_normalize_minmax(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * 2.0 - 1.0


def generate_bad_ar_data(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p1, p2 = 0.6, -0.4
    coeffs = np.array([p1 + p2, -(p1 * p2)])
    x = np.zeros(n_samples, dtype=np.float64)
    x[:2] = rng.standard_normal(2)
    for i in range(2, n_samples):
        x[i] = coeffs[0] * x[i - 1] + coeffs[1] * x[i - 2] + rng.standard_normal()
    return x


def make_windows(xs: np.ndarray, fs: float, starts_sec: np.ndarray, window_sec: float, tau: int, dim: int) -> list[dict]:
    window_samples = int(window_sec * fs)
    windows = []
    for t in starts_sec:
        start = int(t * fs)
        segment = xs[start:start + window_samples]
        windows.append({"points": np.asarray(embed(segment, dim, tau)), "start_sec": float(t), "end_sec": float(t + window_sec)})
    return windows


def fit_one(window: dict, lam: float) -> dict:
    fit = two_torus_fit(window["points"], lam=lam)
    return {**window, **fit}


def run_fits(windows: list[dict], lam: float, n_jobs: int) -> list[dict]:
    return Parallel(n_jobs=n_jobs)(delayed(fit_one)(w, lam=lam) for w in windows)


def metric_array(fits: list[dict], key: str) -> np.ndarray:
    return np.array([float(f[key]) for f in fits], dtype=np.float64)


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def extract_group_stats(results_dict: dict[float, list[dict]], key: str, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([np.mean([f[key] for f in results_dict[x]]) for x in xs], dtype=np.float64)
    sems = np.array([np.std([f[key] for f in results_dict[x]]) / np.sqrt(len(results_dict[x])) for x in xs], dtype=np.float64)
    return means, sems


def group_by_key(fits: list[dict], key: str, values: np.ndarray) -> dict[float, list[dict]]:
    grouped = {v: [] for v in values}
    for fit in fits:
        grouped[fit[key]].append(fit)
    return grouped


def plot_ref3_panel_d(payload: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("frac_inside", "Torus Score", (0.58, 0.90), (0.59, 0.86), (0.30, 0.88)),
        ("mean_error", "Torus Error", (0.008, 0.036), (0.005, 0.035), (0.010, 0.036)),
        ("r_squared", r"$R^2$", (0.955, 1.0), (0.955, 1.0), (0.95, 1.0)),
    ]
    color_alt = "darkred"
    null_color = "#9a9a9a"

    fig, axes = plt.subplots(3, 3, figsize=(9.4, 7.2))

    for row, (key, ylabel, hist_xlim, tau_ylim, win_ylim) in enumerate(metrics):
        real = metric_array(payload["real_fits"], key)
        null = metric_array(payload["null_fits"], key)
        weights_real = np.ones_like(real) / len(real)
        weights_null = np.ones_like(null) / len(null)
        ax = axes[row, 0]
        bins = np.linspace(hist_xlim[0], hist_xlim[1], 31)
        ax.hist(null, bins=bins, weights=weights_null, color=null_color, alpha=0.55, label="Null")
        ax.hist(real, bins=bins, weights=weights_real, color=color_alt, alpha=0.82, label="Data")
        y_top = ax.get_ylim()[1]
        ax.plot([real.mean(), real.mean()], [0, y_top], color="black", lw=1.4, label="Mean")
        _, p = mannwhitneyu(real, null, alternative="two-sided")
        bar_y = y_top * 1.15
        tick_h = y_top * 0.035
        ax.plot([null.mean(), null.mean()], [bar_y - tick_h, bar_y + tick_h], color="black", lw=0.8, clip_on=False)
        ax.plot([real.mean(), real.mean()], [bar_y - tick_h, bar_y + tick_h], color="black", lw=0.8, clip_on=False)
        ax.plot([null.mean(), real.mean()], [bar_y, bar_y], color="black", lw=0.8, clip_on=False)
        ax.text((null.mean() + real.mean()) / 2, bar_y + tick_h, p_to_stars(float(p)), ha="center", va="bottom", fontsize=8, clip_on=False)
        ax.set_xlim(hist_xlim)
        ax.set_ylim(0, y_top * 1.27)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Probability")
        if row == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")

        taus = payload["taus"]
        ax = axes[row, 1]
        d_mean, d_sem = extract_group_stats(payload["tau_results_data"], key, taus)
        n_mean, n_sem = extract_group_stats(payload["tau_results_null"], key, taus)
        ax.fill_between(taus, n_mean - n_sem, n_mean + n_sem, color=null_color, alpha=0.3)
        ax.plot(taus, n_mean, color=null_color, lw=1.1, label="Null")
        ax.fill_between(taus, d_mean - d_sem, d_mean + d_sem, color=color_alt, alpha=0.24)
        ax.plot(taus, d_mean, color=color_alt, lw=1.1, label="Data")
        ax.set_ylim(tau_ylim)
        ax.set_xlim(float(taus.min()), float(taus.max()))
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(ylabel)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

        win_sizes = payload["win_sizes"]
        ax = axes[row, 2]
        d_mean, d_sem = extract_group_stats(payload["win_results_data"], key, win_sizes)
        n_mean, n_sem = extract_group_stats(payload["win_results_null"], key, win_sizes)
        ax.fill_between(win_sizes, n_mean - n_sem, n_mean + n_sem, color=null_color, alpha=0.3)
        ax.plot(win_sizes, n_mean, color=null_color, lw=1.1, label="Null")
        ax.fill_between(win_sizes, d_mean - d_sem, d_mean + d_sem, color=color_alt, alpha=0.24)
        ax.plot(win_sizes, d_mean, color=color_alt, lw=1.1, label="Data")
        ax.set_ylim(win_ylim)
        ax.set_xlim(float(win_sizes.min()), float(win_sizes.max()))
        ax.set_xlabel("Window size (s)")
        ax.set_ylabel(ylabel)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes.ravel():
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle("Population summary of Monkey LFP", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ref3_panel_d_3x3.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the original ref3 AR(2) torus-metric null on a controlled subset.")
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--tau", type=int, default=30)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tau-max", type=int, default=100)
    parser.add_argument("--n-tau-samples", type=int, default=10)
    parser.add_argument("--window-step", type=float, default=0.1)
    parser.add_argument("--n-window-samples", type=int, default=10)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"ref3_ar2_metric_summary_n{args.n_windows}.pkl"
    if cache_path.exists() and not args.recompute:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        data = np.load(DATA_PATH)
        xs = np.asarray(data["xs"], dtype=np.float64)
        fs = float(np.asarray(data["Fs"]).item())
        real = envelope_normalize_minmax(xs)
        null = envelope_normalize_minmax(generate_bad_ar_data(len(xs), seed=args.seed))
        max_start = len(real) / fs - args.window_sec
        rng = np.random.default_rng(args.seed)
        starts = np.sort(rng.uniform(0.0, max_start, args.n_windows))
        real_windows = make_windows(real, fs, starts, args.window_sec, args.tau, args.dim)
        null_windows = make_windows(null, fs, starts, args.window_sec, args.tau, args.dim)
        payload = {
            "args": vars(args),
            "fs": fs,
            "real": real,
            "null": null,
            "starts": starts,
            "real_fits": run_fits(real_windows, args.lam, args.n_jobs),
            "null_fits": run_fits(null_windows, args.lam, args.n_jobs),
        }
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle)
    if not {"fs", "real", "null"}.issubset(payload):
        data = np.load(DATA_PATH)
        xs = np.asarray(data["xs"], dtype=np.float64)
        fs = float(np.asarray(data["Fs"]).item())
        payload["fs"] = fs
        payload["real"] = envelope_normalize_minmax(xs)
        payload["null"] = envelope_normalize_minmax(generate_bad_ar_data(len(xs), seed=args.seed))
    if not {"tau_results_data", "tau_results_null", "win_results_data", "win_results_null"}.issubset(payload):
        rng = np.random.default_rng(args.seed)
        fs = float(payload["fs"])
        real = payload["real"]
        null = payload["null"]

        taus = np.arange(1, args.tau_max + 1)
        all_data_windows = []
        all_null_windows = []
        max_start_tau = len(real) / fs - args.window_sec
        for tau in taus:
            starts = np.sort(rng.uniform(0.0, max_start_tau, args.n_tau_samples))
            data_windows = make_windows(real, fs, starts, args.window_sec, int(tau), args.dim)
            null_windows = make_windows(null, fs, starts, args.window_sec, int(tau), args.dim)
            for window in data_windows:
                window["tau"] = tau
            for window in null_windows:
                window["tau"] = tau
            all_data_windows.extend(data_windows)
            all_null_windows.extend(null_windows)
        all_data_fits = run_fits(all_data_windows, args.lam, args.n_jobs)
        all_null_fits = run_fits(all_null_windows, args.lam, args.n_jobs)
        payload["taus"] = taus
        payload["tau_results_data"] = group_by_key(all_data_fits, "tau", taus)
        payload["tau_results_null"] = group_by_key(all_null_fits, "tau", taus)

        min_samples_needed = (args.dim - 1) * args.tau + 1
        min_window_sec = min_samples_needed / fs
        win_sizes = np.arange(np.ceil(min_window_sec * 10.0) / 10.0, 10.0 + args.window_step / 2.0, args.window_step)
        all_data_windows = []
        all_null_windows = []
        for win_size in win_sizes:
            max_start_ws = len(real) / fs - win_size
            starts = np.sort(rng.uniform(0.0, max_start_ws, args.n_window_samples))
            data_windows = make_windows(real, fs, starts, float(win_size), args.tau, args.dim)
            null_windows = make_windows(null, fs, starts, float(win_size), args.tau, args.dim)
            for window in data_windows:
                window["win_size"] = win_size
            for window in null_windows:
                window["win_size"] = win_size
            all_data_windows.extend(data_windows)
            all_null_windows.extend(null_windows)
        all_data_fits = run_fits(all_data_windows, args.lam, args.n_jobs)
        all_null_fits = run_fits(all_null_windows, args.lam, args.n_jobs)
        payload["win_sizes"] = win_sizes
        payload["win_results_data"] = group_by_key(all_data_fits, "win_size", win_sizes)
        payload["win_results_null"] = group_by_key(all_null_fits, "win_size", win_sizes)

        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle)
    plot_ref3_panel_d(payload)


if __name__ == "__main__":
    main()
