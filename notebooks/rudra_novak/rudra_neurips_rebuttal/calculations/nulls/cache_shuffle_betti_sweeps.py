from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


UNIT_DIR = Path(__file__).resolve().parent
SHUFFLE_CACHE_DIR = UNIT_DIR / "cache"
SWEEP_CACHE_DIR = SHUFFLE_CACHE_DIR / "betti_sweeps"

EXPECTED_STEMS = [
    "fourier_phase_shuffle",
    "iaaft_shuffle",
    "aperiodic_1f",
    "envelope_phase_reset",
    "one_mode_ablation",
]

_RIPSER = None
_TRACE = None


def import_ripser():
    try:
        from ripser import ripser

        return ripser
    except ModuleNotFoundError:
        pass

    candidates = [
        os.environ.get("RIPSER_DEPS"),
        os.environ.get("REF1_RIPSER_DEPS"),
        "/tmp/ref1_ripser_deps",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        sys.path.append(str(path))
        try:
            from ripser import ripser

            return ripser
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError("ripser is not importable; set RIPSER_DEPS or install ripser.")


def init_worker(trace: np.ndarray) -> None:
    global _RIPSER, _TRACE
    _RIPSER = import_ripser()
    _TRACE = trace


def lag_embed(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("Trace is too short for requested lag embedding")
    return np.column_stack([x[i * tau:i * tau + n] for i in range(dim)])


def normalize_cloud(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / (norms + 1e-12)


def compute_betti_robust(dgms: list[np.ndarray], threshold: float = 0.30) -> np.ndarray:
    betti = []
    for dim in range(min(len(dgms), 3)):
        dgm = dgms[dim]
        finite = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite, 1] - dgm[finite, 0]
        if dim == 0:
            betti.append(1)
        else:
            betti.append(int(np.sum(lifetimes > threshold)))
    while len(betti) < 3:
        betti.append(0)
    return np.asarray(betti[:3], dtype=int)


def process_window(task: tuple[int, np.ndarray, int, int, int, float]) -> tuple[np.ndarray, list[np.ndarray]]:
    start, subsample_idx, window, tau, embedding_dim, threshold = task
    if _RIPSER is None or _TRACE is None:
        raise RuntimeError("worker was not initialized")
    segment = _TRACE[start:start + window + 3 * tau]
    points = lag_embed(segment, embedding_dim, tau)[:window]
    cloud = normalize_cloud(points[subsample_idx])
    result = _RIPSER(cloud, maxdim=2)
    dgms = [np.asarray(dgm, dtype=np.float64) for dgm in result["dgms"][:3]]
    betti = compute_betti_robust(dgms, threshold=threshold)
    return betti, dgms


def load_shuffle(stem: str) -> tuple[np.ndarray, float, dict[str, Any], Path]:
    path = SHUFFLE_CACHE_DIR / f"{stem}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing shuffle cache: {path}")
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    return np.asarray(data["xs"], dtype=np.float64), float(np.asarray(data["Fs"]).item()), metadata, path


def make_sampling_plan(n_trace: int, window: int, tau: int, n_samples: int, n_subsample: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    max_start = n_trace - (window + 3 * tau)
    if max_start <= 0:
        raise ValueError("Trace is too short for the notebook windowed Betti sweep")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=n_samples)
    subsample_indices = []
    for _ in starts:
        idx = rng.choice(window, size=min(n_subsample, window), replace=False)
        idx.sort()
        subsample_indices.append(idx)
    return starts.astype(int), np.asarray(subsample_indices, dtype=int)


def cache_one(
    stem: str,
    recompute: bool,
    window: int,
    tau: int,
    embedding_dim: int,
    n_samples: int,
    n_subsample: int,
    seed: int,
    threshold: float,
    workers: int,
) -> Path:
    out_path = SWEEP_CACHE_DIR / f"{stem}_betti_sweep.npz"
    if out_path.exists() and not recompute:
        print(f"cached {stem}: {out_path}", flush=True)
        return out_path

    trace, fs, shuffle_metadata, source_path = load_shuffle(stem)
    starts, subsample_indices = make_sampling_plan(
        len(trace),
        window=window,
        tau=tau,
        n_samples=n_samples,
        n_subsample=n_subsample,
        seed=seed,
    )
    tasks = [
        (int(start), subsample_indices[i], window, tau, embedding_dim, threshold)
        for i, start in enumerate(starts)
    ]

    print(f"computing {stem}: {n_samples} windows, {n_subsample} points/window", flush=True)
    if workers <= 1:
        init_worker(trace)
        results = [process_window(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(trace,)) as pool:
            results = list(pool.map(process_window, tasks))

    betti_sweep = np.asarray([betti for betti, _ in results], dtype=int)
    metadata = {
        "condition": stem,
        "source_shuffle_cache": str(source_path),
        "source_shuffle_metadata": shuffle_metadata,
        "fs": fs,
        "n_trace_samples": int(len(trace)),
        "embedding_dim": int(embedding_dim),
        "tau": int(tau),
        "window": int(window),
        "n_samples": int(n_samples),
        "n_subsample": int(n_subsample),
        "seed": int(seed),
        "ripser_maxdim": 2,
        "betti_threshold": float(threshold),
        "method": "same windowed ripser Betti sweep as monkey_torus_fit copy.ipynb cell 46",
    }

    payload: dict[str, Any] = {
        "betti_sweep": betti_sweep,
        "starts": starts,
        "subsample_indices": subsample_indices,
        "metadata": json.dumps(metadata, sort_keys=True),
    }
    for window_idx, (_, dgms) in enumerate(results):
        for dim, dgm in enumerate(dgms):
            payload[f"window_{window_idx:03d}_dgm_{dim}"] = dgm

    SWEEP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache windowed ripser Betti sweeps for the five monkey LFP shuffles.")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--tau", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-subsample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--betti-threshold", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for stem in EXPECTED_STEMS:
        cache_one(
            stem,
            recompute=args.recompute,
            window=args.window,
            tau=args.tau,
            embedding_dim=args.embedding_dim,
            n_samples=args.n_samples,
            n_subsample=args.n_subsample,
            seed=args.seed,
            threshold=args.betti_threshold,
            workers=max(1, args.workers),
        )


if __name__ == "__main__":
    main()
