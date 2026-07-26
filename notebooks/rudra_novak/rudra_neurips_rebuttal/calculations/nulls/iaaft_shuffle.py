from __future__ import annotations

import numpy as np

from plot_null_diagnostics import plot_one
from shuffle_utils import parse_common_args, run_or_load


STEM = "iaaft_shuffle"


def iaaft(real: np.ndarray, seed: int, iterations: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target_sorted = np.sort(real)
    target_amplitude = np.abs(np.fft.rfft(real))
    y = rng.permutation(real)

    for _ in range(iterations):
        spectrum = np.fft.rfft(y)
        phases = np.angle(spectrum)
        y = np.fft.irfft(target_amplitude * np.exp(1j * phases), n=len(real))
        order = np.argsort(y)
        remapped = np.empty_like(y)
        remapped[order] = target_sorted
        y = remapped
    return y


def main() -> None:
    parser = parse_common_args("Generate an IAAFT-shuffled monkey LFP trace.")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    def compute(real, time, fs):
        y = iaaft(real, seed=args.seed, iterations=args.iterations)
        return y, {
            "condition": STEM,
            "seed": args.seed,
            "iterations": args.iterations,
            "meaning": "match amplitude distribution and Fourier amplitudes while randomizing phase structure",
        }

    run_or_load(STEM, args.recompute, compute)
    plot_one(STEM)


if __name__ == "__main__":
    main()
