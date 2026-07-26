from __future__ import annotations

import numpy as np

from plot_null_diagnostics import plot_one
from shuffle_utils import match_mean_std, parse_common_args, run_or_load


STEM = "aperiodic_1f"


def generate_aperiodic_1f(n: int, fs: float, alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amplitudes = np.ones_like(freqs)
    nonzero = freqs > 0
    amplitudes[nonzero] = 1.0 / np.power(freqs[nonzero], alpha / 2.0)
    amplitudes[0] = 0.0
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0.0
    if n % 2 == 0:
        spectrum[-1] = np.real(spectrum[-1]) + 0j
    return np.fft.irfft(spectrum, n=n)


def main() -> None:
    parser = parse_common_args("Generate an aperiodic-only 1/f monkey-length shuffle.")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    def compute(real, time, fs):
        y = generate_aperiodic_1f(len(real), fs=fs, alpha=args.alpha, seed=args.seed)
        y = match_mean_std(y, real)
        return y, {
            "condition": STEM,
            "seed": args.seed,
            "alpha": args.alpha,
            "meaning": "broadband autocorrelated 1/f-like signal without narrow oscillatory peaks",
        }

    run_or_load(STEM, args.recompute, compute)
    plot_one(STEM)


if __name__ == "__main__":
    main()
