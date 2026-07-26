from __future__ import annotations

from plot_null_diagnostics import plot_one
from shuffle_utils import match_mean_std, parse_common_args, rfft_phase_shuffle, run_or_load


STEM = "fourier_phase_shuffle"


def main() -> None:
    parser = parse_common_args("Generate a Fourier phase-shuffled monkey LFP trace.")
    args = parser.parse_args()

    def compute(real, time, fs):
        y = rfft_phase_shuffle(real, seed=args.seed)
        y = match_mean_std(y, real)
        return y, {"condition": STEM, "seed": args.seed, "meaning": "preserve Fourier amplitudes while randomizing phases"}

    run_or_load(STEM, args.recompute, compute)
    plot_one(STEM)


if __name__ == "__main__":
    main()
