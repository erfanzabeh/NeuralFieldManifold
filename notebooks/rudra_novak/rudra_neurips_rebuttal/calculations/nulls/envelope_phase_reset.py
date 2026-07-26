from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from plot_null_diagnostics import plot_one
from shuffle_utils import match_mean_std, parse_common_args, rfft_phase_shuffle, run_or_load


STEM = "envelope_phase_reset"


def main() -> None:
    parser = parse_common_args("Generate an envelope-preserving phase-reset monkey LFP shuffle.")
    parser.add_argument("--envelope-smooth-sec", type=float, default=1.0)
    args = parser.parse_args()

    def compute(real, time, fs):
        centered = real - np.mean(real)
        envelope = np.abs(signal.hilbert(centered))
        sigma_samples = max(1.0, args.envelope_smooth_sec * fs)
        slow_envelope = gaussian_filter1d(envelope, sigma=sigma_samples)
        slow_envelope = slow_envelope / (np.mean(slow_envelope) + 1e-12)

        carrier = rfft_phase_shuffle(centered, seed=args.seed)
        carrier = carrier / (np.std(carrier) + 1e-12)
        y = carrier * slow_envelope
        y = match_mean_std(y, real)
        return y, {
            "condition": STEM,
            "seed": args.seed,
            "envelope_smooth_sec": args.envelope_smooth_sec,
            "meaning": "preserve slow amplitude envelope while resetting fast phase continuity",
        }

    run_or_load(STEM, args.recompute, compute)
    plot_one(STEM)


if __name__ == "__main__":
    main()
