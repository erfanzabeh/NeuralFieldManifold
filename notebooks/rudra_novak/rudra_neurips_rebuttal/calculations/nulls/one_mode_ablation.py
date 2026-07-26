from __future__ import annotations

import numpy as np
from scipy import signal

from plot_null_diagnostics import plot_one
from shuffle_utils import REBUTTAL_DIR, match_mean_std, notebook_style_psd, parse_common_args, run_or_load


STEM = "one_mode_ablation"
MODE_DETECTION_PATH = REBUTTAL_DIR / "data" / "monkey_chan1001_notched_raw_20s.npz"


def load_mode_detection_trace(default_trace: np.ndarray, default_fs: float) -> tuple[np.ndarray, float, str]:
    if not MODE_DETECTION_PATH.exists():
        return default_trace, default_fs, "topology_ready_trace"
    data = np.load(MODE_DETECTION_PATH)
    x = np.asarray(data["xs"], dtype=np.float64)
    fs = float(np.asarray(data["Fs"]).item())
    return x, fs, str(MODE_DETECTION_PATH)


def detect_modes(real: np.ndarray, fs: float, fmin: float, fmax: float, min_distance_hz: float) -> list[float]:
    freqs, psd = notebook_style_psd(real, fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_m = freqs[mask]
    log_psd_m = np.log10(psd[mask] + 1e-24)
    distance = max(1, int(round(min_distance_hz / np.mean(np.diff(freqs_m)))))
    peaks, _ = signal.find_peaks(log_psd_m, distance=distance, prominence=0.03)
    if len(peaks) == 0:
        return [float(freqs_m[np.argmax(log_psd_m)])]
    prominences = signal.peak_prominences(log_psd_m, peaks)[0]
    ranked = peaks[np.argsort(prominences)[::-1]]
    return [float(freqs_m[i]) for i in ranked[:5]]


def smooth_frequency_band_fft(real: np.ndarray, fs: float, band_low_hz: float, band_high_hz: float, shoulder_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(real)
    freqs = np.fft.rfftfreq(len(real), d=1.0 / fs)
    amplitudes = np.abs(spectrum)
    edited = amplitudes.copy()

    inner = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    left = (freqs >= band_low_hz - shoulder_hz) & (freqs < band_low_hz)
    right = (freqs > band_high_hz) & (freqs <= band_high_hz + shoulder_hz)
    if np.count_nonzero(inner) == 0 or np.count_nonzero(left) == 0 or np.count_nonzero(right) == 0:
        return np.array(real, copy=True)

    anchor_freqs = np.array([np.median(freqs[left]), np.median(freqs[right])])
    anchor_log_amp = np.array([
        np.median(np.log(amplitudes[left] + 1e-24)),
        np.median(np.log(amplitudes[right] + 1e-24)),
    ])
    replacement = np.exp(np.interp(freqs[inner], anchor_freqs, anchor_log_amp))

    n_inner = int(np.count_nonzero(inner))
    weights = np.ones(n_inner)
    df = float(np.mean(np.diff(freqs)))
    edge_bins = min(max(1, int(round(0.25 / df))), max(1, n_inner // 3))
    weights[:edge_bins] = np.linspace(0.0, 1.0, edge_bins)
    weights[-edge_bins:] = np.linspace(1.0, 0.0, edge_bins)
    edited[inner] = (1.0 - weights) * amplitudes[inner] + weights * replacement

    phase = np.exp(1j * np.angle(spectrum))
    edited_spectrum = edited * phase
    edited_spectrum[0] = spectrum[0]
    if len(real) % 2 == 0:
        edited_spectrum[-1] = np.real(edited_spectrum[-1]) + 0j
    return np.fft.irfft(edited_spectrum, n=len(real))


def main() -> None:
    parser = parse_common_args("Generate a one-mode-ablated monkey LFP shuffle.")
    parser.add_argument("--band-low-hz", type=float, default=10.0)
    parser.add_argument("--band-high-hz", type=float, default=32.0)
    parser.add_argument("--shoulder-hz", type=float, default=4.0)
    parser.add_argument("--min-distance-hz", type=float, default=1.5)
    args = parser.parse_args()

    def compute(real, time, fs):
        detection_trace, detection_fs, detection_source = load_mode_detection_trace(real, fs)
        modes = detect_modes(
            detection_trace,
            detection_fs,
            fmin=args.band_low_hz,
            fmax=args.band_high_hz,
            min_distance_hz=args.min_distance_hz,
        )
        y = smooth_frequency_band_fft(
            real,
            fs=fs,
            band_low_hz=args.band_low_hz,
            band_high_hz=args.band_high_hz,
            shoulder_hz=args.shoulder_hz,
        )
        y = match_mean_std(y, real)
        return y, {
            "condition": STEM,
            "seed": args.seed,
            "detected_peaks_in_band_hz": modes,
            "ablation_band_hz": [args.band_low_hz, args.band_high_hz],
            "shoulder_hz": args.shoulder_hz,
            "detection_source": detection_source,
            "meaning": "smooth the marked mid-frequency oscillatory mode into the local spectral background while preserving phase elsewhere",
        }

    run_or_load(STEM, args.recompute, compute)
    plot_one(STEM)


if __name__ == "__main__":
    main()
