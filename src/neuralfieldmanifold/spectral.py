"""Spectral summaries with explicit estimation settings."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from ._validation import sampling_rate, signal_array


@dataclass(frozen=True)
class PowerSpectrum:
    frequencies_hz: np.ndarray
    density: np.ndarray


def welch(
    x,
    *,
    sampling_rate_hz,
    n_fft=4096,
    overlap_samples=2048,
    window="hamming",
    frequency_range_hz=None,
    smooth_bins=None,
):
    """Welch PSD in signal-units squared/Hz; optional display smoothing.

    A Hamming window and constant segment detrending match the tutorials'
    MNE Welch calls. Short signals use their full length as the segment.
    """
    x = signal_array(x)
    fs = sampling_rate(sampling_rate_hz)
    if n_fft < 1 or not 0 <= overlap_samples < n_fft:
        raise ValueError("Require n_fft > overlap_samples >= 0")
    nperseg = min(len(x), n_fft)
    frequencies, density = signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        nfft=n_fft,
        noverlap=min(overlap_samples, nperseg - 1),
        detrend="constant",
    )
    if frequency_range_hz is not None:
        low, high = frequency_range_hz
        if not 0 <= low < high <= fs / 2:
            raise ValueError("Invalid frequency range")
        keep = (frequencies >= low) & (frequencies <= high)
        frequencies, density = frequencies[keep], density[keep]
    if smooth_bins is not None:
        if smooth_bins <= 0:
            raise ValueError("smooth_bins must be positive")
        density = gaussian_filter1d(density, smooth_bins)
    return PowerSpectrum(frequencies, density)
