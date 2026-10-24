"""Welch power spectral density estimation."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from ._validation import sampling_rate, signal_array


@dataclass(frozen=True)
class PowerSpectrum:
    """Frequency coordinates in Hz and their power spectral density."""

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
    """Estimate power spectral density with Welch's method.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Input signal.
    sampling_rate_hz : float
        Sampling rate in Hz.
    n_fft : int
        FFT length and maximum segment length. Short signals use their full
        length as the segment and are zero-padded to this FFT length.
    overlap_samples : int
        Segment overlap, capped at one less than the segment length.
    window : str or tuple
        Window specification accepted by scipy.signal.welch.
    frequency_range_hz : tuple of float or None
        Inclusive lower and upper frequencies to retain.
    smooth_bins : float or None
        Standard deviation of optional Gaussian smoothing, in frequency bins.

    Returns
    -------
    PowerSpectrum
        Frequencies in Hz and density in signal-units squared per Hz.

    Notes
    -----
    Each segment is detrended by subtracting its mean.
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
