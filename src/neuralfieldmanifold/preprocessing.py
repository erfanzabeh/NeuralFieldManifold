"""Filtering, resampling, and envelope normalization for one-dimensional signals."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ._validation import sampling_rate, signal_array


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for notch, bandpass, decimation, and envelope processing.

    Parameters
    ----------
    notch_hz : float or None
        Line frequency to remove, in Hz. None disables notch filtering.
    notch_quality : float
        Notch quality factor (center frequency divided by bandwidth).
    band_hz : tuple of float or None
        Lower and upper bandpass cutoffs in Hz.
    filter_order : int
        Butterworth order used by the bandpass filter.
    output_rate_hz : float or None
        Output sampling rate. Must divide the input rate by an integer.
    envelope_band_hz : tuple of float or None
        Band in Hz used for envelope normalization, after decimation.
        None disables normalization.
    envelope_lowpass_hz : float or None
        Lowpass cutoff for the Hilbert envelope. None leaves it unsmoothed.
    envelope_epsilon : float
        Positive constant added to the envelope before division.

    Notes
    -----
    Steps run in the order listed above. All filter cutoffs must lie below
    Nyquist at the stage where they are applied.
    """

    notch_hz: float | None = None
    notch_quality: float = 30.0
    band_hz: tuple[float, float] | None = None
    filter_order: int = 4
    output_rate_hz: float | None = None
    envelope_band_hz: tuple[float, float] | None = None
    envelope_lowpass_hz: float | None = 3.0
    envelope_epsilon: float = 1e-6


@dataclass(frozen=True)
class ProcessedSignal:
    """Processing results and output sampling rate.

    Attributes
    ----------
    signal : ndarray
        Final processed signal.
    bandpassed : ndarray
        Signal after filtering and decimation, before envelope processing.
    envelope : ndarray or None
        Hilbert amplitude envelope, or None if normalization is disabled.
    sampling_rate_hz : float
        Output sampling rate in Hz.
    """

    signal: np.ndarray
    bandpassed: np.ndarray
    envelope: np.ndarray | None
    sampling_rate_hz: float

    @property
    def duration_seconds(self):
        return len(self.signal) / self.sampling_rate_hz


def _band(band_hz, fs):
    band = np.asarray(band_hz, dtype=float)
    if (
        band.shape != (2,)
        or not np.isfinite(band).all()
        or not 0 < band[0] < band[1] < fs / 2
    ):
        raise ValueError("Band must satisfy 0 < low < high < sampling_rate_hz / 2")
    return band / (fs / 2)


def bandpass(x, *, sampling_rate_hz, band_hz, order=4):
    """Zero-phase Butterworth SOS bandpass; no normalization."""
    x = signal_array(x)
    fs = sampling_rate(sampling_rate_hz)
    sos = signal.butter(order, _band(band_hz, fs), btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def notch(x, *, sampling_rate_hz, frequency_hz, quality=30.0):
    """Zero-phase IIR notch at the specified line frequency."""
    fs = sampling_rate(sampling_rate_hz)
    if not 0 < frequency_hz < fs / 2 or quality <= 0:
        raise ValueError("Invalid notch frequency or quality")
    return signal.filtfilt(*signal.iirnotch(frequency_hz, quality, fs), signal_array(x))


def envelope_normalize(x, *, sampling_rate_hz, band_hz, lowpass_hz=3.0, epsilon=1e-6):
    """Normalize a bandpassed signal by its Hilbert amplitude envelope.

    A fourth-order Butterworth bandpass is applied with zero-phase filtering.
    The Hilbert amplitude can be smoothed by a second-order lowpass filter.
    The output is the filtered signal divided by ``envelope + epsilon``,
    then multiplied by the median envelope.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Input signal.
    sampling_rate_hz : float
        Sampling rate in Hz.
    band_hz : tuple of float
        Bandpass cutoffs in Hz.
    lowpass_hz : float or None
        Envelope smoothing cutoff in Hz.
    epsilon : float
        Positive denominator offset.

    Returns
    -------
    normalized : ndarray
        Envelope-normalized signal.
    filtered : ndarray
        Bandpassed signal before normalization.
    envelope : ndarray
        Hilbert amplitude envelope, smoothed if requested.
    """
    fs = sampling_rate(sampling_rate_hz)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    b, a = signal.butter(4, _band(band_hz, fs), btype="bandpass")
    filtered = signal.filtfilt(b, a, signal_array(x))
    envelope = np.abs(signal.hilbert(filtered))
    if lowpass_hz is not None:
        if not 0 < lowpass_hz < fs / 2:
            raise ValueError("Invalid envelope lowpass frequency")
        b, a = signal.butter(2, lowpass_hz / (fs / 2), btype="low")
        envelope = signal.filtfilt(b, a, envelope)
    normalized = filtered / (envelope + epsilon)
    normalized *= np.median(envelope)
    return normalized, filtered, envelope


def process(x, *, sampling_rate_hz, config):
    """Apply the configured processing steps to a one-dimensional signal.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Input signal.
    sampling_rate_hz : float
        Input sampling rate in Hz.
    config : PreprocessConfig
        Filtering, output rate, and envelope settings.

    Returns
    -------
    ProcessedSignal
        Processed signal, pre-normalization signal, envelope, and output rate.
    """
    fs = sampling_rate(sampling_rate_hz)
    x = signal_array(x)
    if not isinstance(config, PreprocessConfig):
        raise TypeError("config must be a PreprocessConfig")
    if config.notch_hz is not None:
        x = notch(
            x,
            sampling_rate_hz=fs,
            frequency_hz=config.notch_hz,
            quality=config.notch_quality,
        )
    if config.band_hz is not None:
        x = bandpass(
            x, sampling_rate_hz=fs, band_hz=config.band_hz, order=config.filter_order
        )
    if config.output_rate_hz is not None:
        target = sampling_rate(config.output_rate_hz)
        ratio = fs / target
        if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-10):
            raise ValueError("output_rate_hz must divide the input rate by an integer")
        if round(ratio) > 1:
            x = signal.decimate(
                x, int(round(ratio)), ftype="iir", zero_phase=True
            ).astype(float)
        fs = target
    bandpassed = np.asarray(x)
    if config.envelope_band_hz is None:
        return ProcessedSignal(bandpassed.copy(), bandpassed, None, fs)
    normalized, _, envelope = envelope_normalize(
        bandpassed,
        sampling_rate_hz=fs,
        band_hz=config.envelope_band_hz,
        lowpass_hz=config.envelope_lowpass_hz,
        epsilon=config.envelope_epsilon,
    )
    return ProcessedSignal(normalized, bandpassed, envelope, fs)
