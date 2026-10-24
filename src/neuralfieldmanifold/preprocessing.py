"""Explicit notebook-equivalent filtering and envelope normalization."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ._validation import sampling_rate, signal_array


@dataclass(frozen=True)
class PreprocessConfig:
    """Processing order: notch, bandpass, optional decimation, envelope.

    Bands are in Hz. ``None`` disables that step. ``output_rate_hz`` uses
    integer IIR decimation, as in the recorded-data tutorials; an exact
    integer ratio is required instead of silently changing the output rate.
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
    """Return normalized signal, envelope-band signal, and Hilbert envelope.

    Uses transfer-function Butterworth filters and median-envelope rescaling
    exactly as in the recorded-data notebooks. This differs intentionally
    from the SOS filter used by :func:`bandpass`.
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
    """Process a 1D signal using only the explicitly configured steps."""
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
