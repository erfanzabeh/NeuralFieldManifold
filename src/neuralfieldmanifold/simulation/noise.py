"""NumPy random signals. Random streams are not JAX seed-compatible."""

import numpy as np

from .._validation import positive_integer


def white_noise(n_samples, *, seed=0):
    """Return standard-normal samples from NumPy's default generator."""
    return np.random.default_rng(seed).normal(
        size=positive_integer(n_samples, "n_samples")
    )


def colored_noise(n_samples, *, exponent=1.0, seed=0, spectrum=None):
    """Unit-variance spectral noise with power proportional to 1/f**exponent.

    ``spectrum`` optionally supplies unshaped complex random Fourier bins
    for controlled comparisons and replay, independent of the RNG backend.
    Otherwise real and imaginary components use NumPy's generator.
    """
    n_samples = positive_integer(n_samples, "n_samples")
    if n_samples < 2 or not np.isfinite(exponent):
        raise ValueError("Require n_samples >= 2 and finite exponent")
    if spectrum is None:
        rng = np.random.default_rng(seed)
        size = n_samples // 2 + 1
        spectrum = rng.normal(size=size) + 1j * rng.normal(size=size)
    spectrum = np.asarray(spectrum)
    if spectrum.shape != (n_samples // 2 + 1,) or not np.isfinite(spectrum).all():
        raise ValueError("spectrum must have n_samples // 2 + 1 finite complex entries")
    frequencies = np.fft.rfftfreq(n_samples)
    weights = np.zeros_like(frequencies)
    weights[1:] = 1.0 / frequencies[1:] ** (exponent / 2)
    x = np.fft.irfft(spectrum * weights, n=n_samples)
    return (x - x.mean()) / (x.std() + 1e-12)
