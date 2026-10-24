"""White and power-law colored noise generators."""

import numpy as np

from .._validation import positive_integer


def white_noise(n_samples, *, seed=0):
    """Return standard-normal samples from NumPy's default generator."""
    return np.random.default_rng(seed).normal(
        size=positive_integer(n_samples, "n_samples")
    )


def colored_noise(n_samples, *, exponent=1.0, seed=0, spectrum=None):
    """Generate zero-mean, unit-variance noise with a power-law spectrum.

    Parameters
    ----------
    n_samples : int
        Number of output samples, at least two.
    exponent : float
        Spectral exponent; power is proportional to 1/f**exponent.
    seed : int
        Seed for the NumPy random generator.
    spectrum : array_like of complex or None
        Optional unweighted Fourier bins, shape (n_samples // 2 + 1,).
        When supplied, these replace the random Fourier coefficients.

    Returns
    -------
    ndarray, shape (n_samples,)
        Real-valued colored noise.
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
