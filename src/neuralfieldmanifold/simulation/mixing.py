"""Oscillatory AR sources and distance-weighted linear sensor mixtures."""

import numpy as np

from .._validation import positive_integer, sampling_rate


def oscillatory_ar(
    frequencies_hz,
    pole_radii,
    *,
    sampling_rate_hz,
    n_samples,
    noise_std=0.05,
    seed=0,
    random_initialization=False,
    amplitude=1.0,
):
    """Generate an oscillatory AR source scaled to the requested amplitude.

    The AR characteristic polynomial combines conjugate pole pairs.
    The output is divided by its standard deviation and multiplied
    by amplitude.

    ``random_initialization=True`` initializes the lag history with Gaussian
    draws. Otherwise the initial history is zero.
    """
    fs = sampling_rate(sampling_rate_hz)
    n_samples = positive_integer(n_samples, "n_samples")
    frequencies, radii = np.asarray(frequencies_hz), np.asarray(pole_radii)
    if (
        frequencies.ndim != 1
        or frequencies.shape != radii.shape
        or not frequencies.size
    ):
        raise ValueError("Expected matching nonempty mode vectors")
    if (
        not np.isfinite(frequencies).all()
        or not np.isfinite(radii).all()
        or np.any((frequencies <= 0) | (frequencies >= fs / 2))
        or np.any((radii <= 0) | (radii > 1))
        or noise_std < 0
    ):
        raise ValueError("Invalid mode parameters")
    rng = np.random.default_rng(seed)
    poles = [
        root
        for f, r in zip(frequencies, radii)
        for root in (
            r * np.exp(1j * 2 * np.pi * f / fs),
            r * np.exp(-1j * 2 * np.pi * f / fs),
        )
    ]
    coefficients = np.real(-np.poly(poles)[1:])
    order = len(coefficients)
    if n_samples <= order:
        raise ValueError("n_samples must exceed twice the mode count")
    x = np.zeros(n_samples)
    noise = rng.normal(0, noise_std, n_samples)
    if random_initialization:
        x[:order] = rng.normal(0, noise_std, order)
    if len(frequencies) == 1:
        angle = 2 * np.pi * frequencies[0] / fs
        first, second = 2 * radii[0] * np.cos(angle), -(radii[0] ** 2)
        for t in range(2, n_samples):
            x[t] = first * x[t - 1] + second * x[t - 2] + noise[t]
    else:
        for t in range(order, n_samples):
            x[t] = sum(coefficients[i] * x[t - i - 1] for i in range(order)) + noise[t]
    return amplitude * x / (np.std(x) + 1e-12)


def mixing_matrix(*, n_sensors, n_sources, spatial_spread):
    """Row-normalized Gaussian distance weights on uniformly spaced locations."""
    n_sensors = positive_integer(n_sensors, "n_sensors")
    n_sources = positive_integer(n_sources, "n_sources")
    if not np.isfinite(spatial_spread) or spatial_spread <= 0:
        raise ValueError("spatial_spread must be positive")
    sources, sensors = np.linspace(0, 1, n_sources), np.linspace(0, 1, n_sensors)
    distances = np.abs(sensors[:, None] - sources[None, :])
    weights = np.exp(-(distances**2) / (2 * spatial_spread**2))
    return weights / weights.sum(axis=1, keepdims=True)
