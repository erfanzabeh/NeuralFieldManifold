"""Autoregressive simulation with fixed or time-varying coefficients."""

import numpy as np

from .._validation import positive_integer, sampling_rate, signal_array


class SimulationRejected(ValueError):
    """A generated signal exceeded the requested amplitude or power bounds."""


def coefficients_from_modes(frequencies_hz, pole_radii, *, sampling_rate_hz):
    """Build AR(2K) coefficients from conjugate oscillatory pole pairs."""
    fs = sampling_rate(sampling_rate_hz)
    frequencies, radii = np.asarray(frequencies_hz), np.asarray(pole_radii)
    if (
        frequencies.ndim != 1
        or frequencies.shape != radii.shape
        or len(frequencies) == 0
    ):
        raise ValueError("Expected equally sized nonempty frequency and radius vectors")
    if not np.isfinite(frequencies).all() or np.any(
        (frequencies <= 0) | (frequencies >= fs / 2)
    ):
        raise ValueError("Frequencies must lie between zero and Nyquist")
    if not np.isfinite(radii).all() or np.any((radii <= 0) | (radii > 1)):
        raise ValueError("Pole radii must satisfy 0 < radius <= 1")
    polynomial = np.array([1.0])
    for frequency, radius in zip(frequencies, radii):
        angle = 2 * np.pi * frequency / fs
        polynomial = np.convolve(
            polynomial, [1.0, -2 * radius * np.cos(angle), radius**2]
        )
    return -polynomial[1:]


def ar(
    coefficients,
    *,
    n_samples,
    noise_std=1.0,
    seed=42,
    innovations=None,
    initial_lags=None,
):
    """Stationary AR with newest-lag-first coefficients, returned as float32.

    The first ``order`` samples are zero unless ``initial_lags`` (oldest
    first) is supplied. Innovations are additive samples, already scaled;
    supplying them bypasses random number generation.
    """
    coefficients = signal_array(coefficients)
    n_samples = positive_integer(n_samples, "n_samples")
    if n_samples <= len(coefficients) or noise_std < 0:
        raise ValueError("Invalid signal length or noise scale")
    noise = (
        np.random.default_rng(seed).normal(0, noise_std, n_samples)
        if innovations is None
        else signal_array(innovations)
    )
    if len(noise) != n_samples:
        raise ValueError("innovations must have n_samples entries")
    x = np.zeros(n_samples, dtype=np.float64)
    order = len(coefficients)
    if initial_lags is not None:
        initial = signal_array(initial_lags)
        if len(initial) != order:
            raise ValueError("initial_lags must have order entries")
        x[:order] = initial
    for t in range(order, n_samples):
        x[t] = coefficients @ x[t - order : t][::-1] + noise[t]
    return x.astype(np.float32)


def tvar(
    coefficients,
    *,
    noise_std=0.3,
    seed=42,
    burn_in_samples=500,
    target_power=None,
    power_window_samples=40,
    max_abs=1e6,
    innovations=None,
):
    """Simulate a signal with time-varying autoregressive coefficients.

    Parameters
    ----------
    coefficients : array_like, shape (n_samples, order)
        Coefficient path, with lag 1 first.
    noise_std : float
        Standard deviation of generated Gaussian innovations.
    seed : int
        Random seed.
    burn_in_samples : int
        Number of initial samples to discard. Burn-in uses the first
        coefficient vector.
    target_power : float or None
        Optional rolling mean-square power target. Enables per-sample
        multiplicative correction, clipped to 0.8-1.2, and rejects signals
        whose final rolling power exceeds the target.
    power_window_samples : int
        Window length for power control.
    max_abs : float
        Maximum permitted absolute amplitude.
    innovations : array_like or None
        Additive noise, already scaled, including burn-in samples.

    Returns
    -------
    ndarray, shape (n_samples,)
        Simulated signal with float32 dtype.

    Raises
    ------
    SimulationRejected
        If the amplitude or rolling-power bounds are exceeded.
    """
    coefficients = np.asarray(coefficients)
    if (
        coefficients.ndim != 2
        or min(coefficients.shape) < 1
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("coefficients must be finite with shape (time, order)")
    if not isinstance(burn_in_samples, int) or burn_in_samples < 0 or noise_std < 0:
        raise ValueError("Invalid burn-in or noise scale")
    window = positive_integer(power_window_samples, "power_window_samples")
    if target_power is not None and (
        not np.isfinite(target_power) or target_power <= 0
    ):
        raise ValueError("target_power must be positive")
    length, order = coefficients.shape
    total = length + burn_in_samples
    if total <= order:
        raise ValueError("Signal is shorter than its AR order")
    noise = (
        np.random.default_rng(seed).normal(0, noise_std, total)
        if innovations is None
        else signal_array(innovations)
    )
    if len(noise) != total:
        raise ValueError("innovations length must include burn-in")
    x = np.zeros(total, dtype=np.float64)
    x[:order] = noise[:order]
    for t in range(order, total):
        index = max(0, min(t - burn_in_samples, length - 1))
        x[t] = np.dot(coefficients[index], x[t - order : t][::-1]) + noise[t]
        if not np.isfinite(x[t]) or abs(x[t]) > max_abs:
            raise SimulationRejected("Simulation exceeded max_abs")
        if target_power is not None and t >= window - 1:
            power = np.mean(x[t - window + 1 : t + 1] ** 2)
            if power > 0:
                x[t] *= np.clip(np.sqrt(target_power / power), 0.8, 1.2)
    result = x[burn_in_samples:].astype(np.float32)
    if target_power is not None and len(result) >= window:
        cumulative = np.r_[0.0, np.cumsum(result**2)]
        power = (cumulative[window:] - cumulative[:-window]) / window
        if np.any(power > target_power):
            raise SimulationRejected("Simulation exceeded the rolling power target")
    return result
