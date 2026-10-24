"""Shared validation for signal arrays, dimensions, and sampling rates."""

import numpy as np


def signal_array(x, *, min_samples=1):
    x = np.asarray(x)
    if x.ndim != 1 or x.size < min_samples:
        raise ValueError(f"Expected a 1D signal with at least {min_samples} samples")
    if (
        not np.issubdtype(x.dtype, np.number)
        or np.iscomplexobj(x)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite real-valued signal samples")
    return x


def positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def sampling_rate(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite")
    return float(value)
