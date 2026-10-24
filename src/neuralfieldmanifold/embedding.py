"""Delay coordinates and autoregressive design matrices, using NumPy."""

import numpy as np

from ._validation import positive_integer, signal_array


def delay(x, *, dimension, tau_samples):
    """Return ``[x(t), x(t-tau), ...]`` with shape ``(N-(d-1)*tau, d)``.

    Rows are ordered in time; the first column is the newest sample. No
    centering, scaling, projection, or delay selection is performed.
    """
    dimension = positive_integer(dimension, "dimension")
    tau_samples = positive_integer(tau_samples, "tau_samples")
    x = signal_array(x, min_samples=(dimension - 1) * tau_samples + 1)
    n = len(x) - (dimension - 1) * tau_samples
    rows = np.arange(n)[:, None]
    cols = (dimension - 1 - np.arange(dimension)) * tau_samples
    return x[rows + cols]


def lag_matrix(x, *, order):
    """Return lag-1-first predictors and targets for an AR model."""
    order = positive_integer(order, "order")
    x = signal_array(x, min_samples=order + 1)
    predictors = np.zeros((len(x) - order, order))
    for k in range(order):
        predictors[:, k] = x[order - 1 - k : len(x) - 1 - k]
    return predictors, x[order:].copy()


def windows(x, *, length_samples, step_samples):
    """Return complete sliding windows and their integer start indices."""
    length_samples = positive_integer(length_samples, "length_samples")
    step_samples = positive_integer(step_samples, "step_samples")
    x = signal_array(x, min_samples=length_samples)
    starts = np.arange(0, len(x) - length_samples + 1, step_samples)
    return np.stack([x[start : start + length_samples] for start in starts]), starts
