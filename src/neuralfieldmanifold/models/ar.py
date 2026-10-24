"""Ordinary least-squares autoregression and explicit prediction modes."""

from dataclasses import dataclass

import numpy as np

from .._validation import positive_integer, signal_array
from ..embedding import lag_matrix


@dataclass(frozen=True)
class PredictionMetrics:
    mse: float
    mae: float
    correlation: float


def prediction_metrics(observed, predicted):
    observed, predicted = signal_array(observed), signal_array(predicted)
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted signals must have the same shape")
    yt, yp = observed - observed.mean(), predicted - predicted.mean()
    denominator = np.sqrt((yt @ yt) * (yp @ yp))
    correlation = float((yt @ yp) / denominator) if denominator else float("nan")
    return PredictionMetrics(
        float(np.mean((observed - predicted) ** 2)),
        float(np.mean(np.abs(observed - predicted))),
        correlation,
    )


@dataclass(frozen=True)
class ARFit:
    intercept: float
    coefficients: np.ndarray

    @property
    def order(self):
        return len(self.coefficients)

    @property
    def poles(self):
        return np.roots(np.r_[1.0, -self.coefficients])

    def predict(self, signal):
        """One-step predictions, aligned to ``signal[order:]``."""
        predictors, _ = lag_matrix(signal, order=self.order)
        return (
            np.column_stack([np.ones(len(predictors)), predictors])
            @ np.r_[self.intercept, self.coefficients]
        )

    def forecast(self, signal, *, start_sample, n_samples, refresh_samples=None):
        """Recursive forecast; optionally re-anchor to observed history.

        ``None`` is free-running. ``1`` is one-step/teacher-forced. Larger
        intervals reproduce the tutorials' hybrid prediction exactly.
        """
        x = signal_array(signal, min_samples=self.order)
        n_samples = positive_integer(n_samples, "n_samples")
        start_sample = positive_integer(start_sample, "start_sample")
        if not self.order <= start_sample <= len(x):
            raise ValueError("start_sample must have a complete observed lag history")
        if refresh_samples is not None:
            refresh_samples = positive_integer(refresh_samples, "refresh_samples")
            if start_sample + (
                (n_samples - 1) // refresh_samples
            ) * refresh_samples > len(x):
                raise ValueError(
                    "Observed signal is too short for the requested history refreshes"
                )
        history = x[start_sample - self.order : start_sample].tolist()
        predictions = []
        for t in range(n_samples):
            if refresh_samples is not None and t % refresh_samples == 0:
                absolute = start_sample + t
                history = x[absolute - self.order : absolute].tolist()
            value = self.intercept + np.dot(self.coefficients, list(reversed(history)))
            predictions.append(value)
            history.pop(0)
            history.append(value)
        return np.asarray(predictions)

    def information_criteria(self, signal):
        """Return AIC and BIC on the supplied signal using ``order + 1`` parameters."""
        signal = signal_array(signal, min_samples=self.order + 1)
        residual = signal[self.order :] - self.predict(signal)
        n = len(residual)
        variance = np.sum(residual**2) / n
        k = self.order + 1
        return {
            "aic": float(n * np.log(variance + 1e-12) + 2 * k),
            "bic": float(n * np.log(variance + 1e-12) + k * np.log(max(n, 2))),
        }


class AR:
    """AR model specification. ``fit`` returns an independent :class:`ARFit`."""

    def __init__(self, *, order):
        self.order = positive_integer(order, "order")

    def fit(self, signal):
        predictors, target = lag_matrix(signal, order=self.order)
        design = np.column_stack([np.ones(len(predictors)), predictors])
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        return ARFit(float(weights[0]), weights[1:])
