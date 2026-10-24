"""Two-regime threshold autoregression with residual-bootstrap simulation."""

from dataclasses import dataclass

import numpy as np

from .._validation import positive_integer, signal_array
from ..embedding import lag_matrix
from .ar import ARFit


@dataclass(frozen=True)
class TARFit:
    lower: ARFit
    upper: ARFit
    threshold: float
    delay_samples: int
    lower_residuals: np.ndarray
    upper_residuals: np.ndarray

    def simulate(self, initial_lags, *, n_samples, seed=0, match_reference=None):
        """Simulate using newest-first initial lags and bootstrapped residuals.

        Mean/variance matching is performed only when a reference is
        explicitly supplied. It is not a held-out prediction metric.
        """
        n_samples = positive_integer(n_samples, "n_samples")
        state = (
            signal_array(initial_lags, min_samples=self.lower.order)
            .astype(float)
            .copy()
        )
        if len(state) != self.lower.order:
            raise ValueError("initial_lags must contain exactly order values")
        rng = np.random.default_rng(seed)
        out = np.empty(n_samples)
        for t in range(n_samples):
            is_lower = state[self.delay_samples - 1] <= self.threshold
            model = self.lower if is_lower else self.upper
            pool = self.lower_residuals if is_lower else self.upper_residuals
            innovation = rng.choice(pool) if len(pool) else 0.0
            out[t] = model.intercept + np.dot(model.coefficients, state) + innovation
            state[1:] = state[:-1]
            state[0] = out[t]
        if match_reference is not None:
            reference = signal_array(match_reference)
            out = (out - out.mean()) * (
                reference.std() / (out.std() or 1.0)
            ) + reference.mean()
        return out


class TAR:
    """Fit two OLS regimes separated by a delayed signal threshold."""

    def __init__(self, *, order, delay_samples=1, threshold=0.0):
        self.order = positive_integer(order, "order")
        self.delay_samples = positive_integer(delay_samples, "delay_samples")
        if self.delay_samples > self.order or not np.isfinite(threshold):
            raise ValueError("Require delay_samples <= order and a finite threshold")
        self.threshold = float(threshold)

    def fit(self, signal):
        x = signal_array(signal, min_samples=self.order + 2)
        predictors, targets = lag_matrix(x, order=self.order)
        variable = x[self.order - self.delay_samples : len(x) - self.delay_samples]
        threshold = self.threshold
        lower = variable <= threshold
        if min(lower.sum(), (~lower).sum()) < self.order + 5:
            raise ValueError(
                "Too few samples in a regime; select an explicit supported threshold"
            )
        models, residuals = [], []
        for mask in (lower, ~lower):
            design = np.column_stack([np.ones(mask.sum()), predictors[mask]])
            weights = np.linalg.lstsq(design, targets[mask], rcond=None)[0]
            models.append(ARFit(float(weights[0]), weights[1:]))
            residuals.append(targets[mask] - design @ weights)
        return TARFit(*models, threshold, self.delay_samples, *residuals)
