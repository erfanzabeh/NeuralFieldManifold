"""Cross-signal delay-cloud Procrustes comparison, without topology claims."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import procrustes

from ..embedding import delay
from .._validation import positive_integer


@dataclass(frozen=True)
class Disparity:
    mean: float
    standard_deviation: float
    pair_indices: np.ndarray
    values: np.ndarray


def pairwise_disparity(signals, *, dimension, tau_samples, target_points=2000):
    """Procrustes disparities for all pairs of ``(signal, time)`` rows.

    Uses the tutorial's floor-stride subsampling rule. ``target_points`` is
    a target, not a strict cap. Preserve correspondence between time samples;
    unrelated recordings are not interchangeable with simultaneous channels.
    """
    signals = np.asarray(signals)
    target_points = positive_integer(target_points, "target_points")
    if signals.ndim != 2 or len(signals) < 2:
        raise ValueError("Expected at least two equal-length 1D signals")
    clouds = [delay(x, dimension=dimension, tau_samples=tau_samples) for x in signals]
    clouds = [x[:: max(1, len(x) // target_points)] for x in clouds]
    size = min(map(len, clouds))
    pairs, values = [], []
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            _, _, value = procrustes(clouds[i][:size], clouds[j][:size])
            pairs.append((i, j))
            values.append(value)
    values = np.asarray(values)
    return Disparity(
        float(values.mean()), float(values.std()), np.asarray(pairs), values
    )
