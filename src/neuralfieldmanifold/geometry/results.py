"""Named settings and outputs shared by geometric fitting routines."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitConfig:
    """Containment/compactness tradeoff and angular homogeneity settings.

    ``hole_ratio=None`` selects 0.5 for a ring and 0.75 for a torus.
    These are geometric fitting settings, not topological constraints.
    """

    regularization: float = 0.1
    hole_ratio: float | None = None
    angular_modes: int = 3

    def __post_init__(self):
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if self.hole_ratio is not None and not 0 < self.hole_ratio < 1:
            raise ValueError("hole_ratio must be between zero and one")
        if not isinstance(self.angular_modes, int) or self.angular_modes < 0:
            raise ValueError("angular_modes must be a nonnegative integer")


@dataclass(frozen=True)
class GeometryFit:
    """Fitted geometry and native numerical diagnostics.

    ``score`` is fraction contained; ``error`` is mean outside distance.
    Ring containment/error use the fitted plane, while ring ``mse`` includes
    out-of-plane distance. Torus distances are three-dimensional. Neither
    score nor error is a persistent-homology or topology guarantee.
    """

    kind: str
    center: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    major_radii: tuple[float, float]
    tube_radius: float
    inner_radii: tuple[float, float] | None
    score: float
    error: float
    mse: float
    r_squared: float
    hole_quality: float
    fraction_in_hole: float
    success: bool
    evaluations: int
    optimizer_message: str

    @property
    def features(self):
        """15 geometry features: radii, fit quality, normal, and plane axes.

        Order is R1, R2, tube radius, MSE, error, score, normal xyz,
        u-axis xyz, v-axis xyz. Orientation signs follow the fitted PCA
        frame; this function does not impose an additional sign convention.
        """
        return np.r_[
            self.major_radii,
            self.tube_radius,
            self.mse,
            self.error,
            self.score,
            self.normal,
            self.u_axis,
            self.v_axis,
        ]


def validate_cloud(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Expected at least four points of shape (N, 3)")
    if not np.isfinite(points).all() or np.any(np.ptp(points, axis=0) <= 0):
        raise ValueError("Point cloud must be finite with nonzero coordinate extents")
    return points
