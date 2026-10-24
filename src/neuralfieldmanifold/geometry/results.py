"""Named settings and outputs shared by geometric fitting routines."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitConfig:
    """Control the compactness and hole penalties in geometric fitting.

    Parameters
    ----------
    regularization : float
        Nonnegative weight for compactness and hole penalties relative to
        point-containment residuals.
    hole_ratio : float or None
        Hole constraint ratio, between zero and one. None selects 0.5 for
        a ring or 0.75 for a torus.
    angular_modes : int
        Number of Fourier modes in the angular homogeneity penalty.
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
    """Fitted geometry, residual measures, and optimizer diagnostics.

    Attributes
    ----------
    kind : str
        Geometry type: ring or torus.
    center : ndarray, shape (3,)
        Center in input coordinates.
    normal, u_axis, v_axis : ndarray, shape (3,)
        Fitted orientation vectors.
    major_radii : tuple of float
        Ellipse radii along u_axis and v_axis.
    tube_radius : float
        Torus tube radius, or half the mean annular width for a ring.
    inner_radii : tuple of float or None
        Inner ellipse radii for a ring; None for a torus.
    score : float
        Fraction of points contained in the fitted geometry.
    error : float
        Mean outside distance, in input units.
    mse : float
        Mean squared outside distance. For rings, includes normal offsets.
    r_squared : float
        One minus the ratio of squared outside distances to total centered
        point-cloud variation. Ring outside distances are measured in-plane.
    hole_quality : float
        Relative hole-size measure of the fitted geometry.
    fraction_in_hole : float
        Fraction of points in the central hole.
    success : bool
        Whether the optimizer reported convergence.
    evaluations : int
        Number of optimizer function evaluations.
    optimizer_message : str
        Optimizer termination message.

    Notes
    -----
    Ring containment and mean error are measured in the fitted plane.
    Torus distances are three-dimensional.
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
