"""Elliptical ring and torus fits; geometric containment, not topology tests."""

from .results import FitConfig, GeometryFit
from .ring import fit_ring
from .torus import fit_torus
from .disparity import Disparity, pairwise_disparity

__all__ = [
    "FitConfig",
    "GeometryFit",
    "fit_ring",
    "fit_torus",
    "Disparity",
    "pairwise_disparity",
]
