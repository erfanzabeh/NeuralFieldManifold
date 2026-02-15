"""
Generators
==========
"""

from .lorenz import lorenz
from .noise import white_noise, pink_noise, brown_noise, colored_noise
from .ou import ou_exact, ou_euler
from .tvvar import tvvar
from .tvar import (
    sinusoid,
    fourier,
    quasiperiodic,
    polynomial_drift,
    logistic_transition,
    multi_sigmoid,
    gaussian_bumps,
    smooth_random,
)

__all__ = [
    "lorenz",
    "white_noise",
    "pink_noise", 
    "brown_noise",
    "colored_noise",
    "ou_exact",
    "ou_euler",
    "tvar",
    "tvvar",
    "sinusoid",
    "fourier",
    "quasiperiodic",
    "polynomial_drift",
    "logistic_transition",
    "multi_sigmoid",
    "gaussian_bumps",
    "smooth_random",
]

