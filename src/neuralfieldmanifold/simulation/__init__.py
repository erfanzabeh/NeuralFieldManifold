"""NumPy simulations and time-varying coefficient schedules."""

from . import schedules
from .autoregressive import SimulationRejected, ar, coefficients_from_modes, tvar
from .systems import lorenz, ornstein_uhlenbeck, rotating_var
from .noise import colored_noise, white_noise
from .mixing import mixing_matrix, oscillatory_ar

__all__ = [
    "schedules",
    "ar",
    "tvar",
    "coefficients_from_modes",
    "lorenz",
    "ornstein_uhlenbeck",
    "rotating_var",
    "colored_noise",
    "white_noise",
    "SimulationRejected",
    "mixing_matrix",
    "oscillatory_ar",
]
