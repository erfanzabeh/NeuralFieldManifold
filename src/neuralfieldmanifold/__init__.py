"""Neural field dynamics and delay-coordinate geometry.

Import as ``import neuralfieldmanifold as nfm``. Numerical functions accept
arrays; recordings and fit results expose named attributes.
"""

from importlib import import_module

from . import datasets, embedding, geometry, preprocessing, simulation, spectral

__version__ = "0.1.0"
__all__ = [
    "datasets",
    "embedding",
    "geometry",
    "models",
    "plotting",
    "preprocessing",
    "simulation",
    "spectral",
    "training",
]


def __getattr__(name):
    if name in {"models", "training", "plotting"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
