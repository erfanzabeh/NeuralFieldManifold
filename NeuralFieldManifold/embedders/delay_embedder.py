"""
Delay Embedder (Takens)
=======================

State-space reconstruction from scalar time series via delay embedding.

Takens' Theorem:
    y_t = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]

This reconstructs an m-dimensional manifold that is diffeomorphic 
to the original attractor.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(1, 2))
def embed(x: jnp.ndarray, m: int, tau: int) -> jnp.ndarray:
    """
    General m-dimensional delay embedding.
    
    Parameters
    ----------
    x : jnp.ndarray
        1D time series.
    m : int
        Embedding dimension.
    tau : int
        Delay in samples.
    
    Returns
    -------
    jnp.ndarray
        Embedded matrix of shape (n - (m-1)*tau, m).
    """
    n = len(x) - (m - 1) * tau
    rows = jnp.arange(n)[:, None]
    cols = (m - 1 - jnp.arange(m)) * tau
    return x[rows + cols]

