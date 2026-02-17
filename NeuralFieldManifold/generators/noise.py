import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def white_noise(n_steps: int, seed: int = 0):
    """
    Generate white noise (flat PSD).
    
    Parameters
    ----------
    n_steps : int
        Number of samples.
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Noise signal of shape (n_steps,), zero mean, unit variance.
    
    Example
    -------
    >>> x = white_noise(10000)
    """
    key = random.PRNGKey(seed)
    return random.normal(key, shape=(n_steps,))


@partial(jax.jit, static_argnums=(0,))
def pink_noise(n_steps: int, seed: int = 0):
    """
    Generate pink (1/f) noise.
    
    Parameters
    ----------
    n_steps : int
        Number of samples.
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Noise signal of shape (n_steps,), zero mean, unit variance.
    
    Example
    -------
    >>> x = pink_noise(10000)
    """
    return colored_noise(n_steps, 1.0, seed)


@partial(jax.jit, static_argnums=(0,))
def brown_noise(n_steps: int, seed: int = 0):
    """
    Generate brown/red (1/f²) noise.
    
    Parameters
    ----------
    n_steps : int
        Number of samples.
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Noise signal of shape (n_steps,), zero mean, unit variance.
    
    Example
    -------
    >>> x = brown_noise(10000)
    """
    return colored_noise(n_steps, 2.0, seed)


@partial(jax.jit, static_argnums=(0,))
def colored_noise(n_steps: int, alpha: float = 1.0, seed: int = 0):
    """
    Generate 1/f^α colored noise via spectral shaping.
    
    Parameters
    ----------
    n_steps : int
        Number of samples.
    alpha : float
        Spectral exponent.
        - α=0: white noise (flat PSD)
        - α=1: pink noise (1/f)
        - α=2: brown/red noise (1/f²)
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Noise signal of shape (n_steps,), zero mean, unit variance.
    
    Example
    -------
    >>> x = colored_noise(10000, alpha=1.5)
    """
    key = random.PRNGKey(seed)
    key1, key2 = random.split(key)
    n_freq = n_steps // 2 + 1
    
    re = random.normal(key1, shape=(n_freq,))
    im = random.normal(key2, shape=(n_freq,))
    X = re + 1j * im
    
    freqs = jnp.fft.rfftfreq(n_steps)
    
    mag_shape = jnp.where(
        freqs > 0,
        1.0 / (freqs ** (alpha / 2.0)),
        0.0
    )
    
    X_shaped = X * mag_shape
    x = jnp.fft.irfft(X_shaped, n=n_steps)
    
    x = (x - jnp.mean(x)) / (jnp.std(x) + 1e-12)
    
    return x