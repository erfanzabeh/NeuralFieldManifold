import jax
import jax.numpy as jnp
from jax import random
from functools import partial


def _rotation_z(theta):
    """3D rotation matrix around z-axis."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def _rotation_y(theta):
    """3D rotation matrix around y-axis."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ])


@partial(jax.jit, static_argnums=(0, 1))
def tvvar(n_steps: int, burn_in: int = 300, alpha: jnp.ndarray = None, 
          beta: jnp.ndarray = None, rotation_period1: float = 500.0, 
          rotation_period2: float = 900.0, noise_scale: float = 1.0, 
          seed: int = 0) -> jnp.ndarray:
    """
    3D Time-Varying VAR(2) with rotating coupling matrices.
    
    Ensures stability by using orthogonal rotations to preserve eigenvalues
    while the coupling structure evolves smoothly over time.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps (after burn-in).
    burn_in : int
        Number of initial samples to discard.
    alpha : jnp.ndarray or None
        Eigenvalues for first-lag matrix A1 (length 3).
        Default: [0.45, 0.35, 0.30]
    beta : jnp.ndarray or None
        Eigenvalues for second-lag matrix A2 (length 3).
        Negative values help with damping.
        Default: [-0.10, -0.08, -0.06]
    rotation_period1 : float
        Period for first rotation (samples).
    rotation_period2 : float
        Period for second rotation (samples).
    noise_scale : float
        Scale of innovation noise.
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Signal of shape (n_steps, 3).
    
    Example
    -------
    >>> data = tvvar(5000)  # (5000, 3)
    >>> X1, X2, X3 = data[:, 0], data[:, 1], data[:, 2]
    """
    if alpha is None:
        alpha = jnp.array([0.45, 0.35, 0.30])
    if beta is None:
        beta = jnp.array([-0.10, -0.08, -0.06])
    
    key = random.PRNGKey(seed)
    T_total = n_steps + burn_in
    
    noise = random.normal(key, shape=(T_total, 3)) * noise_scale
    
    def step(carry, inputs):
        X_prev1, X_prev2 = carry
        k, eps = inputs
        
        th1 = 2.0 * jnp.pi * k / rotation_period1
        th2 = 2.0 * jnp.pi * k / rotation_period2 + 0.7
        
        R1 = _rotation_z(th1) @ _rotation_y(th1 * 0.5)
        R2 = _rotation_y(th2) @ _rotation_z(th2 * 0.5)
        
        A1 = R1 @ jnp.diag(alpha) @ R1.T
        A2 = R2 @ jnp.diag(beta) @ R2.T
        
        X_new = A1 @ X_prev1 + A2 @ X_prev2 + eps
        
        return (X_new, X_prev1), X_new
    
    init_carry = (jnp.zeros(3), jnp.zeros(3))
    inputs = (jnp.arange(T_total, dtype=jnp.float32), noise)
    
    _, X_full = jax.lax.scan(step, init_carry, inputs)
    
    return X_full[burn_in:]
