import jax
import jax.numpy as jnp
from jax import random
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def ou_exact(n_steps: int, dt: float = 0.005, theta: float = 2.0, mu: float = 0.0, 
             sigma: float = 1.0, x0: float = 0.0, seed: int = 0):
    """
    Ornstein-Uhlenbeck process using exact analytical solution.
    
    The conditional distribution P(x_{t+dt} | x_t) is Gaussian with:
        mean = x_t * exp(-θ*dt) + μ * (1 - exp(-θ*dt))
        var  = σ² / (2θ) * (1 - exp(-2θ*dt))
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    theta : float
        Reversion rate. Characteristic time scale τ = 1/θ.
    mu : float
        Long-term mean.
    sigma : float
        Noise strength.
    x0 : float
        Initial state.
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Signal of shape (n_steps,).
    
    Example
    -------
    >>> x = ou_exact(10000, dt=0.005, theta=2.0)
    """
    key = random.PRNGKey(seed)
    
    exp_neg_theta_dt = jnp.exp(-theta * dt)
    one_minus_exp = 1.0 - exp_neg_theta_dt
    var = (sigma**2 / (2.0 * theta)) * (1.0 - jnp.exp(-2.0 * theta * dt))
    sd = jnp.sqrt(var)
    
    noise = random.normal(key, shape=(n_steps,)) * sd
    
    def step(x_prev, eps):
        mean = x_prev * exp_neg_theta_dt + mu * one_minus_exp
        x_new = mean + eps
        return x_new, x_new
    
    _, x = jax.lax.scan(step, x0, noise)
    
    return x


@partial(jax.jit, static_argnums=(0,))
def ou_euler(n_steps: int, dt: float = 0.005, theta: float = 2.0, mu: float = 0.0,
             sigma: float = 1.0, x0: float = 0.0, seed: int = 0):
    """
    Ornstein-Uhlenbeck process using Euler-Maruyama approximation.
    
    Approximation (first-order in dt):
        x_{t+dt} ≈ x_t - θ*dt*(x_t - μ) + σ*sqrt(dt)*ε
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    theta : float
        Reversion rate. Characteristic time scale τ = 1/θ.
    mu : float
        Long-term mean.
    sigma : float
        Noise strength.
    x0 : float
        Initial state.
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Signal of shape (n_steps,).
    
    Example
    -------
    >>> x = ou_euler(10000, dt=0.005, theta=2.0)
    """
    key = random.PRNGKey(seed)
    
    sd = sigma * jnp.sqrt(dt)
    noise = random.normal(key, shape=(n_steps,)) * sd
    
    def step(x_prev, eps):
        mean = x_prev - theta * dt * (x_prev - mu)
        x_new = mean + eps
        return x_new, x_new
    
    _, x = jax.lax.scan(step, x0, noise)
    
    return x
