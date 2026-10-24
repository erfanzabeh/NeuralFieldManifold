"""Classical dynamical systems with NumPy integration and random streams."""

import numpy as np

from .._validation import positive_integer


def lorenz(
    n_samples=12000,
    *,
    dt=0.005,
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
    initial_state=(1.0, 1.0, 1.0),
):
    """Fourth-order Runge--Kutta trajectory, shape ``(n_samples, 3)``.

    The first returned sample is the state after one integration step,
    matching the original tutorial. No transient is discarded implicitly.
    """
    n_samples = positive_integer(n_samples, "n_samples")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("initial_state must have three finite coordinates")
    x, y, z = state
    out = np.empty((n_samples, 3))

    def derivative(x, y, z):
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    for i in range(n_samples):
        k1 = derivative(x, y, z)
        k2 = derivative(
            x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2]
        )
        k3 = derivative(
            x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2]
        )
        k4 = derivative(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        x += (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        out[i] = x, y, z
    return out


def ornstein_uhlenbeck(
    n_samples,
    *,
    dt=0.005,
    theta=2.0,
    mean=0.0,
    sigma=1.0,
    initial=0.0,
    seed=0,
    method="exact",
    innovations=None,
):
    """OU process using exact or Euler transitions; first state follows one step.

    Supplied ``innovations`` are standard-normal draws, not scaled noise.
    """
    n_samples = positive_integer(n_samples, "n_samples")
    if dt <= 0 or theta <= 0 or sigma < 0 or method not in {"exact", "euler"}:
        raise ValueError("Invalid OU parameters")
    draws = (
        np.random.default_rng(seed).normal(size=n_samples)
        if innovations is None
        else np.asarray(innovations)
    )
    if draws.shape != (n_samples,) or not np.isfinite(draws).all():
        raise ValueError("Invalid innovations")
    rate = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
    state = initial
    out = np.empty(n_samples)
    for i, draw in enumerate(draws):
        state = (
            state * rate + mean * (1 - rate) + sd * draw
            if method == "exact"
            else state - theta * dt * (state - mean) + sigma * np.sqrt(dt) * draw
        )
        out[i] = state
    return out


def rotating_var(
    n_samples,
    *,
    burn_in_samples=300,
    alpha=(0.45, 0.35, 0.30),
    beta=(-0.10, -0.08, -0.06),
    periods_samples=(500.0, 900.0),
    noise_scale=1.0,
    seed=0,
    innovations=None,
):
    """Three-dimensional rotating VAR(2), returned as a NumPy array.

    Innovations have shape ``(n_samples + burn_in_samples, 3)`` and are
    supplied already scaled when provided.
    """
    n_samples = positive_integer(n_samples, "n_samples")
    if burn_in_samples < 0 or noise_scale < 0 or min(periods_samples) <= 0:
        raise ValueError("Invalid burn-in, noise scale, or rotation periods")
    alpha, beta = np.asarray(alpha), np.asarray(beta)
    if alpha.shape != (3,) or beta.shape != (3,):
        raise ValueError("alpha and beta must have three entries")
    total = n_samples + burn_in_samples
    noise = (
        np.random.default_rng(seed).normal(size=(total, 3)) * noise_scale
        if innovations is None
        else np.asarray(innovations)
    )
    if noise.shape != (total, 3) or not np.isfinite(noise).all():
        raise ValueError("Invalid innovations")

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def ry(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    previous, older = np.zeros(3), np.zeros(3)
    out = np.empty((total, 3))
    for k in range(total):
        t1 = 2 * np.pi * k / periods_samples[0]
        t2 = 2 * np.pi * k / periods_samples[1] + 0.7
        r1, r2 = rz(t1) @ ry(t1 * 0.5), ry(t2) @ rz(t2 * 0.5)
        current = (
            r1 @ np.diag(alpha) @ r1.T @ previous
            + r2 @ np.diag(beta) @ r2.T @ older
            + noise[k]
        )
        out[k] = current
        previous, older = current, previous
    return out[burn_in_samples:]
