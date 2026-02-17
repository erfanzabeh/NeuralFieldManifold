"""
Neural Operator — Continuous Delay Kernel
==========================================

Model TVAR as an integral operator over continuous delays:

    ŷ_t = c(t) + ∫ k_t(τ) x(t-τ) dτ

The kernel k_t(τ) is parameterized via Fourier features over τ,
with time-varying amplitudes from a causal context encoder.

Key advantage: Sampling-rate invariant — can handle non-uniform Δt.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


def fractional_delay_samples(
    x: torch.Tensor, 
    tau_grid: Union[torch.Tensor, list], 
    dt: float, 
    t_offset: int = 0
) -> torch.Tensor:
    """
    Sample x(t - τ) with linear interpolation for continuous delays.
    
    Parameters
    ----------
    x : Tensor, shape (B, T)
        Input time series
    tau_grid : Tensor or list, shape (L,)
        Delay values in seconds
    dt : float
        Sampling interval in seconds
    t_offset : int
        Offset for absolute time indexing
    
    Returns
    -------
    Xlags : Tensor, shape (B, T, L)
        Interpolated lagged values
    """
    B, T = x.shape
    device, dtype = x.device, x.dtype

    if isinstance(tau_grid, torch.Tensor):
        tau_idx = tau_grid.to(device=device, dtype=dtype) / float(dt)
    else:
        tau_idx = torch.as_tensor(tau_grid, device=device, dtype=dtype) / float(dt)

    t_idx = torch.arange(T, device=device, dtype=dtype).view(1, T, 1) + float(t_offset)

    src = t_idx - tau_idx.view(1, 1, -1)  # [1, T, L]
    src0 = torch.clamp(torch.floor(src), 0, T - 1).to(torch.long)
    src1 = torch.clamp(src0 + 1, 0, T - 1)
    w = (src - src0.to(dtype)).to(dtype)

    idx0 = src0.expand(B, -1, -1).contiguous()
    idx1 = src1.expand(B, -1, -1).contiguous()

    L = tau_idx.numel()
    x_exp = x.unsqueeze(-1).repeat(1, 1, L).contiguous()

    x0 = torch.gather(x_exp, 1, idx0)
    x1 = torch.gather(x_exp, 1, idx1)
    
    return (1 - w) * x0 + w * x1


def total_variation_time(k: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer over time dimension."""
    return (k[:, 1:, :] - k[:, :-1, :]).abs().mean()


def l1_energy(k: torch.Tensor) -> torch.Tensor:
    """L1 sparsity regularizer."""
    return k.abs().mean()


class TVAROperator(nn.Module):
    """
    Time-Varying AR as a Neural Operator over continuous delays.
    
    Models the prediction as an integral:
        y_t = c(t) + ∫ k_t(τ) x(t-τ) dτ
    
    The kernel k_t(τ) is parameterized with Fourier features over τ,
    with time-varying amplitudes from a causal context encoder.
    
    Parameters
    ----------
    L : int
        Number of delay points to discretize
    tau_min : float
        Minimum delay (seconds)
    tau_max : float
        Maximum delay (seconds)
    n_modes : int
        Number of Fourier modes for kernel
    hidden : int
        Channels in context encoder
    
    Example
    -------
    >>> model = TVAROperator(L=128, tau_max=0.5, n_modes=16)
    >>> yhat = model(x, dt=0.005)  # x: [B, T]
    >>> yhat, k, c, Xlags = model(x, dt=0.005, return_kernel=True)
    """
    
    def __init__(
        self, 
        L: int = 128, 
        tau_min: float = 0.0, 
        tau_max: float = 0.5, 
        n_modes: int = 16, 
        hidden: int = 64
    ):
        super().__init__()
        assert tau_max > tau_min >= 0.0
        self.L = L
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.register_buffer("tau_grid", torch.linspace(tau_min, tau_max, L))

        # Causal context encoder (1D convs, left-padded)
        self.ctx = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=9, padding=8, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.ReLU()
        )

        # Fourier basis over τ
        self.register_buffer("freqs", torch.linspace(0.0, math.pi, n_modes))
        self.head_a = nn.Linear(hidden, n_modes)  # cos amplitudes
        self.head_b = nn.Linear(hidden, n_modes)  # sin amplitudes
        self.bias = nn.Linear(hidden, 1)  # c(t), time-varying intercept

        # Global gain for stability
        self.kernel_gain = nn.Parameter(torch.tensor(0.1))

    def make_kernel(self, h: torch.Tensor) -> torch.Tensor:
        """
        Generate time-varying kernel from context features.
        
        Parameters
        ----------
        h : Tensor, shape (B, T, H)
            Context features
        
        Returns
        -------
        k : Tensor, shape (B, T, L)
            Kernel values over delay grid
        """
        B, T, H = h.shape
        a = self.head_a(h)  # [B, T, M]
        b = self.head_b(h)  # [B, T, M]
        
        # Fourier features over τ
        tau = self.tau_grid.view(1, 1, self.L, 1)  # [1, 1, L, 1]
        omega = self.freqs.view(1, 1, 1, -1)  # [1, 1, 1, M]
        cosF = torch.cos(omega * tau)  # [1, 1, L, M]
        sinF = torch.sin(omega * tau)  # [1, 1, L, M]
        
        # Combine with time-varying amplitudes
        k = (a.unsqueeze(2) * cosF + b.unsqueeze(2) * sinF).sum(-1)  # [B, T, L]
        
        return self.kernel_gain * k

    def forward(
        self, 
        x: torch.Tensor, 
        dt: float, 
        t_offset: int = 0, 
        return_kernel: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T)
            Input time series
        dt : float
            Sampling interval in seconds
        t_offset : int
            Offset for absolute time indexing
        return_kernel : bool
            If True, also return intermediate values
        
        Returns
        -------
        yhat : Tensor, shape (B, T)
            Predicted values
        k : Tensor, shape (B, T, L)
            Kernel (if return_kernel=True)
        c : Tensor, shape (B, T)
            Bias (if return_kernel=True)
        Xlags : Tensor, shape (B, T, L)
            Lagged values (if return_kernel=True)
        """
        B, T = x.shape

        # Causal context features
        x1 = x.unsqueeze(1)  # [B, 1, T]
        h = self.ctx(F.pad(x1, (32, 0)))  # [B, H, T+32]
        h = h[..., -T:]  # Crop to length T
        hT = h.transpose(1, 2)  # [B, T, H]

        k = self.make_kernel(hT)  # [B, T, L]
        c = self.bias(hT).squeeze(-1)  # [B, T]

        # Sample lagged signal at continuous τ-grid
        Xlags = fractional_delay_samples(x, self.tau_grid, float(dt), t_offset=t_offset)

        # Riemann sum approximation of integral
        delta_tau = (self.tau_max - self.tau_min) / max(self.L - 1, 1)
        yhat = (k * Xlags).sum(-1) * delta_tau + c  # [B, T]

        if return_kernel:
            return yhat, k, c, Xlags
        return yhat
