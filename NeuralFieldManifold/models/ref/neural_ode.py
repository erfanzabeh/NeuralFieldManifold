"""
Neural ODE + TVAR with Levinson-Durbin Stability
=================================================

A Neural ODE evolves a latent state z(t) over time. The latent is mapped 
through tanh to obtain reflection coefficients κ ∈ (-1, 1), which are 
converted to stable AR coefficients via Levinson-Durbin recursion.

Reference: Inspired by Neural ODE literature (Chen et al., 2018)
"""

import torch
import torch.nn as nn
from typing import Tuple


def levinson_order2(kappa: torch.Tensor) -> torch.Tensor:
    """
    Levinson-Durbin recursion for AR(2) from reflection coefficients.
    
    Maps reflection coefficients κ ∈ (-1, 1) to stable AR coefficients.
    
    Parameters
    ----------
    kappa : Tensor, shape (..., 2)
        Reflection coefficients (use tanh to constrain to (-1, 1))
    
    Returns
    -------
    a : Tensor, shape (..., 2)
        AR(2) coefficients [a1, a2] guaranteed to be stable
    """
    k1 = kappa[..., 0]
    k2 = kappa[..., 1]
    a2 = k2
    a1 = k1 * (1.0 - k2)
    return torch.stack([a1, a2], dim=-1)


def levinson_durbin(kappa: torch.Tensor) -> torch.Tensor:
    """
    General Levinson-Durbin recursion for AR(p) from reflection coefficients.
    
    Maps reflection coefficients κ ∈ (-1, 1)^p to stable AR coefficients.
    
    Parameters
    ----------
    kappa : Tensor, shape (..., p)
        Reflection coefficients (use tanh to constrain to (-1, 1))
    
    Returns
    -------
    a : Tensor, shape (..., p)
        AR(p) coefficients guaranteed to be stable
    """
    p = kappa.shape[-1]
    batch_shape = kappa.shape[:-1]
    device = kappa.device
    dtype = kappa.dtype
    
    # Initialize
    a = torch.zeros(*batch_shape, p, device=device, dtype=dtype)
    a[..., 0] = kappa[..., 0]
    
    for i in range(1, p):
        k_i = kappa[..., i]
        # Update previous coefficients
        a_prev = a[..., :i].clone()
        a[..., :i] = a_prev + k_i.unsqueeze(-1) * a_prev.flip(-1)
        a[..., i] = k_i
    
    return a


class ODEFunc(nn.Module):
    """
    Defines the ODE dynamics dz/dt = f_θ(z).
    
    A simple MLP that maps z → dz/dt.
    
    Parameters
    ----------
    dim : int
        Latent dimension (typically = AR order p)
    hidden : int
        Hidden layer dimension
    """
    
    def __init__(self, dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Encoder(nn.Module):
    """
    Maps an initial window of observations to the initial latent state z0.
    
    Parameters
    ----------
    L : int
        Window size (number of initial observations to encode)
    hidden : int
        Hidden dimension
    out_dim : int
        Output dimension (= AR order p)
    """
    
    def __init__(self, L: int = 30, hidden: int = 32, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.net(x0)


class NeuralODE_TVAR(nn.Module):
    """
    Neural ODE for Time-Varying AR estimation with guaranteed stability.
    
    The latent state z(t) evolves via an ODE, then is mapped to stable
    AR coefficients using Levinson-Durbin recursion.
    
    Parameters
    ----------
    p : int
        AR order
    L : int
        Window size for encoding z0
    hidden : int
        Hidden dimension for ODE function
    
    Example
    -------
    >>> model = NeuralODE_TVAR(p=2, L=30, hidden=32)
    >>> xhat, a, z = model(x_seq, phi_seq, dt=1.0)
    """
    
    def __init__(self, p: int = 2, L: int = 30, hidden: int = 32):
        super().__init__()
        self.p = p
        self.L = L
        self.func = ODEFunc(dim=p, hidden=hidden)
        self.enc = Encoder(L=L, hidden=hidden, out_dim=p)

    def forward(
        self, 
        x_seq: torch.Tensor, 
        phi_seq: torch.Tensor, 
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing.
        
        Parameters
        ----------
        x_seq : Tensor, shape (T,) or (B, T)
            Observed time series
        phi_seq : Tensor, shape (T, p) or (B, T, p)
            Lagged features [x_{t-1}, ..., x_{t-p}]
        dt : float
            Integration time step
        
        Returns
        -------
        xhat : Tensor, shape (T,) or (B, T)
            One-step predictions (teacher-forced)
        a : Tensor, shape (T, p) or (B, T, p)
            Time-varying AR coefficients
        z : Tensor, shape (T, p) or (B, T, p)
            Latent ODE trajectory
        """
        # Handle both batched and unbatched input
        if x_seq.dim() == 1:
            return self._forward_unbatched(x_seq, phi_seq, dt)
        else:
            return self._forward_batched(x_seq, phi_seq, dt)
    
    def _forward_unbatched(
        self, 
        x_seq: torch.Tensor, 
        phi_seq: torch.Tensor, 
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = x_seq.shape[0]
        p = self.p
        
        # Encode initial window to z0
        z0 = self.enc(x_seq[:self.L].unsqueeze(0)).squeeze(0)  # (p,)

        # Euler integration of ODE
        z_list = [z0]
        for _ in range(1, T):
            z_prev = z_list[-1]
            z_next = z_prev + dt * self.func(z_prev)
            z_list.append(z_next)
        z = torch.stack(z_list, dim=0)  # (T, p)

        # Map to stable AR coefficients via Levinson-Durbin
        kappa = torch.tanh(z)  # reflection coeffs in (-1, 1)
        if p == 2:
            a = levinson_order2(kappa)
        else:
            a = levinson_durbin(kappa)

        # One-step prediction
        xhat = torch.zeros(T, device=x_seq.device, dtype=x_seq.dtype)
        xhat[p:] = (a[p:] * phi_seq[p:]).sum(dim=1)
        
        return xhat, a, z
    
    def _forward_batched(
        self, 
        x_seq: torch.Tensor, 
        phi_seq: torch.Tensor, 
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = x_seq.shape
        p = self.p
        
        # Encode initial window to z0
        z0 = self.enc(x_seq[:, :self.L])  # (B, p)

        # Euler integration of ODE
        z_list = [z0]
        for _ in range(1, T):
            z_prev = z_list[-1]
            z_next = z_prev + dt * self.func(z_prev)
            z_list.append(z_next)
        z = torch.stack(z_list, dim=1)  # (B, T, p)

        # Map to stable AR coefficients
        kappa = torch.tanh(z)
        if p == 2:
            a = levinson_order2(kappa)
        else:
            a = levinson_durbin(kappa)

        # One-step prediction
        xhat = torch.zeros(B, T, device=x_seq.device, dtype=x_seq.dtype)
        xhat[:, p:] = (a[:, p:] * phi_seq[:, p:]).sum(dim=-1)
        
        return xhat, a, z
