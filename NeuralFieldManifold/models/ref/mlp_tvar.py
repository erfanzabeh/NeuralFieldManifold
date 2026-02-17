"""
MLP TVAR — Hyper-Network
========================

A "hyper-network" MLP that takes the lag vector as input and outputs 
time-varying AR coefficients. The prediction is a learned linear 
combination of lags.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLPTVAR(nn.Module):
    """
    MLP-based Time-Varying AR model (Hyper-Network approach).
    
    The MLP takes the lag vector z_t = [x_{t-1}, ..., x_{t-p}] and outputs
    time-varying coefficients a(t) and bias b(t). Prediction is:
    
        x̂_t = a(z_t)ᵀ z_t + b(z_t)
    
    Parameters
    ----------
    p_max : int
        Maximum lag order
    hidden : int
        Hidden dimension
    depth : int
        Number of hidden layers
    dropout : float
        Dropout rate
    
    Example
    -------
    >>> model = MLPTVAR(p_max=10, hidden=64, depth=2)
    >>> Z = torch.randn(32, 10)  # [B, p_max]
    >>> pred, coeffs, bias = model(Z)
    """
    
    def __init__(
        self, 
        p_max: int, 
        hidden: int = 128, 
        depth: int = 3, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = p_max
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)

        # Output: [a_1, ..., a_p_max, b]
        self.head = nn.Linear(hidden, p_max + 1)
        self.p_max = p_max

    def forward(
        self, 
        Z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        Z : Tensor, shape (B, p_max)
            Lag features [x_{t-1}, ..., x_{t-p}]
        
        Returns
        -------
        pred : Tensor, shape (B,)
            Predicted next value
        coeffs : Tensor, shape (B, p_max)
            Learned time-varying AR coefficients
        bias : Tensor, shape (B,)
            Learned time-varying bias
        """
        h = self.backbone(Z)
        out = self.head(h)  # [B, p_max+1]
        
        coeffs = out[:, :self.p_max]  # [B, p_max]
        bias = out[:, self.p_max:]  # [B, 1]
        
        # Dot product + bias
        pred = (coeffs * Z).sum(dim=1, keepdim=True) + bias
        
        return pred.squeeze(1), coeffs, bias.squeeze(1)
