"""
Transformer AR (Fixed Lag Order)
================================

Standard transformer encoder operating on a fixed-size lag sequence. 
Uses a CLS token to summarize the lag information and predict the next value.
"""

import torch
import torch.nn as nn


class TransformerAR(nn.Module):
    """
    Transformer-based AR model with fixed lag order.
    
    Uses a CLS token to summarize the lag sequence and predict the next value.
    Suitable for learning nonlinear interactions between lags.
    
    Parameters
    ----------
    Pmax : int
        Maximum lag order (sequence length)
    d_model : int
        Transformer model dimension
    nhead : int
        Number of attention heads
    depth : int
        Number of transformer layers
    dropout : float
        Dropout rate
    
    Example
    -------
    >>> model = TransformerAR(Pmax=20, d_model=64, depth=2)
    >>> X_seq = torch.randn(32, 20, 1)  # [B, Pmax, 1]
    >>> yhat = model(X_seq)  # [B]
    """
    
    def __init__(
        self, 
        Pmax: int, 
        d_model: int = 64, 
        nhead: int = 4, 
        depth: int = 2, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.Pmax = Pmax
        self.d_model = d_model

        # Project scalar lag values to d_model
        self.in_proj = nn.Linear(1, d_model)

        # Learned positional embeddings for lag positions 0..Pmax-1
        self.pos = nn.Parameter(torch.zeros(1, Pmax, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        # CLS token to summarize the sequence
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        
        # Output projection
        self.out = nn.Linear(d_model, 1)

    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X_seq : Tensor, shape (B, Pmax, 1)
            Lag sequence where position 0 is x_{t-1}, position 1 is x_{t-2}, etc.
        
        Returns
        -------
        yhat : Tensor, shape (B,)
            Predicted next value
        """
        h = self.in_proj(X_seq) + self.pos  # (B, Pmax, d)
        cls = self.cls.expand(X_seq.size(0), 1, -1)  # (B, 1, d)
        h = torch.cat([cls, h], dim=1)  # (B, 1+Pmax, d)

        h = self.encoder(h)  # (B, 1+Pmax, d)
        h_cls = h[:, 0, :]  # (B, d) — CLS token output
        yhat = self.out(h_cls).squeeze(-1)  # (B,)
        
        return yhat
