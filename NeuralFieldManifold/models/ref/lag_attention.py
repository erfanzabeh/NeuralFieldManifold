"""
Lag-Attention TVAR (Transformer-based)
======================================

Build a "lag bank" of past values and use attention to learn which lags 
are important at each time step. Sparse top-k selection focuses on the 
most relevant lags.

Two variants:
- LagAttentionTVAR: Full transformer encoder over lag tokens
- LagAttentionTVARFast: Bilinear scoring (no transformer, faster)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def build_lag_bank(x_tensor: torch.Tensor, L: int) -> torch.Tensor:
    """
    Build lag bank from time series.
    
    Parameters
    ----------
    x_tensor : Tensor, shape (B, T)
        Input time series
    L : int
        Number of lags
    
    Returns
    -------
    Xlags : Tensor, shape (B, T, L)
        Xlags[:, t, l] = x_{t-(l+1)}, left-padded with zeros
    """
    B, T = x_tensor.shape
    pads = F.pad(x_tensor, (L, 0))  # [B, T+L]
    idx_base = torch.arange(T, device=x_tensor.device).view(1, T, 1)
    lag_offsets = torch.arange(L, device=x_tensor.device).view(1, 1, L)
    gather_idx = L - 1 + idx_base + lag_offsets
    Xlags = pads[:, gather_idx.squeeze(0)]
    return Xlags


def topk_mask_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask all but top-k logits with -inf for sparse softmax."""
    if k >= logits.shape[-1]:
        return logits
    topk = torch.topk(logits, k, dim=-1)
    mask = torch.full_like(logits, float('-inf'))
    return mask.scatter(-1, topk.indices, topk.values)


class LagAttentionTVAR(nn.Module):
    """
    Lag-Attention TVAR with full Transformer encoder.
    
    Learns time-varying attention weights over a bank of lagged values.
    Uses sparse top-k selection for efficiency.
    
    Parameters
    ----------
    L : int
        Number of lags in the bank
    d_model : int
        Model dimension
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads
    topk : int
        Top-k sparsity in attention
    dropout : float
        Dropout rate
    use_var : bool
        If True, also predict log-variance for NLL loss
    
    Example
    -------
    >>> model = LagAttentionTVAR(L=128, d_model=64, topk=8)
    >>> mu, logvar, w = model(x)  # x: [B, T]
    """
    
    def __init__(
        self, 
        L: int = 256, 
        d_model: int = 128, 
        n_layers: int = 2, 
        n_heads: int = 4, 
        topk: int = 8, 
        dropout: float = 0.1,
        use_var: bool = False
    ):
        super().__init__()
        self.L = L
        self.topk = topk
        self.use_var = use_var

        # Embeddings for lag indices and values
        self.lag_embed = nn.Embedding(L + 1, d_model)
        self.val_proj = nn.Linear(1, d_model)

        # Transformer encoder for lag tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, 
            batch_first=True
        )
        self.lag_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Causal context (conv over past)
        self.ctx_conv = nn.Conv1d(1, d_model, kernel_size=9, padding=8, dilation=1)
        self.ctx_proj = nn.Linear(d_model, d_model)

        # Scoring: pair(lag_enc, context) → scalar
        self.score = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.bias_head = nn.Linear(d_model, 1)
        
        if use_var:
            self.logvar_head = nn.Linear(d_model, 1)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T)
        
        Returns
        -------
        mu : Tensor, shape (B, T)
            Predicted mean
        logvar : Tensor or None
            Log-variance (if use_var=True)
        w : Tensor, shape (B, T, L)
            Attention weights over lags
        """
        B, T = x.shape
        L = self.L

        Xlags = build_lag_bank(x, L)  # [B, T, L]
        lag_vals = Xlags.unsqueeze(-1)  # [B, T, L, 1]
        lag_ids = torch.arange(1, L + 1, device=x.device).view(1, 1, L).expand(B, T, L)

        Hval = self.val_proj(lag_vals).squeeze(-2)  # [B, T, L, d]
        Hidx = self.lag_embed(lag_ids)  # [B, T, L, d]
        Hlag = Hval + Hidx  # [B, T, L, d]

        # Encode lag tokens per time step
        Hlag_flat = Hlag.view(B * T, L, -1)
        Henc = self.lag_encoder(Hlag_flat).view(B, T, L, -1)  # [B, T, L, d]

        # Causal context from x
        ctx = self.ctx_conv(x.unsqueeze(1))  # [B, d, T+pad]
        ctx = ctx[..., :T].transpose(1, 2)  # [B, T, d]
        ctx = self.ctx_proj(ctx)  # [B, T, d]

        # Score lags with context
        ctx_exp = ctx.unsqueeze(2).expand(-1, -1, L, -1)
        pair = torch.cat([Henc, ctx_exp], dim=-1)  # [B, T, L, 2d]
        logits = self.score(pair).squeeze(-1)  # [B, T, L]

        logits_masked = topk_mask_logits(logits, k=self.topk)
        w = torch.softmax(logits_masked, dim=-1)  # [B, T, L]

        mu_ar = (w * Xlags).sum(dim=-1)  # [B, T]
        c = self.bias_head(ctx).squeeze(-1)  # [B, T]
        mu = mu_ar + c

        logvar = None
        if self.use_var:
            logvar = self.logvar_head(ctx).squeeze(-1).clamp(-8, 8)
        
        return mu, logvar, w


class LagAttentionTVARFast(nn.Module):
    """
    Fast Lag-Attention TVAR (no Transformer, bilinear scoring).
    
    Uses bilinear attention: score = <Wq·ctx, Wk·Hlag> instead of
    full transformer, making it much faster for long sequences.
    
    Parameters
    ----------
    L : int
        Number of lags
    d_model : int
        Model dimension
    topk : int
        Top-k sparsity
    use_var : bool
        If True, predict log-variance
    
    Example
    -------
    >>> model = LagAttentionTVARFast(L=256, d_model=64, topk=16)
    >>> mu, logvar, w = model(x)
    """
    
    def __init__(
        self, 
        L: int = 256, 
        d_model: int = 128, 
        topk: int = 8, 
        use_var: bool = False
    ):
        super().__init__()
        self.L = L
        self.topk = topk
        self.use_var = use_var

        self.lag_embed = nn.Embedding(L + 1, d_model)
        self.val_proj = nn.Linear(1, d_model)

        # Causal context (left-padded conv)
        self.ctx_pad = nn.ConstantPad1d((8, 0), 0)
        self.ctx_conv = nn.Conv1d(1, d_model, kernel_size=9, padding=0)
        self.ctx_proj = nn.Linear(d_model, d_model)

        # Bilinear scorer
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)

        self.bias_head = nn.Linear(d_model, 1)
        if use_var:
            self.logvar_head = nn.Linear(d_model, 1)

        # Init for stability
        for m in [self.Wq, self.Wk, self.val_proj, self.bias_head]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, T = x.shape
        L = self.L

        Xlags = build_lag_bank(x, L)  # [B, T, L]
        lag_vals = Xlags.unsqueeze(-1)  # [B, T, L, 1]
        lag_ids = torch.arange(1, L + 1, device=x.device).view(1, 1, L).expand(B, T, L)

        Hval = self.val_proj(lag_vals).squeeze(-2)  # [B, T, L, d]
        Hidx = self.lag_embed(lag_ids)  # [B, T, L, d]
        Hlag = Hval + Hidx  # [B, T, L, d]

        # Causal context
        ctx = self.ctx_conv(self.ctx_pad(x.unsqueeze(1)))  # [B, d, T]
        ctx = ctx.transpose(1, 2)  # [B, T, d]
        ctx = self.ctx_proj(ctx)  # [B, T, d]

        # Bilinear scores
        q = self.Wq(ctx)  # [B, T, d]
        k = self.Wk(Hlag)  # [B, T, L, d]
        logits = torch.einsum('btd,btld->btl', q, k)  # [B, T, L]

        # Top-k masking
        if (self.topk is not None) and (self.topk < L):
            topk_vals = torch.topk(logits, self.topk, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            logits = mask.scatter(-1, topk_vals.indices, topk_vals.values)

        w = torch.softmax(logits, dim=-1)  # [B, T, L]
        mu_ar = (w * Xlags).sum(dim=-1)  # [B, T]
        c = self.bias_head(ctx).squeeze(-1)  # [B, T]
        mu = mu_ar + c

        logvar = None
        if self.use_var:
            logvar = self.logvar_head(ctx).squeeze(-1).clamp(-8, 8)
        
        return mu, logvar, w


def gaussian_nll(
    y: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    
    Parameters
    ----------
    y : Tensor
        Target values
    mu : Tensor
        Predicted mean
    logvar : Tensor or None
        Predicted log-variance (if None, assumes unit variance)
    
    Returns
    -------
    nll : Tensor
        Element-wise NLL
    """
    if logvar is None:
        logvar = torch.zeros_like(mu)
    return 0.5 * (logvar + (y - mu)**2 / (logvar.exp() + 1e-8))
