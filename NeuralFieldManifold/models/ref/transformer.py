import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerTVAR(nn.Module):
    """
    Transformer-based Time-Varying AR model with linear coefficient outputs.

    Processes lag tokens with optional time conditioning to produce
    time-varying AR coefficients. Prediction is forced to be linear:
    yhat = sum_k a_k(t) * x_{t-k} + b(t)
    """

    def __init__(
        self,
        max_ar_order=6,
        n_classes=5,
        d_model=64,
        nhead=4,
        depth=2,
        dropout=0.0,
        use_time=True,
    ):
        """
        Args:
            max_ar_order: Maximum AR order (number of lag tokens)
            n_classes: Number of AR order classes for classification head
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            depth: Number of transformer encoder layers
            dropout: Dropout rate
            use_time: Whether to condition on time index
        """
        super().__init__()
        self.max_ar_order = max_ar_order
        self.n_classes = n_classes
        self.d_model = d_model
        self.use_time = use_time

        # Project scalar lag token -> d_model
        self.in_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Time embedding (if using time)
        if self.use_time:
            self.time_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        # Learned positional embedding for lag index
        self.pos = nn.Parameter(torch.zeros(1, max_ar_order, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

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

        # Output heads
        self.coeff_head = nn.Linear(d_model, 1)  # Per-lag coefficient
        self.bias_head = nn.Linear(d_model, 1)  # Bias from pooled representation
        self.p_head = nn.Linear(d_model, n_classes)  # Order classification

        # Small init for stable early predictions
        nn.init.normal_(self.coeff_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.normal_(self.bias_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias_head.bias)
        nn.init.normal_(self.p_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.p_head.bias)

    def forward(self, x, t_input=None):
        """
        Args:
            x: (N, max_ar_order) or (N, max_ar_order, 1) lag window
            t_input: (N,) or (N, 1) normalized time index (optional)

        Returns:
            coeffs: (N, max_ar_order) predicted AR coefficients
            p_logits: (N, n_classes) logits for order classification
            p_hard: (N,) predicted order class (argmax)
            x_hat: (N,) predicted next value
        """
        # Handle input shape
        if x.dim() == 2:
            x_seq = x.unsqueeze(-1)  # (N, P) -> (N, P, 1)
        else:
            x_seq = x  # Already (N, P, 1)

        N = x_seq.shape[0]

        # Encode lag tokens with positional embedding
        h = self.in_proj(x_seq) + self.pos  # (N, P, d_model)

        # Add time conditioning if available
        if self.use_time and t_input is not None:
            if t_input.dim() == 1:
                t_input = t_input.unsqueeze(-1)  # (N,) -> (N, 1)
            t_emb = self.time_proj(t_input.unsqueeze(1))  # (N, 1, d_model)
            h = h + t_emb  # Broadcast to all lag positions

        # Transformer encoding
        h = self.encoder(h)  # (N, P, d_model)

        # Pooled representation for bias and order prediction
        h_pooled = h.mean(dim=1)  # (N, d_model)

        # Per-lag coefficients
        coeffs = self.coeff_head(h).squeeze(-1)  # (N, P)

        # Bias from mean-pooled representation
        bias = self.bias_head(h_pooled).squeeze(-1)  # (N,)

        # Order prediction
        p_logits = self.p_head(h_pooled)  # (N, n_classes)
        p_hard = torch.argmax(p_logits, dim=-1)  # (N,)

        # Forced linear AR prediction
        x_flat = x_seq.squeeze(-1)  # (N, P)
        x_hat = (coeffs * x_flat).sum(dim=1) + bias  # (N,)

        return coeffs, p_logits, p_hard, x_hat
