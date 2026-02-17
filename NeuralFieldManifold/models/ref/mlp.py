import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPTVAR(nn.Module):
    """
    MLP-based Time-Varying AR model with multi-head outputs.

    Predicts AR coefficients, bias, and order from a window of past observations.
    """

    def __init__(
        self, max_ar_order=6, n_classes=5, hidden_dim=128, depth=3, dropout=0.1
    ):
        """
        Args:
            max_ar_order: Maximum AR order (window size / number of lags)
            n_classes: Number of AR order classes for classification head
            hidden_dim: Hidden layer dimension
            depth: Number of hidden layers in backbone
            dropout: Dropout rate
        """
        super().__init__()
        self.max_ar_order = max_ar_order
        self.n_classes = n_classes

        # Backbone: shared feature extractor
        layers = []
        in_dim = max_ar_order
        for _ in range(depth):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.coeff_head = nn.Linear(hidden_dim, max_ar_order)  # AR coefficients
        self.bias_head = nn.Linear(hidden_dim, 1)  # Bias term
        self.p_head = nn.Linear(hidden_dim, n_classes)  # Order classification

    def forward(self, x):
        """
        Args:
            x: (N, max_ar_order) window of past observations [x_{t-1}, ..., x_{t-p}]

        Returns:
            coeffs: (N, max_ar_order) predicted AR coefficients
            p_logits: (N, n_classes) logits for order classification
            p_hard: (N,) predicted order class (argmax)
            x_hat: (N,) predicted next value
        """
        N = x.shape[0]

        # Shared backbone
        h = self.backbone(x)  # (N, hidden_dim)

        # Output heads
        coeffs = self.coeff_head(h)  # (N, max_ar_order)
        bias = self.bias_head(h).squeeze(-1)  # (N,)
        p_logits = self.p_head(h)  # (N, n_classes)

        # Order prediction
        p_hard = torch.argmax(p_logits, dim=-1)  # (N,)

        # Predicted value: linear combination of lags
        x_hat = (coeffs * x).sum(dim=1) + bias  # (N,)

        return coeffs, p_logits, p_hard, x_hat
