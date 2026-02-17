import torch
import torch.nn as nn


class ARTransformer(nn.Module):
    """Transformer encoder baseline for time-varying AR coefficient estimation.

    Projects each scalar time step to a *d_model*-dimensional embedding,
    adds learnable positional encodings, passes through a stack of
    Transformer encoder layers, and projects each position to
    ``max_ar_order`` coefficient values.
    """

    def __init__(self, seq_len=600, n_classes=5, max_ar_order=6, d_model=64, nhead=4, num_layers=2):
        """Initialise the AR-Transformer model.

        Parameters
        ----------
        seq_len : int, optional
            Length of each input time series. Default is 600.
        n_classes : int, optional
            Number of AR-order classes (unused; kept for API
            compatibility). Default is 5.
        max_ar_order : int, optional
            Maximum AR order, i.e. number of coefficient channels.
            Default is 6.
        d_model : int, optional
            Transformer hidden dimension. Default is 64.
        nhead : int, optional
            Number of attention heads. Default is 4.
        num_layers : int, optional
            Number of Transformer encoder layers. Default is 2.
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.max_ar_order = max_ar_order
        
        # Project each timestep (scalar) to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project each timestep embedding to max_ar_order coefficients
        self.output_proj = nn.Linear(d_model, max_ar_order)
    
    def forward(self, x, temperature=1.0):
        """Forward pass: predict time-varying AR coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(N, seq_len)``.
        temperature : float, optional
            Unused; kept for API compatibility. Default is 1.0.

        Returns
        -------
        coeffs : torch.Tensor
            Predicted AR coefficients of shape
            ``(N, seq_len, max_ar_order)``.
        p_logits : torch.Tensor
            Zeros of shape ``(N, n_classes)`` (no order prediction).
        p_hard : torch.Tensor
            Zeros of shape ``(N,)`` (no order prediction).
        x_hat : torch.Tensor
            Reconstructed signal of shape ``(N, seq_len)``.
        """
        N = x.shape[0]
        device = x.device
        
        # (N, 600) -> (N, 600, 1) -> (N, 600, d_model)
        x_embed = self.input_proj(x.unsqueeze(-1))
        x_embed = x_embed + self.pos_embed
        
        # Transformer: (N, 600, d_model) -> (N, 600, d_model)
        x_transformed = self.transformer(x_embed)
        
        # Project to coefficients: (N, 600, d_model) -> (N, 600, 6)
        coeffs = self.output_proj(x_transformed)
        
        # No order prediction — return zeros for compatibility
        p_logits = torch.zeros(N, self.n_classes, device=device)
        p_hard = torch.zeros(N, dtype=torch.long, device=device)
        
        # AR reconstruction
        x_lagged = torch.zeros(N, self.seq_len, self.max_ar_order, device=device)
        for k in range(self.max_ar_order):
            x_lagged[:, k+1:, k] = x[:, :self.seq_len - k - 1]
        
        x_hat = (coeffs * x_lagged).sum(dim=-1)
        
        return coeffs, p_logits, p_hard, x_hat