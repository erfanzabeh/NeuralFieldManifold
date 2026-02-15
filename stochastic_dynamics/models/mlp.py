import torch
import torch.nn as nn


class ARMLP(nn.Module):
    """Simple MLP baseline for time-varying AR coefficient estimation.

    Maps an input sequence directly to ``seq_len × max_ar_order``
    coefficient values through a three-layer feed-forward network.
    No AR-order classification is performed.
    """

    def __init__(self, seq_len=600, n_classes=5, max_ar_order=6, hidden_dim=128):
        """Initialise the AR-MLP model.

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
        hidden_dim : int, optional
            Hidden-layer width. Default is 128.
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.max_ar_order = max_ar_order
        
        # Simple MLP: input -> hidden -> hidden -> coefficients
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * max_ar_order)
        )
    
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
        
        # MLP outputs all coefficients directly
        coeffs = self.mlp(x).view(N, self.seq_len, self.max_ar_order)  # (N, 600, 6)
        
        # No order prediction — return zeros for compatibility
        p_logits = torch.zeros(N, self.n_classes, device=device)
        p_hard = torch.zeros(N, dtype=torch.long, device=device)
        
        # AR reconstruction: x_hat[t] = sum_k coeffs[t,k] * x[t-k-1]
        x_lagged = torch.zeros(N, self.seq_len, self.max_ar_order, device=device)
        for k in range(self.max_ar_order):
            x_lagged[:, k+1:, k] = x[:, :self.seq_len - k - 1]
        
        x_hat = (coeffs * x_lagged).sum(dim=-1)  # (N, 600)
        
        return coeffs, p_logits, p_hard, x_hat