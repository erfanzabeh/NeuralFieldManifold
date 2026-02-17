import torch
import torch.nn as nn


class AnalyticalAR(nn.Module):
    """Non-learned AR baseline that solves least-squares per sample.

    This module has **no trainable parameters**. For each input sequence it
    builds the lag matrix, solves the normal equations via
    ``torch.linalg.lstsq``, and returns fixed (time-invariant) AR
    coefficients together with the reconstructed signal.
    """

    def __init__(self, seq_len=600, n_classes=5, max_ar_order=6):
        """Initialise the analytical AR baseline.

        Parameters
        ----------
        seq_len : int, optional
            Length of each input time series. Default is 600.
        n_classes : int, optional
            Number of AR-order classes (kept for API compatibility).
            Default is 5.
        max_ar_order : int, optional
            Maximum autoregressive order used for the lag matrix.
            Default is 6.
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.max_ar_order = max_ar_order
    
    def forward(self, x, temperature=1.0):
        """Run the analytical AR fit on a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(N, seq_len)``.
        temperature : float, optional
            Unused; kept for API compatibility with learned models.

        Returns
        -------
        coeffs : torch.Tensor
            Time-invariant AR coefficients broadcast to shape
            ``(N, seq_len, max_ar_order)``.  The first *max_ar_order*
            time steps are zeroed out.
        p_logits : torch.Tensor
            Zeros of shape ``(N, n_classes)`` (no order prediction).
        p_hard : torch.Tensor
            Zeros of shape ``(N,)`` (no order prediction).
        x_hat : torch.Tensor
            Reconstructed signal of shape ``(N, seq_len)``.
        """
        N = x.shape[0]
        device = x.device
        p = self.max_ar_order
        
        # Build lag matrix for each sample
        # X[:, t, k] = x[:, t+p-1-k] for t in [0, seq_len-p), k in [0, p)
        # y[:, t] = x[:, t+p]
        
        T_out = self.seq_len - p  # number of valid prediction points
        
        X_lag = torch.zeros(N, T_out, p, device=device)
        for k in range(p):
            X_lag[:, :, k] = x[:, p - 1 - k : self.seq_len - 1 - k]
        
        y = x[:, p:]  # (N, T_out)
        
        # Add intercept column: (N, T_out, p+1)
        ones = torch.ones(N, T_out, 1, device=device)
        D = torch.cat([ones, X_lag], dim=-1)  # (N, T_out, p+1)
        
        # Solve least squares per sample: D @ w = y
        # torch.linalg.lstsq expects (N, T_out, p+1) and (N, T_out, 1)
        w = torch.linalg.lstsq(D, y.unsqueeze(-1)).solution  # (N, p+1, 1)
        w = w.squeeze(-1)  # (N, p+1) -> [c, a1, a2, ..., ap]
        
        # Predict on the valid portion
        y_hat_valid = (D @ w.unsqueeze(-1)).squeeze(-1)  # (N, T_out)
        
        # Build full x_hat with zeros for first p timesteps
        x_hat = torch.zeros(N, self.seq_len, device=device)
        x_hat[:, p:] = y_hat_valid
        
        # Build coeffs: (N, seq_len, max_ar_order)
        # Coefficients are constant across time (same w for all t)
        # w[:, 1:] are the AR coefficients (excluding intercept)
        ar_coeffs = w[:, 1:]  # (N, p)
        coeffs = ar_coeffs.unsqueeze(1).expand(N, self.seq_len, p).clone()
        
        # Zero out first p timesteps to match x_hat
        coeffs[:, :p, :] = 0
        
        # No order prediction
        p_logits = torch.zeros(N, self.n_classes, device=device)
        p_hard = torch.zeros(N, dtype=torch.long, device=device)
        
        return coeffs, p_logits, p_hard, x_hat