import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLagEmbed(nn.Module):
    """Deep Lag Embedding network for joint AR-order and coefficient estimation.

    The architecture has two blocks:

    * **P-Block** — predicts the AR order class via Gumbel-Softmax, enabling
      discrete order selection that is differentiable during training.
    * **D-Block** — conditioned on the input series and the soft order
      prediction, outputs time-varying AR coefficients that are masked
      according to the predicted order.
    """

    def __init__(self, seq_len=600, n_classes=5, max_ar_order=6, hidden_dim=128):
        """Initialise the DeepLagEmbed model.

        Parameters
        ----------
        seq_len : int, optional
            Length of each input time series. Default is 600.
        n_classes : int, optional
            Number of AR-order classes (e.g., 5 for p ∈ {2, 3, 4, 5, 6}).
            Default is 5.
        max_ar_order : int, optional
            Maximum AR order, i.e. number of coefficient channels.
            Default is 6.
        hidden_dim : int, optional
            Hidden-layer width for both blocks. Default is 128.
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.max_ar_order = max_ar_order
        
        # P-Block: predicts order class (n_classes outputs)
        self.p_block = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # D-Block: (N, seq_len + n_classes) -> (N, seq_len * max_ar_order)
        self.d_block = nn.Sequential(
            nn.Linear(seq_len + n_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * max_ar_order)
        )
    
    def forward(self, x, temperature=1.0):
        """Forward pass: predict AR order and time-varying coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(N, seq_len)``.
        temperature : float, optional
            Temperature for Gumbel-Softmax sampling. Lower values make
            the order selection closer to argmax. Default is 1.0.

        Returns
        -------
        coeffs : torch.Tensor
            Predicted time-varying AR coefficients of shape
            ``(N, seq_len, max_ar_order)``, masked by the predicted order.
        p_logits : torch.Tensor
            Raw logits for the AR-order classifier of shape
            ``(N, n_classes)``.
        p_hard : torch.Tensor
            Hard (argmax) predicted order class indices of shape ``(N,)``.
        x_hat : torch.Tensor
            Reconstructed signal of shape ``(N, seq_len)`` obtained by
            applying the predicted coefficients to the lagged input.
        """
        # x: (N, seq_len)
        N = x.shape[0]
        
        # P-Block: predict class (0 to n_classes-1)
        p_logits = self.p_block(x)  # (N, n_classes)
        p_soft = F.gumbel_softmax(p_logits, tau=temperature, hard=True)  # (N, n_classes)
        p_hard = torch.argmax(p_soft, dim=-1)  # (N,) values in [0, n_classes-1]
        
        # Convert class to AR order: class 0 -> p=2, class 4 -> p=6
        # So we need (p_hard + 2) coefficients active
        # Mask: indices 0...(p_hard+1) should be active
        indices = torch.arange(self.max_ar_order, device=x.device).unsqueeze(0)  # (1, max_ar_order)
        mask = (indices <= p_hard.unsqueeze(1) + 1).float()  # (N, max_ar_order)
        
        # D-Block: concat x and p_soft
        d_input = torch.cat([x, p_soft], dim=1)  # (N, seq_len + n_classes)
        coeffs_raw = self.d_block(d_input)  # (N, seq_len * max_ar_order)
        coeffs_raw = coeffs_raw.view(N, self.seq_len, self.max_ar_order)  # (N, seq_len, max_ar_order)
        
        # Apply mask
        coeffs = coeffs_raw * mask.unsqueeze(1)  # (N, seq_len, max_ar_order)
        
        # Vectorized AR: build lag matrix and do batch matmul
        # x_lagged[:, t, k] = x[:, t - k - 1]
        x_lagged = torch.zeros(N, self.seq_len, self.max_ar_order, device=x.device)
        for k in range(self.max_ar_order):
            x_lagged[:, k+1:, k] = x[:, :self.seq_len - k - 1]
        
        # x_hat[n, t] = sum_k coeffs[n, t, k] * x_lagged[n, t, k]
        x_hat = (coeffs * x_lagged).sum(dim=-1)  # (N, seq_len)
        
        return coeffs, p_logits, p_hard, x_hat
