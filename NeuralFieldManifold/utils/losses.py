import torch
import torch.nn.functional as F

def loss_p_one(p_logits, p_true):
    """Within-one accuracy for AR-order classification.

    Returns the fraction of predictions whose argmax class index is
    within 1 of the true class index.

    Parameters
    ----------
    p_logits : torch.Tensor
        Raw logits of shape ``(N, n_classes)``.
    p_true : torch.Tensor
        Ground-truth class indices of shape ``(N,)``.

    Returns
    -------
    float
        Fraction in [0, 1]; 1 means every prediction is within 1 of truth.
    """
    p_pred = p_logits.argmax(dim=-1)
    return float((torch.abs(p_pred - p_true) <= 1).float().mean())


def loss_p(p_logits, p_true):
    """Cross-entropy loss for AR-order classification.

    Parameters
    ----------
    p_logits : torch.Tensor
        Raw logits of shape ``(N, n_classes)``.
    p_true : torch.Tensor
        Ground-truth class indices of shape ``(N,)``.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(p_logits, p_true)

def loss_ar(x, x_hat, p_max):
    """Mean squared error between the true and reconstructed signal.

    Only the portion after the first *p_max* time steps is considered
    to avoid the zero-padded region where lagged values are unavailable.

    Parameters
    ----------
    x : torch.Tensor
        True input signal of shape ``(N, T)``.
    x_hat : torch.Tensor
        Reconstructed signal of shape ``(N, T)``.
    p_max : int
        Maximum AR order; the first *p_max* steps are excluded.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    return F.mse_loss(x_hat[:, p_max:], x[:, p_max:])

def loss_energy(x_hat, P0=0.5, W=40):
    """Sliding-window power constraint loss.

    Penalises deviations of the local mean-squared power from the
    target level *P0* by computing the MSE between windowed power and
    *P0* across all positions.

    Parameters
    ----------
    x_hat : torch.Tensor
        Reconstructed signal of shape ``(N, T)``.
    P0 : float, optional
        Target power level. Default is 0.5.
    W : int, optional
        Sliding-window size. Default is 40.

    Returns
    -------
    torch.Tensor
        Scalar energy loss.
    """
    # Use unfold for sliding window — only full windows (no zero-padding bias)
    N, T = x_hat.shape
    if T <= W:
        return torch.tensor(0.0, device=x_hat.device)
    windows = x_hat[:, W:].unfold(1, W, 1)  # (N, T-W, W)
    powers = (windows ** 2).mean(dim=-1)  # (N, T-W)
    return ((powers - P0) ** 2).mean()

def loss_smooth(coeffs):
    """Temporal smoothness penalty on AR coefficients.

    Computes the mean squared first-order difference of the
    coefficient trajectories along the time axis.

    Parameters
    ----------
    coeffs : torch.Tensor
        Predicted AR coefficients of shape ``(N, T, max_ar_order)``.

    Returns
    -------
    torch.Tensor
        Scalar smoothness loss.
    """
    diff = coeffs[:, 1:, :] - coeffs[:, :-1, :]
    return torch.mean(torch.sum(diff ** 2, dim=2))

def loss_order(p_logits):
    """
    Regularizer. Computes expected order index from softmax probabilities and penalizes higher values.
    """
    # p_logits: (N, n_classes) where class 0 -> p=2, class 4 -> p=6
    n_classes = p_logits.shape[1]
    probs = F.softmax(p_logits, dim=-1)  # (N, n_classes)
    indices = torch.arange(n_classes, device=p_logits.device, dtype=p_logits.dtype)  # [0, 1, 2, 3, 4]
    expected_order = (probs * indices).sum(dim=-1)  # (N,) expected class index
    return expected_order.mean()  # Minimize expected order