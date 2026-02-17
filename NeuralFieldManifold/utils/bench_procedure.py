import torch
import numpy as np


def bench_loop(model, X_val, coef_val, p_val, device, p_min=2, p_max=6, p_max_order=6, batch_size=32):
    """Evaluate a trained model on validation data and return benchmark metrics.

    Runs the model in inference mode over the validation set and computes
    coefficient MSE, signal reconstruction MSE, per-order accuracy,
    and aggregate order-prediction statistics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with the standard ``(coeffs, p_logits, p_hard, x_hat)``
        forward signature.
    X_val : array-like or torch.Tensor
        Validation time series of shape ``(N, T)``.
    coef_val : np.ndarray
        Ground-truth AR coefficients of shape ``(N, T, max_ar_order)``.
    p_val : array-like or torch.Tensor
        Ground-truth 0-indexed order class labels of shape ``(N,)``.
    device : torch.device
        Device on which to run inference.
    p_min : int, optional
        Minimum AR order corresponding to class index 0. Default is 2.
    p_max : int, optional
        Maximum AR order. Default is 6.
    p_max_order : int, optional
        Number of leading time steps to skip when computing signal MSE.
        Default is 6.
    batch_size : int, optional
        Inference batch size. Default is 32.

    Returns
    -------
    results : dict
        Dictionary with keys ``'coeff_mse'``, ``'signal_mse'``,
        ``'p_mae'``, ``'p_mape'``, per-order accuracies
        (``'p2_acc'``, …, ``'p6_acc'``), and overall ``'p_acc'``.
    """
    model.eval()
    
    n_classes = p_max - p_min + 1
    N = X_val.shape[0]
    all_coeffs_pred = []
    all_p_pred = []
    all_x_hat = []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x_batch = X_val[i:i+batch_size]
            if not torch.is_tensor(x_batch):
                x_batch = torch.tensor(x_batch, dtype=torch.float32)
            x_batch = x_batch.to(device)
            
            coeffs_pred, _, p_hard, x_hat = model(x_batch)
            all_coeffs_pred.append(coeffs_pred.cpu().numpy())
            all_p_pred.append(p_hard.cpu().numpy())
            all_x_hat.append(x_hat.cpu().numpy())
    
    all_coeffs_pred = np.concatenate(all_coeffs_pred, axis=0)
    all_p_pred = np.concatenate(all_p_pred, axis=0)
    all_x_hat = np.concatenate(all_x_hat, axis=0)
    p_true = p_val.numpy() if torch.is_tensor(p_val) else p_val
    X_np = X_val.numpy() if torch.is_tensor(X_val) else X_val
    
    coeff_mse = float(np.mean((all_coeffs_pred - coef_val) ** 2))
    signal_mse = float(np.mean((all_x_hat[:, p_max_order:] - X_np[:, p_max_order:]) ** 2))
    
    # Convert class indices back to actual p values for MAE calculation
    p_true_actual = p_true + p_min
    p_pred_actual = all_p_pred + p_min
    
    # Mean absolute delta p / p (relative error)
    p_mae = float(np.mean(np.abs(p_pred_actual - p_true_actual)))
    p_mape = float(np.mean(np.abs(p_pred_actual - p_true_actual) / p_true_actual))
    
    results = {
        'coeff_mse': coeff_mse, 
        'signal_mse': signal_mse,
        'p_mae': p_mae,        # Mean absolute error in p
        'p_mape': p_mape,      # Mean absolute percentage error |delta_p| / p_true
    }
    for p_idx in range(n_classes):
        p_actual = p_idx + p_min
        mask = (p_true == p_idx)
        if mask.sum() > 0:
            acc = (all_p_pred[mask] == p_true[mask]).mean()
        else:
            acc = 0.0
        results[f'p{p_actual}_acc'] = float(acc)
    
    results['p_acc'] = float((all_p_pred == p_true).mean())
    
    return results