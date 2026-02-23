import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_confusion_matrix(model, val_loader, device, p_min=2, p_max=6, ax=None):
    """Plot a normalised confusion matrix for AR-order predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with the standard forward signature.
    val_loader : DataLoader
        Validation data yielding ``(x_batch, p_batch)`` tuples.
    device : torch.device
        Device on which to run inference.
    p_min : int, optional
        Minimum AR order corresponding to class 0. Default is 2.
    p_max : int, optional
        Maximum AR order. Default is 6.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If *None*, a new figure is created.

    Returns
    -------
    cm : np.ndarray
        Row-normalised confusion matrix of shape
        ``(n_classes, n_classes)``.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    model.eval()
    all_p_pred = []
    all_p_true = []
    
    with torch.no_grad():
        for x_batch, p_batch in val_loader:
            x_batch = x_batch.to(device)
            _, _, p_hard, _ = model(x_batch)
            all_p_pred.extend(p_hard.cpu().numpy())
            all_p_true.extend(p_batch.cpu().numpy())
    
    n_classes = p_max - p_min + 1  # 5 classes for p=2,3,4,5,6
    cm = confusion_matrix(all_p_true, all_p_pred) / np.sum(confusion_matrix(all_p_true, all_p_pred), axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(cm, display_labels=[f'p={i+p_min}' for i in range(n_classes)])
    disp.plot(ax=ax, cmap='magma', colorbar=False)
    return cm

def plot_history(history, model=None, val_loader=None, device=None, p_min=2, p_max=6):
    """Plot training and validation metrics over epochs.

    Produces a 2×6 grid of subplots showing per-epoch loss
    components and accuracy for both training and validation.
    If a *model* and *val_loader* are supplied, the final subplot
    displays a confusion matrix.

    Parameters
    ----------
    history : dict
        Training history as returned by :func:`train_loop`.
    model : torch.nn.Module or None, optional
        Trained model for generating the confusion matrix.
    val_loader : DataLoader or None, optional
        Validation data for the confusion matrix.
    device : torch.device or None, optional
        Compute device for confusion matrix inference.
    p_min : int, optional
        Minimum AR order for confusion matrix labels. Default is 2.
    p_max : int, optional
        Maximum AR order for confusion matrix labels. Default is 6.
    """
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    
    metrics = ['loss', 'p', 'ar', 'energy', 'smooth', 'p_acc']
    titles = ['Total Loss', 'P Loss (CE)', 'AR Loss', 'Energy Loss', 'Smooth Loss', 'P Accuracy']
    
    # Train metrics (row 0)     
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[0, i].plot(history[f'train_{metric}'], "k")
        axes[0, i].set_title(f'Train {title}')
        axes[0, i].set_xlabel('Epoch')
    
    # Val metrics (row 1, first 5)
    for i, (metric, title) in enumerate(zip(metrics[:5], titles[:5])):
        axes[1, i].plot(history[f'val_{metric}'], "k")
        axes[1, i].set_title(f'Val {title}')
        axes[1, i].set_xlabel('Epoch')
    
    # Confusion matrix (row 1, last col)
    if model is not None and val_loader is not None and device is not None:
        plot_confusion_matrix(model, val_loader, device, p_min=p_min, p_max=p_max, ax=axes[1, 5])
    else:
        axes[1, 5].plot(history['val_p_acc'])
        axes[1, 5].set_title('Val P Accuracy')
        axes[1, 5].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()   

def plot_coefficients_by_p(model, X, coeffs_true, p_true, device, p_max=6, p_min=2, title=""):
    """Visualise predicted vs. true AR coefficients for each order class.

    Creates one subplot per AR-order class showing the true
    coefficient trajectories (black) overlaid with the model
    predictions (dark red) for one sample per class.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X : array-like or torch.Tensor
        Input time series of shape ``(N, T)``.
    coeffs_true : np.ndarray
        Ground-truth coefficients of shape ``(N, T, max_ar_order)``.
    p_true : array-like or torch.Tensor
        0-indexed order class labels of shape ``(N,)``.
    device : torch.device
        Compute device.
    p_max : int, optional
        Maximum AR order. Default is 6.
    p_min : int, optional
        Minimum AR order. Default is 2.
    title : str, optional
        Overall figure title.
    """
    model.eval()
    
    n_classes = p_max - p_min + 1  # 5 classes for p=2,3,4,5,6
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    
    with torch.no_grad():
        for p_idx in range(n_classes):
            ax = axes[p_idx]
            p_actual = p_idx + p_min  # True AR order: 2,3,4,5,6
            
            mask = (p_true == p_idx).numpy() if torch.is_tensor(p_true) else (p_true == p_idx)
            if not mask.any():
                ax.set_title(f'p={p_actual} (no example)')
                continue
            
            idx = np.where(mask)[0][0]
            
            if torch.is_tensor(X):
                x = X[idx:idx+1].clone().detach().to(device)
            else:
                x = torch.from_numpy(X[idx:idx+1]).float().to(device)
            
            coeffs_pred, _, p_hard, _ = model(x)
            coeffs_pred = coeffs_pred.cpu().numpy()[0]  # (600, max_ar_order)
            p_pred = p_hard.item() + p_min  # Predicted AR order
            
            # Plot true coefficients (p_actual of them)
            for k in range(p_actual):
                label = 'True' if k == 0 else None
                ax.plot(coeffs_true[idx, :, k], 'k-', alpha=0.7, label=label)
            
            # Plot predicted coefficients (p_pred of them)
            for k in range(p_pred):
                label = 'Pred' if k == 0 else None
                ax.plot(coeffs_pred[:, k], color='darkred', alpha=0.7, label=label)
            
            ax.set_title(f'True p={p_actual}, Pred p={p_pred}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Coefficient')
            ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_coefficients_by_p_cutoff(model, X, coeffs_true, p_true, device, p_max=6, p_min=2, cutoff=500, title=""):
    """Visualise predicted vs. true AR coefficients, truncated at *cutoff*.

    Identical to :func:`plot_coefficients_by_p` but only plots the first
    *cutoff* time-steps so that late-window artefacts are excluded.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X : array-like or torch.Tensor
        Input time series of shape ``(N, T)``.
    coeffs_true : np.ndarray
        Ground-truth coefficients of shape ``(N, T, max_ar_order)``.
    p_true : array-like or torch.Tensor
        0-indexed order class labels of shape ``(N,)``.
    device : torch.device
        Compute device.
    p_max : int, optional
        Maximum AR order. Default is 6.
    p_min : int, optional
        Minimum AR order. Default is 2.
    cutoff : int, optional
        Number of initial time-steps to display. Default is 500.
    title : str, optional
        Overall figure title.
    """
    model.eval()

    n_classes = p_max - p_min + 1
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))

    with torch.no_grad():
        for p_idx in range(n_classes):
            ax = axes[p_idx]
            p_actual = p_idx + p_min

            mask = (p_true == p_idx).numpy() if torch.is_tensor(p_true) else (p_true == p_idx)
            if not mask.any():
                ax.set_title(f'p={p_actual} (no example)')
                continue

            idx = np.where(mask)[0][0]

            if torch.is_tensor(X):
                x = X[idx:idx+1].clone().detach().to(device)
            else:
                x = torch.from_numpy(X[idx:idx+1]).float().to(device)

            coeffs_pred, _, p_hard, _ = model(x)
            coeffs_pred = coeffs_pred.cpu().numpy()[0]  # (T, max_ar_order)
            p_pred = p_hard.item() + p_min

            # Truncate to cutoff
            t = min(cutoff, coeffs_true.shape[1])

            for k in range(p_actual):
                label = 'True' if k == 0 else None
                ax.plot(coeffs_true[idx, :t, k], 'k-', alpha=0.7, label=label)

            for k in range(p_pred):
                label = 'Pred' if k == 0 else None
                ax.plot(coeffs_pred[:t, k], color='darkred', alpha=0.7, label=label)

            ax.set_xlim(0, t)
            ax.set_title(f'True p={p_actual}, Pred p={p_pred}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Coefficient')
            ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def plot_tvar_sample(x, coeffs, p, W=40, P0=0.5):
    """Plot a single TVAR sample: coefficients, signal, and local power.

    Produces a three-panel figure showing (top to bottom) the
    time-varying AR coefficients, the generated signal, and the
    sliding-window power with the constraint threshold.

    Parameters
    ----------
    x : np.ndarray
        Time series of shape ``(T,)``.
    coeffs : np.ndarray
        AR coefficient array of shape ``(T, p)``.
    p : int
        AR order (number of coefficient tracks to plot).
    W : int, optional
        Sliding-window size for power computation. Default is 40.
    P0 : float, optional
        Power constraint threshold (shown as a dashed line).
        Default is 0.5.
    """
    T = len(x)
    
    # Compute sliding window power
    power = np.zeros(T)
    for i in range(T):
        start = max(0, i - W + 1)
        power[i] = np.mean(x[start:i + 1] ** 2)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    
    # Coefficients
    for k in range(p):
        axes[0].plot(coeffs[:, k], label=f'a{k+1}')
    axes[0].set_ylabel('Coefficients')
    axes[0].legend(loc='upper right')
    
    # Signal
    axes[1].plot(x)
    axes[1].set_ylabel('Amplitude')
    
    # Power
    axes[2].plot(power, color='tab:orange')
    axes[2].axhline(P0, color='k', linestyle='--')
    axes[2].set_ylabel('Power')
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()