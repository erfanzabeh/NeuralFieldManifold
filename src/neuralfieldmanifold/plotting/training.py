"""Training histories and coefficient-recovery figures."""

import numpy as np


def training_history(history, *, model=None, validation_loader=None, device="cpu"):
    """Plot losses and order accuracy in the tutorial's two-by-six layout.

    Returns the Matplotlib figure and axes. Supplying a model and validation
    loader adds an AR-order confusion matrix in the final panel.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    metrics = [
        "total",
        "order_classification",
        "reconstruction",
        "energy",
        "smoothness",
        "order_accuracy",
    ]
    titles = [
        "Total Loss",
        "P Loss (CE)",
        "AR Loss",
        "Energy Loss",
        "Smooth Loss",
        "P Accuracy",
    ]
    for row, (phase, label) in enumerate((("train", "Train"), ("validation", "Val"))):
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            if (
                row == 1
                and col == 5
                and model is not None
                and validation_loader is not None
            ):
                import torch
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

                observed, predicted = [], []
                model.to(device).eval()
                with torch.no_grad():
                    for signals, orders in validation_loader:
                        output = model(signals.to(device), sample_order=True)
                        observed.extend(orders.detach().cpu().numpy())
                        predicted.extend(output.order.cpu().numpy())
                counts = confusion_matrix(observed, predicted, labels=model.orders)
                totals = counts.sum(axis=1, keepdims=True)
                normalized = np.divide(
                    counts,
                    totals,
                    out=np.zeros_like(counts, dtype=float),
                    where=totals != 0,
                )
                ConfusionMatrixDisplay(
                    normalized, display_labels=[f"p={p}" for p in model.orders]
                ).plot(ax=axes[row, col], cmap="magma", colorbar=False)
            else:
                axes[row, col].plot([record[phase][metric] for record in history], "k")
                axes[row, col].set(title=f"{label} {title}", xlabel="Epoch")
    fig.tight_layout()
    return fig, axes


def coefficients(
    model, signals, true_coefficients, true_orders, *, device="cpu", title=""
):
    """Overlay true and predicted coefficients for one sample of each AR order.

    Signals have shape (sample, time), coefficients (sample, time, lag),
    and true_orders contains actual AR orders. Returns the figure and axes.
    """
    import matplotlib.pyplot as plt
    import torch

    orders = (
        true_orders.detach().cpu().numpy()
        if torch.is_tensor(true_orders)
        else np.asarray(true_orders)
    )
    fig, axes = plt.subplots(
        1, len(model.orders), figsize=(4 * len(model.orders), 4), squeeze=False
    )
    model.to(device).eval()
    with torch.no_grad():
        for ax, order in zip(axes[0], model.orders):
            indices = np.flatnonzero(orders == order)
            if not len(indices):
                ax.set_title(f"p={order} (no example)")
                continue
            index = indices[0]
            x = torch.as_tensor(
                signals[index : index + 1], dtype=torch.float32, device=device
            )
            output = model(x, sample_order=True)
            predicted = output.coefficients[0].cpu().numpy()
            predicted_order = int(output.order[0])
            for lag in range(order):
                ax.plot(
                    true_coefficients[index, :, lag],
                    "k-",
                    alpha=0.7,
                    label="True" if lag == 0 else None,
                )
            for lag in range(predicted_order):
                ax.plot(
                    predicted[:, lag],
                    color="darkred",
                    alpha=0.7,
                    label="Pred" if lag == 0 else None,
                )
            ax.set(
                title=f"True p={order}, Pred p={predicted_order}",
                xlabel="Time",
                ylabel="Coefficient",
            )
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes[0]
