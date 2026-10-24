"""Prediction, coefficient, and AR-order evaluation."""

import numpy as np
import torch


def evaluate(
    model,
    signals,
    *,
    true_coefficients=None,
    true_orders=None,
    batch_size=32,
    device="cpu",
    sample_order=False,
):
    """Evaluate DeepLagField predictions and optional ground-truth targets.

    Prediction MSE excludes the leading maximum-order lag context.
    Coefficient MSE includes all time positions and lag coefficients.
    When true_orders are supplied, returns exact and within-one order
    accuracy, mean absolute order error, and mean relative order error.

    Inputs have shape (batch, sequence_length). Returned predictions,
    coefficients, and selected orders are NumPy arrays. Order selection is
    deterministic unless sample_order=True.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    signals = torch.as_tensor(signals, dtype=torch.float32)
    if (
        signals.ndim != 2
        or signals.shape[1] != model.sequence_length
        or not len(signals)
    ):
        raise ValueError("Expected nonempty (batch, sequence_length) signals")
    model.to(device).eval()
    coefficients, predictions, orders = [], [], []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            output = model(
                signals[start : start + batch_size].to(device),
                sample_order=sample_order,
            )
            coefficients.append(output.coefficients.cpu().numpy())
            predictions.append(output.prediction.cpu().numpy())
            orders.append(output.order.cpu().numpy())
    coefficients, predictions, orders = map(
        np.concatenate, (coefficients, predictions, orders)
    )
    result = {
        "coefficients": coefficients,
        "prediction": predictions,
        "order": orders,
        "prediction_mse": float(
            np.mean(
                (
                    predictions[:, model.max_order :]
                    - signals.numpy()[:, model.max_order :]
                )
                ** 2
            )
        ),
    }
    if true_coefficients is not None:
        truth = np.asarray(true_coefficients)
        if truth.shape != coefficients.shape:
            raise ValueError("true_coefficients must match model output shape")
        result["coefficient_mse"] = float(np.mean((coefficients - truth) ** 2))
    if true_orders is not None:
        truth = np.asarray(true_orders)
        if truth.shape != orders.shape or not np.isin(truth, model.orders).all():
            raise ValueError(
                "true_orders must contain one supported actual order per signal"
            )
        result.update(
            order_accuracy=float(np.mean(orders == truth)),
            within_one_accuracy=float(np.mean(np.abs(orders - truth) <= 1)),
            order_mae=float(np.mean(np.abs(orders - truth))),
            order_relative_error=float(np.mean(np.abs(orders - truth) / truth)),
        )
    return result
