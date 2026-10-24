"""Reconstruction, order selection, power, and coefficient-smoothness losses."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    """Nonnegative weights for the five components returned by compute_losses."""

    reconstruction: float = 1.0
    order_classification: float = 10.0
    energy: float = 0.1
    smoothness: float = 0.05
    order_penalty: float = 0.0

    def __post_init__(self):
        if any(not 0 <= v < float("inf") for v in self.__dict__.values()):
            raise ValueError("Loss weights must be finite and nonnegative")


def compute_losses(
    output,
    signal,
    *,
    orders,
    true_order=None,
    weights=None,
    target_power=0.5,
    energy_window_samples=40,
):
    """Compute loss components and their weighted sum.

    Reconstruction is prediction MSE after the maximum-order lag context.
    Order classification is cross-entropy against the supplied AR orders.
    Energy penalizes rolling mean-square prediction power relative to
    target_power, after skipping the first energy_window_samples samples.
    Smoothness penalizes squared successive differences of coefficients,
    summed over lags and averaged over batch and time.
    The optional order penalty is the expected index of the candidate order.

    Parameters
    ----------
    output : DynamicsOutput
        Model coefficients, order logits, and signal prediction.
    signal : torch.Tensor, shape (batch, time)
        Target sequences.
    orders : sequence of int
        Candidate AR orders corresponding to the logit columns.
    true_order : torch.Tensor, shape (batch,), or None
        AR order for each sequence. Required when order classification has
        nonzero weight.
    weights : LossWeights or None
        Nonnegative loss weights. None uses LossWeights defaults.
    target_power : float
        Target mean-square prediction amplitude.
    energy_window_samples : int
        Rolling window length for the energy term.

    Returns
    -------
    dict of str to torch.Tensor
        Scalar reconstruction, order_classification, energy, smoothness,
        order_penalty, and total losses.
    """
    weights = LossWeights() if weights is None else weights
    p_max = output.coefficients.shape[-1]
    if (
        signal.ndim != 2
        or signal.shape != output.prediction.shape
        or signal.shape[1] <= p_max
    ):
        raise ValueError(
            "Signal shape must match prediction and exceed the maximum order"
        )
    if (
        not isinstance(energy_window_samples, int)
        or energy_window_samples < 1
        or target_power < 0
    ):
        raise ValueError("Invalid energy settings")
    zero = output.prediction.sum() * 0
    if true_order is None:
        if weights.order_classification:
            raise ValueError(
                "true_order is required when order_classification has nonzero weight"
            )
        classification = zero
    else:
        if true_order.shape != (len(signal),):
            raise ValueError("true_order must have shape (batch,)")
        order_values = torch.as_tensor(orders, device=true_order.device)
        matches = true_order[:, None] == order_values[None, :]
        if not matches.any(dim=1).all():
            raise ValueError("true_order contains an unsupported AR order")
        classification = F.cross_entropy(
            output.order_logits, matches.long().argmax(dim=1)
        )
    reconstruction = F.mse_loss(output.prediction[:, p_max:], signal[:, p_max:])
    difference = output.coefficients[:, 1:] - output.coefficients[:, :-1]
    smoothness = torch.mean(torch.sum(difference**2, dim=2))
    length, window = signal.shape[1], energy_window_samples
    if length <= window:
        energy = zero
    elif length < 2 * window:
        if weights.energy:
            raise ValueError(
                "Energy loss requires sequence_length >= 2 * energy_window_samples"
            )
        energy = zero
    else:
        power = (output.prediction[:, window:].unfold(1, window, 1) ** 2).mean(-1)
        energy = ((power - target_power) ** 2).mean()
    indices = torch.arange(
        len(orders), device=signal.device, dtype=output.order_logits.dtype
    )
    order_penalty = (F.softmax(output.order_logits, dim=-1) * indices).sum(-1).mean()
    values = dict(
        reconstruction=reconstruction,
        order_classification=classification,
        energy=energy,
        smoothness=smoothness,
        order_penalty=order_penalty,
    )
    values["total"] = sum(
        getattr(weights, name) * value for name, value in values.items()
    )
    return values
