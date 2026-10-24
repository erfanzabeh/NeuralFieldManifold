"""Training is an explicit call; imports and documentation builds never train."""

from dataclasses import dataclass, field

import numpy as np
import torch

from .losses import LossWeights, compute_losses


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weights: LossWeights = field(default_factory=LossWeights)
    target_power: float = 0.5
    energy_window_samples: int = 40
    temperature: float = 1.0
    stochastic_validation: bool = False
    gradient_clip: float = 1.0


def train(
    model,
    train_loader,
    *,
    validation_loader=None,
    config=None,
    device="cpu",
    callback=None,
):
    """Train on ``(signal, actual_order)`` batches and return epoch records.

    Uses Adam, gradient clipping, and ReduceLROnPlateau. No seed, global
    device setting, file write, checkpoint, or plot is selected implicitly.
    Pass ``stochastic_validation=True`` to reproduce Gumbel sampling during
    validation in the original model tutorial. Losses are batch means.
    """
    config = TrainingConfig() if config is None else config
    if config.epochs < 1 or config.learning_rate <= 0 or config.gradient_clip <= 0:
        raise ValueError("epochs, learning_rate, and gradient_clip must be positive")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    history = []
    for epoch in range(config.epochs):
        record = {"epoch": epoch + 1}
        for phase, loader in (
            ("train", train_loader),
            ("validation", validation_loader),
        ):
            if loader is None:
                continue
            training = phase == "train"
            model.train(training)
            values, correct, within_one, count = [], 0, 0, 0
            with torch.set_grad_enabled(training):
                for signal, orders in loader:
                    signal, orders = signal.to(device), orders.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(
                        signal,
                        temperature=config.temperature,
                        sample_order=training or config.stochastic_validation,
                    )
                    losses = compute_losses(
                        output,
                        signal,
                        orders=model.orders,
                        true_order=orders,
                        weights=config.weights,
                        target_power=config.target_power,
                        energy_window_samples=config.energy_window_samples,
                    )
                    if training:
                        losses["total"].backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.gradient_clip
                        )
                        optimizer.step()
                    values.append(
                        {name: float(value.detach()) for name, value in losses.items()}
                    )
                    correct += int((output.order == orders).sum())
                    within_one += int((torch.abs(output.order - orders) <= 1).sum())
                    count += len(signal)
            if not values:
                raise ValueError(f"{phase} loader is empty")
            record[phase] = {
                name: float(np.mean([v[name] for v in values])) for name in values[0]
            }
            record[phase].update(
                order_accuracy=correct / count, within_one_accuracy=within_one / count
            )
        scheduler.step(record.get("validation", record["train"])["total"])
        history.append(record)
        if callback is not None:
            callback(record)
    model.eval()
    return history
