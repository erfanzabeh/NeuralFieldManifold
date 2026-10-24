"""Joint order selection and time-varying AR coefficient estimation."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .._validation import positive_integer


@dataclass(frozen=True)
class DynamicsOutput:
    """Predicted dynamics for a batch of input sequences.

    Attributes
    ----------
    coefficients : torch.Tensor, shape (batch, time, max_order)
        Lag-1-first coefficients, masked beyond each selected order.
    order_logits : torch.Tensor, shape (batch, n_orders)
        Unnormalized order-selection scores.
    order : torch.Tensor, shape (batch,)
        Selected AR order for each sequence.
    prediction : torch.Tensor, shape (batch, time)
        AR reconstruction using observed lag history. Missing history at
        the start of the sequence is zero-padded.
    """

    coefficients: torch.Tensor
    order_logits: torch.Tensor
    order: torch.Tensor
    prediction: torch.Tensor


class DeepLagField(nn.Module):
    """Jointly estimate AR order and time-varying coefficients.

    Parameters
    ----------
    sequence_length : int
        Number of time samples per input sequence.
    orders : tuple of int
        Increasing candidate AR orders, each smaller than sequence_length.
    hidden_size : int
        Hidden width of the order and dynamics multilayer perceptrons.

    Notes
    -----
    The order network selects a lag mask that conditions the dynamics network.
    Coefficients beyond the selected order are zeroed. Training uses hard
    straight-through Gumbel sampling; evaluation uses argmax unless
    sample_order=True is passed to forward.
    """

    def __init__(self, *, sequence_length=600, orders=(2, 3, 4, 5, 6), hidden_size=128):
        super().__init__()
        self.sequence_length = positive_integer(sequence_length, "sequence_length")
        self.hidden_size = positive_integer(hidden_size, "hidden_size")
        orders = tuple(positive_integer(p, "order") for p in orders)
        if (
            not orders
            or tuple(sorted(set(orders))) != orders
            or max(orders) >= sequence_length
        ):
            raise ValueError(
                "orders must be unique, increasing, and smaller than sequence_length"
            )
        self.orders = orders
        self.max_order = max(orders)
        self.order_block = nn.Sequential(
            nn.Linear(sequence_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(orders)),
        )
        self.dynamics_block = nn.Sequential(
            nn.Linear(sequence_length + len(orders), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, sequence_length * self.max_order),
        )

    def forward(self, x, *, temperature=1.0, sample_order=None):
        """Estimate coefficients, select orders, and reconstruct each sequence.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, sequence_length)
            Floating-point input sequences.
        temperature : float
            Positive temperature for Gumbel sampling.
        sample_order : bool or None
            Sample orders if True, or use argmax if False. None samples
            in training mode and uses argmax in evaluation mode.

        Returns
        -------
        DynamicsOutput
            Coefficients, order logits, selected orders, and predictions.
        """
        if (
            x.ndim != 2
            or x.shape[1] != self.sequence_length
            or not x.is_floating_point()
        ):
            raise ValueError(
                "Expected a floating tensor of shape (batch, sequence_length)"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = self.order_block(x)
        if sample_order is None:
            sample_order = self.training
        if sample_order:
            selection = F.gumbel_softmax(logits, tau=temperature, hard=True)
        else:
            selection = F.one_hot(logits.argmax(-1), len(self.orders)).to(x.dtype)
        indices = selection.argmax(-1)
        orders = torch.as_tensor(self.orders, device=x.device)[indices]
        active = (
            torch.arange(self.max_order, device=x.device)[None, :] < orders[:, None]
        ).to(x.dtype)
        raw = self.dynamics_block(torch.cat([x, selection], dim=1))
        coefficients = (
            raw.reshape(len(x), self.sequence_length, self.max_order)
            * active[:, None, :]
        )
        lagged = x.new_zeros(len(x), self.sequence_length, self.max_order)
        for k in range(self.max_order):
            lagged[:, k + 1 :, k] = x[:, : self.sequence_length - k - 1]
        return DynamicsOutput(
            coefficients, logits, orders, (coefficients * lagged).sum(-1)
        )

    def save(self, path):
        """Save architecture settings and weights, without optimizer state."""
        torch.save(
            {
                "config": dict(
                    sequence_length=self.sequence_length,
                    orders=self.orders,
                    hidden_size=self.hidden_size,
                ),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load a package checkpoint in deterministic evaluation mode."""
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(**payload["config"]).to(device)
        model.load_state_dict(payload["state_dict"])
        return model.eval()
