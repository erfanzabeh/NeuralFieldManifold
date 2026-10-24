"""DeepLagField training, evaluation, and synthetic datasets."""

from .losses import LossWeights, compute_losses
from .trainer import TrainingConfig, train
from .evaluation import evaluate
from .datasets import TVARDataset, make_tvar_dataset, split_dataset

__all__ = [
    "LossWeights",
    "compute_losses",
    "TrainingConfig",
    "train",
    "evaluate",
    "TVARDataset",
    "make_tvar_dataset",
    "split_dataset",
]
