"""Synthetic time-varying AR datasets and stratified training splits."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..simulation import SimulationRejected, schedules, tvar

FAMILIES = {
    "sinusoid": schedules.sinusoid,
    "fourier": schedules.fourier,
    "quasiperiodic": schedules.quasiperiodic,
    "poly_drift": schedules.polynomial_drift,
    "logistic": schedules.logistic_transition,
    "multi_sigmoid": schedules.multi_sigmoid,
    "gaussian_bumps": schedules.gaussian_bumps,
    "smooth_random": schedules.smooth_random,
}


@dataclass(frozen=True)
class TVARDataset:
    """Synthetic signals, coefficient paths, AR orders, and generation labels.

    Signals have shape (sample, time), coefficients have shape
    (sample, time, coefficient_width), and per-sample labels have shape
    (sample,). Coefficients beyond the AR order are zero-padded.
    """

    signals: np.ndarray
    coefficients: np.ndarray
    orders: np.ndarray
    family_indices: np.ndarray
    family_names: tuple[str, ...]
    seeds: np.ndarray
    noise_std: np.ndarray

    def subset(self, indices):
        return TVARDataset(
            self.signals[indices],
            self.coefficients[indices],
            self.orders[indices],
            self.family_indices[indices],
            self.family_names,
            self.seeds[indices],
            self.noise_std[indices],
        )

    def save(self, path):
        """Save numeric arrays and Unicode labels; no pickled Python objects."""
        np.savez_compressed(
            path,
            signals=self.signals,
            coefficients=self.coefficients,
            orders=self.orders,
            family_indices=self.family_indices,
            family_names=np.asarray(self.family_names),
            seeds=self.seeds,
            noise_std=self.noise_std,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["signals"],
                data["coefficients"],
                data["orders"],
                data["family_indices"],
                tuple(data["family_names"].tolist()),
                data["seeds"],
                data["noise_std"],
            )


def _sample(task):
    family, order, length, width, seed, noise, target, window, max_retries = task
    for attempt in range(max_retries):
        actual_seed = seed + attempt
        coefficients = FAMILIES[family](
            length, order, np.random.default_rng(actual_seed)
        )
        try:
            signal = tvar(
                coefficients,
                noise_std=noise,
                seed=actual_seed + 123,
                burn_in_samples=300,
                target_power=target,
                power_window_samples=window,
            )
        except SimulationRejected:
            continue
        padded = np.zeros((length, width), dtype=np.float32)
        padded[:, :order] = coefficients.astype(np.float32)
        return signal, padded, actual_seed
    raise SimulationRejected(
        f"No accepted {family} order-{order} sample after {max_retries} attempts"
    )


def make_tvar_dataset(
    *,
    n_per_family=10,
    n_samples=600,
    families=None,
    orders=(2, 3, 4, 5, 6),
    coefficient_width=None,
    noise_range=(0.2, 0.5),
    seed=123,
    target_power=0.5,
    power_window_samples=40,
    workers=1,
    max_retries=10000,
):
    """Generate synthetic data balanced by coefficient family and AR order.

    A fixed seed is assigned to each generation task, keeping the dataset
    reproducible for a fixed seed across worker counts.
    """
    families = tuple(FAMILIES if families is None else families)
    orders = tuple(orders)
    width = max(orders) if coefficient_width is None else coefficient_width
    if (
        n_per_family < 1
        or n_samples < 2
        or workers < 1
        or not families
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("Invalid dataset size, families, or worker count")
    if (
        not orders
        or min(orders) < 1
        or width < max(orders)
        or not 0 <= noise_range[0] <= noise_range[1]
    ):
        raise ValueError("Invalid orders, coefficient width, or noise range")
    rng = np.random.default_rng(seed)
    assignments = [
        order
        for i, order in enumerate(orders)
        for _ in range(
            n_per_family // len(orders) + int(i < n_per_family % len(orders))
        )
    ]
    tasks, actual_orders, family_indices, noises = [], [], [], []
    for family_index, family in enumerate(families):
        for order in rng.permutation(assignments):
            sample_seed = int(rng.integers(0, 2**31 - 1))
            noise = float(rng.uniform(*noise_range))
            tasks.append(
                (
                    family,
                    int(order),
                    n_samples,
                    width,
                    sample_seed,
                    noise,
                    target_power,
                    power_window_samples,
                    max_retries,
                )
            )
            actual_orders.append(order)
            family_indices.append(family_index)
            noises.append(noise)
    if workers == 1:
        results = list(map(_sample, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_sample, tasks))
    return TVARDataset(
        np.stack([r[0] for r in results]),
        np.stack([r[1] for r in results]),
        np.asarray(actual_orders),
        np.asarray(family_indices),
        families,
        np.asarray([r[2] for r in results]),
        np.asarray(noises, dtype=np.float32),
    )


def split_dataset(dataset, *, train_fraction=0.8, seed=42):
    """Split a TVAR dataset into training and validation sets, stratified by AR order.

    Requires scikit-learn, available in the tutorials extra.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(dataset.signals))
    train_indices, validation_indices = train_test_split(
        indices, train_size=train_fraction, stratify=dataset.orders, random_state=seed
    )
    return dataset.subset(train_indices), dataset.subset(validation_indices)
