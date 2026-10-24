"""Bundled sample neural recordings and their sampling rates."""

from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np

NAMES = ("monkey_lfp", "mouse_lfp")


@dataclass(frozen=True)
class Recording:
    """One-dimensional raw sample, sampling rate in Hz, and sample name."""

    signal: np.ndarray
    sampling_rate_hz: float
    name: str

    @property
    def duration_seconds(self):
        return len(self.signal) / self.sampling_rate_hz


def load(name):
    """Load a bundled raw neural recording.

    Parameters
    ----------
    name : {"monkey_lfp", "mouse_lfp"}
        Monkey LFP is 20 seconds at 500 Hz. Mouse LFP is five seconds at 20 kHz.

    Returns
    -------
    Recording
        One-dimensional signal, sampling rate in Hz, and sample name.
        The duration is available as duration_seconds.
    """
    if name not in NAMES:
        raise ValueError(f"Unknown recording {name!r}; choose from {NAMES}")
    with as_file(files(__package__).joinpath(f"{name}.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return Recording(data["signal"], float(data["sampling_rate_hz"]), name)
