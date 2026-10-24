"""Small 1D recorded segments, loaded locally without a network request."""

from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np

NAMES = ("monkey_lfp", "mouse_lfp")


@dataclass(frozen=True)
class Recording:
    signal: np.ndarray
    sampling_rate_hz: float
    name: str

    @property
    def duration_seconds(self):
        return len(self.signal) / self.sampling_rate_hz


def load(name):
    """Load one recorded 1D array and its sampling rate.

    ``monkey_lfp`` is 20 seconds at 500 Hz; ``mouse_lfp`` is five seconds
    at 20 kHz. These are raw tutorial inputs, not processed substitutes or
    complete datasets. The arrays contain no animal or channel identifiers.
    """
    if name not in NAMES:
        raise ValueError(f"Unknown recording {name!r}; choose from {NAMES}")
    with as_file(files(__package__).joinpath(f"{name}.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return Recording(data["signal"], float(data["sampling_rate_hz"]), name)
