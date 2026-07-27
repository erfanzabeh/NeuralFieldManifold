#!/usr/bin/env python
"""Convert mouse EEG MAT sessions into clean per-session NumPy files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


UNIT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "NeuralFieldManifold").exists():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(UNIT_DIR)
SOURCE_DIR = REPO_ROOT / "notebooks" / "data" / "EEG_EEG1.1A-B_EMG_EMG.1"
OUT_DIR = UNIT_DIR / "eeg_npy_data"

WINDOW_SEC = 2.0
STATE_NAMES = {0: "wake", 1: "nrem", 2: "rem"}


def session_from_path(path: Path) -> tuple[str, int, int]:
    match = re.search(r"_m(\d+)-(\d+)\.mat$", path.name)
    if not match:
        raise ValueError(f"Could not parse session minute range from {path.name}")
    start_min, end_min = (int(match.group(1)), int(match.group(2)))
    return f"session_m{start_min:04d}_{end_min:04d}", start_min, end_min


def extract_struct_field(struct: np.ndarray, field: str) -> np.ndarray:
    return np.asarray(struct[0, 0][field])


def convert_one(mat_path: Path, force: bool) -> dict[str, object]:
    session_id, start_min, end_min = session_from_path(mat_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    signal_path = OUT_DIR / f"{session_id}_signal.npy"
    time_path = OUT_DIR / f"{session_id}_time.npy"
    state_path = OUT_DIR / f"{session_id}_state.npy"

    mat = sio.loadmat(mat_path)
    eg_dat = mat["egDat"]
    sampling_rate = float(extract_struct_field(eg_dat, "SamplingRate").ravel()[0])
    original_state_len = int(np.asarray(mat["state"]).size)

    if not force and signal_path.exists() and time_path.exists() and state_path.exists():
        signal = np.load(signal_path, mmap_mode="r")
        time = np.load(time_path, mmap_mode="r")
        state_aligned = np.load(state_path, mmap_mode="r")
    else:
        time = extract_struct_field(eg_dat, "Tim").ravel()
        signal = extract_struct_field(eg_dat, "Data").ravel()
        state = np.asarray(mat["state"]).ravel().astype(np.int16)

        window_samples = int(round(WINDOW_SEC * sampling_rate))
        n_windows = len(signal) // window_samples
        if len(state) < n_windows:
            raise ValueError(
                f"{mat_path.name} has {len(state)} labels but {n_windows} signal windows"
            )
        state_aligned = state[:n_windows]

        np.save(signal_path, signal)
        np.save(time_path, time)
        np.save(state_path, state_aligned)

    counts = {name: int(np.sum(np.asarray(state_aligned) == label)) for label, name in STATE_NAMES.items()}
    return {
        "session_id": session_id,
        "source_file": str(mat_path.relative_to(REPO_ROOT)),
        "start_min": start_min,
        "end_min": end_min,
        "duration_min": end_min - start_min,
        "sampling_rate_hz": sampling_rate,
        "signal_samples": int(len(signal)),
        "time_samples": int(len(time)),
        "original_state_len": original_state_len,
        "aligned_state_len": int(len(state_aligned)),
        "state_trimmed": int(original_state_len - len(state_aligned)),
        "wake_count": counts["wake"],
        "nrem_count": counts["nrem"],
        "rem_count": counts["rem"],
        "signal_path": str(signal_path.relative_to(UNIT_DIR)),
        "time_path": str(time_path.relative_to(UNIT_DIR)),
        "state_path": str(state_path.relative_to(UNIT_DIR)),
    }


def run(force: bool) -> None:
    mat_files = sorted(
        SOURCE_DIR.glob("*.mat"),
        key=lambda p: session_from_path(p)[1],
    )
    if len(mat_files) != 24:
        raise RuntimeError(f"Expected 24 MAT files in {SOURCE_DIR}, found {len(mat_files)}")

    rows = [convert_one(path, force=force) for path in mat_files]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "session_manifest.csv", index=False)

    print(f"Converted {len(rows)} sessions into {OUT_DIR.relative_to(REPO_ROOT)}")
    print(manifest[["session_id", "wake_count", "nrem_count", "rem_count", "state_trimmed"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rewrite existing .npy files.")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
