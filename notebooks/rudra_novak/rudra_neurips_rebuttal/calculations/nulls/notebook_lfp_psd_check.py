from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from mne.time_frequency import psd_array_welch


UNIT_DIR = Path(__file__).resolve().parent
REBUTTAL_DIR = UNIT_DIR.parent.parent
PROJECT_DATA_DIR = REBUTTAL_DIR.parent / "data"
DATA_DIR = REBUTTAL_DIR / "data"
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"

FS = 500
GLOBAL_CHAN = 1001
LOCAL_FILE = "NSP8_array16_LFP.mat"
LOCAL_CHAN = 41
N_SAMPLES = 10000


def load_notebook_filtered_lfp() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce MonkeyData.ipynb's `filtered_LFP` without loading all 16 arrays."""
    mat_path = PROJECT_DATA_DIR / LOCAL_FILE
    mat = sio.loadmat(mat_path, variable_names=["lfp"])
    raw = np.asarray(mat["lfp"][:N_SAMPLES, LOCAL_CHAN], dtype=np.float64)
    filtered = signal.filtfilt(*signal.iirnotch(50, 30, FS), raw)
    time = np.arange(len(filtered), dtype=np.float64) / FS
    return filtered, time


def compute_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psd, freqs = psd_array_welch(
        x[None, :],
        sfreq=FS,
        fmin=0.5,
        fmax=200.0,
        n_fft=4096,
        n_overlap=2048,
        average="mean",
        verbose=False,
    )
    return freqs, gaussian_filter1d(psd[0], sigma=2)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    filtered, time = load_notebook_filtered_lfp()
    metadata = {
        "source_notebook": "notebooks/rudra_novak/MonkeyData.ipynb",
        "source_cell": 6,
        "source_mat_file": str(PROJECT_DATA_DIR / LOCAL_FILE),
        "global_channel": GLOBAL_CHAN,
        "local_channel": LOCAL_CHAN,
        "fs": FS,
        "n_samples": N_SAMPLES,
        "preprocessing": "iirnotch(50 Hz, Q=30) only; this is the PSD display trace before the later 1-50 Hz torus bandpass",
    }

    np.savez_compressed(
        DATA_DIR / "monkey_chan1001_notched_raw_20s.npz",
        xs=filtered,
        time=time,
        Fs=np.asarray(FS),
        metadata=json.dumps(metadata, sort_keys=True),
    )

    freqs, psd = compute_psd(filtered)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.loglog(freqs, psd, color="dimgray", lw=2)
    ax.axvspan(1.0, 50.0, color="0.7", alpha=0.45, zorder=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u)")
    ax.set_title("LFP PSD")
    ax.grid(True, ls="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "notebook_style_lfp_psd_check.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)

    print(PLOTS_DIR / "notebook_style_lfp_psd_check.png")
    print(DATA_DIR / "monkey_chan1001_notched_raw_20s.npz")


if __name__ == "__main__":
    main()
