"""Recreate recorded-data tutorial calculations through the public API."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import neuralfieldmanifold as nfm

SETTINGS = {
    "monkey": dict(
        sample="monkey_lfp", high=50, order=4, train_fraction=0.9, refresh=5, tau=30
    ),
    "mouse": dict(
        sample="mouse_lfp", high=50, order=3, train_fraction=0.1, refresh=10, tau=10
    ),
    "mouse_lfp": dict(
        sample="mouse_lfp", high=20, order=3, train_fraction=0.1, refresh=10, tau=10
    ),
}


def calculate(name):
    """Return the named numerical outputs; no files or plots are created."""
    settings = SETTINGS[name]
    recording = nfm.datasets.load(settings["sample"])
    config = nfm.preprocessing.PreprocessConfig(
        notch_hz=50,
        band_hz=(1, settings["high"]),
        output_rate_hz=500,
        envelope_band_hz=(1, 80),
        envelope_lowpass_hz=3,
    )
    processed = nfm.preprocessing.process(
        recording.signal, sampling_rate_hz=recording.sampling_rate_hz, config=config
    )
    x = processed.signal
    order = settings["order"]
    count = int(settings["train_fraction"] * (len(x) - order))
    fit = nfm.models.AR(order=order).fit(x[: count + order])
    observed = x[order + count :]
    predicted = fit.forecast(
        x,
        start_sample=order + count,
        n_samples=len(observed),
        refresh_samples=settings["refresh"],
    )
    metrics = nfm.models.prediction_metrics(observed, predicted)
    full = np.r_[
        x[:order],
        fit.forecast(
            x,
            start_sample=order,
            n_samples=len(x) - order,
            refresh_samples=settings["refresh"],
        ),
    ]
    matched = predicted - predicted.mean()
    matched = matched * (observed.std() / (matched.std() + 1e-12)) + observed.mean()
    cloud = nfm.embedding.delay(
        observed.astype(np.float32), dimension=3, tau_samples=settings["tau"]
    )
    predicted_cloud = nfm.embedding.delay(
        matched.astype(np.float32), dimension=3, tau_samples=settings["tau"]
    )
    fft = min(4096, len(x))
    spectrum = nfm.spectral.welch(
        x,
        sampling_rate_hz=500,
        n_fft=fft,
        overlap_samples=fft // 2,
        frequency_range_hz=(0.5, 200),
    )
    order_mse = []
    for p in range(2, 7):
        n_train = int(0.9 * (len(x) - p))
        model = nfm.models.AR(order=p).fit(x[: n_train + p])
        predictions = model.predict(x)[n_train:]
        order_mse.append(
            nfm.models.prediction_metrics(x[p + n_train :], predictions).mse
        )
    return dict(
        raw=recording.signal,
        processed=x,
        bandpassed=processed.bandpassed,
        weights=np.r_[fit.intercept, fit.coefficients],
        observed=observed,
        predicted=predicted,
        full_prediction=full,
        matched_prediction=matched,
        cloud=cloud,
        predicted_cloud=predicted_cloud,
        psd=spectrum.density,
        frequencies=spectrum.frequencies_hz,
        metrics=np.array([metrics.mse, metrics.mae, metrics.correlation]),
        order_mse=np.asarray(order_mse),
    )


def figure(name, result):
    """Plot the recording, spectrum, prediction, and delay geometry."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    time = np.arange(len(result["processed"])) / 500
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(time, result["processed"], color="black", lw=0.5)
    ax.set(title="Processed signal", xlabel="Time (s)", ylabel="Signal")
    ax = fig.add_subplot(2, 3, 2)
    ax.loglog(result["frequencies"], result["psd"], color="black")
    ax.set(
        title="Power spectrum",
        xlabel="Frequency (Hz)",
        ylabel="PSD (signal units squared/Hz)",
    )
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(range(2, 7), result["order_mse"], "o-", color="black")
    ax.set(title="Held-out one-step error", xlabel="AR order", ylabel="MSE")
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(result["observed"], color="black", lw=0.7, label="Observed")
    ax.plot(result["predicted"], color="darkred", lw=0.6, label="Predicted")
    ax.set(title="Hybrid prediction", xlabel="Sample", ylabel="Signal")
    ax.legend()
    ax = fig.add_subplot(2, 3, 5)
    ax.bar(np.arange(1, len(result["weights"])), result["weights"][1:], color="darkred")
    ax.set(title="AR coefficients", xlabel="Lag", ylabel="Coefficient")
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    nfm.plotting.embedding(result["cloud"], ax=ax)
    nfm.plotting.embedding(result["predicted_cloud"], ax=ax, color="darkred")
    ax.set_title("Delay embedding")
    fig.suptitle(name.replace("_", " "))
    return fig


def run_one(task):
    name, directory = task
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    result = calculate(name)
    np.savez_compressed(directory / f"{name}.npz", **result)
    fig = figure(name, result)
    fig.savefig(directory / f"{name}.png", dpi=160)
    plt.close(fig)
    return dict(
        tutorial=name,
        mse=float(result["metrics"][0]),
        correlation=float(result["metrics"][2]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/recordings"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(3, max(1, args.workers))) as pool:
        print(
            json.dumps(
                list(
                    pool.map(run_one, [(name, str(args.output)) for name in SETTINGS])
                ),
                indent=2,
            )
        )
