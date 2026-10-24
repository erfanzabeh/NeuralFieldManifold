<h1 align="center">Neural Field Manifold</h1>
<p align="center"><em>Recover low-dimensional manifold geometry from noisy, autocorrelated time series</em></p>

<p align="center">
  <a href="https://erfanzabeh.github.io/NeuralFieldManifold/citation.html"><img alt="NeurIPS 2026 paper" src="https://img.shields.io/badge/NeurIPS-2026-68478D.svg"></a>
  <a href="https://erfanzabeh.github.io/NeuralFieldManifold/installation.html"><img alt="Python 3.10 or later" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"></a>
  <a href="https://github.com/erfanzabeh/NeuralFieldManifold/actions/workflows/docs.yml"><img alt="Documentation" src="https://img.shields.io/github/actions/workflow/status/erfanzabeh/NeuralFieldManifold/docs.yml?branch=main&label=Documentation&logo=github"></a>
  <a href="https://github.com/erfanzabeh/NeuralFieldManifold/actions/workflows/build.yml"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/erfanzabeh/NeuralFieldManifold/build.yml?branch=main&label=Build&logo=github"></a>
  <a href="https://github.com/erfanzabeh/NeuralFieldManifold/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/erfanzabeh/NeuralFieldManifold"></a>
  <a href="https://github.com/erfanzabeh/NeuralFieldManifold/issues"><img alt="Open issues" src="https://img.shields.io/github/issues/erfanzabeh/NeuralFieldManifold"></a>
  <a href="https://erfanzabeh.github.io/NeuralFieldManifold/installation.html"><img alt="PyPI: XXXX (forthcoming)" src="https://img.shields.io/badge/PyPI-XXXX-3775A9?logo=pypi&logoColor=white"></a>
  <a href="https://erfanzabeh.github.io/NeuralFieldManifold/citation.html"><img alt="arXiv: XXXX (forthcoming)" src="https://img.shields.io/badge/arXiv-XXXX-B31B1B?logo=arxiv&logoColor=white"></a>
</p>

## Overview

**Neural Field Manifold** connects local signal dynamics to the geometry of
delay embeddings. Use it to process neural recordings, fit autoregressive
models, reconstruct delay-coordinate trajectories, and quantify ring and
torus geometry.

The framework models local signals as autoregressive processes and relates
their oscillatory modes to delay-embedding geometry. Under the paper's
assumptions, `K` sustained oscillatory modes correspond to a `K`-torus.
The tutorials examine how noise, nonstationarity, and source mixing affect
that reconstruction, and compare intrinsic dynamics with PCA dimensionality.

The package includes **DeepLagField**, a PyTorch model that jointly estimates
autoregressive order and time-varying coefficients. NumPy and SciPy tools
support classical AR models, synthetic dynamical systems, and geometric fits.

<p align="center">
  <img src="docs/_static/overview.jpg" alt="From oscillatory signals to delay-coordinate geometry" width="90%">
</p>

## Installation

Requires Python 3.10 or later. Install from GitHub:

```sh
python -m pip install "neuralfieldmanifold @ git+https://github.com/erfanzabeh/NeuralFieldManifold.git@main"
```

For plotting and the tutorial notebooks:

```sh
python -m pip install "neuralfieldmanifold[tutorials] @ git+https://github.com/erfanzabeh/NeuralFieldManifold.git@main"
```

## Quick Start

Load a recording, process the signal, and fit a torus to its delay embedding:

```python
import neuralfieldmanifold as nfm

recording = nfm.datasets.load("monkey_lfp")
processed = nfm.preprocessing.process(
    recording.signal,
    sampling_rate_hz=recording.sampling_rate_hz,
    config=nfm.preprocessing.PreprocessConfig(
        notch_hz=50,
        band_hz=(1, 50),
        envelope_band_hz=(1, 80),
        envelope_lowpass_hz=3,
    ),
)
cloud = nfm.embedding.delay(processed.signal, dimension=3, tau_samples=30)
fit = nfm.geometry.fit_torus(cloud)
print(f"Torus score: {fit.score:.3f}; error: {fit.error:.3f}")
```

Bundled samples include 20 seconds of monkey LFP at 500 Hz and five seconds
of mouse LFP at 20 kHz. To use your own recording, supply a one-dimensional
NumPy array and its sampling rate. Processing bands and delays should be
chosen for that recording.

## Physics-Informed Reconstruction of the Manifold

DeepLagField couples an order-selection network to a dynamics network. The
selected order determines which lag coefficients are active; the dynamics
network estimates those coefficients over time. Training combines signal
reconstruction, supervised order selection, target-power regularization, and
temporal smoothness.

<p align="center">
  <img src="docs/_static/arch.jpg" alt="DeepLagField order-selection and dynamics networks" width="90%">
  <br><sub>The order block selects the active autoregressive lags. Conditioned on that selection, the dynamics block estimates time-varying coefficients that reconstruct the signal from its lag history.</sub>
</p>

The [DeepLagField tutorial](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/model.html)
covers dataset generation, training, evaluation, and coefficient recovery.

### Synthetic Dynamics

Generate a time-varying AR signal and pass a window to DeepLagField:

```python
import numpy as np
import torch
import neuralfieldmanifold as nfm

coefficients = nfm.simulation.schedules.sinusoid(
    10_000, 4, np.random.default_rng(0)
)
signal = nfm.simulation.tvar(coefficients, noise_std=0.1, seed=42)
embedding = nfm.embedding.delay(signal, dimension=6, tau_samples=15)

model = nfm.models.DeepLagField(sequence_length=600, orders=(2, 3, 4, 5, 6))
model.eval()
with torch.no_grad():
    output = model(torch.from_numpy(signal[:600]).unsqueeze(0))
```

This constructs an untrained model. See the training tutorial to estimate
weights before interpreting its predicted coefficients or order.

## Tutorials

| Tutorial | Topics |
|---|---|
| [Monkey V1 LFP](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/monkey.html) | Filtering, AR prediction, and delay geometry |
| [Mice Hippocampus EEG](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/mouse.html) | Reconstruction, residual diagnostics, window and delay sweeps |
| [Mice Hippocampus LFP](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/mouse_lfp.html) | Signal processing, prediction, and multichannel comparison |
| [Intrinsic vs. embedding dimensionality](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/intrinsic_extrinsic.html) | PCA, AR roots, colored noise, and nonstationarity |
| [Intrinsic manifold reconstruction](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/simulations.html) | Lorenz dynamics, AR order selection, and threshold AR |
| [DeepLagField](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/model.html) | Schedule families, training, order classification, and coefficient recovery |
| [Volume conduction](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/volume_conduction_manifold.html) | Source mixing, spatial consistency, and real-data comparisons |

The [tutorial library](https://erfanzabeh.github.io/NeuralFieldManifold/tutorials/index.html) also
lists upcoming tutorials on strongly autocorrelated signals and DeepLagField
fine-tuning.

## Package Structure

| Modules | Purpose |
|---|---|
| `datasets` | Sample recordings and sampling rates |
| `preprocessing`, `spectral` | Filtering, envelope normalization, and power spectra |
| `embedding` | Delay coordinates, lag matrices, and sliding windows |
| `models` | Linear AR, threshold AR, and DeepLagField |
| `training` | Synthetic datasets, losses, model training, and evaluation |
| `geometry` | Ring and torus fitting and trajectory comparisons |
| `simulation` | AR and time-varying AR signals, dynamical systems, and noise |
| `plotting` | Trajectories, training histories, and coefficient comparisons |

The [API reference](https://erfanzabeh.github.io/NeuralFieldManifold/api.html)
documents parameters, array shapes, and returned results.

## Gallery

<p align="center">
  <img src="docs/_static/monkey.jpg" alt="Monkey LFP delay geometry and reconstruction" width="100%">
  <br><sub>Monkey LFP reconstruction</sub>
</p>
<p align="center">
  <img src="docs/_static/mouse1.jpg" alt="Mouse neural recording delay geometry and reconstruction" width="100%">
  <br><sub>Mouse neural recording reconstruction</sub>
</p>
<p align="center">
  <img src="docs/_static/mouse2.jpg" alt="Mouse neural recording low-frequency reconstruction" width="100%">
  <br><sub>Low-frequency reconstruction</sub>
</p>

## Documentation and Support

See the [user guide](https://erfanzabeh.github.io/NeuralFieldManifold/getting_started.html)
and [API reference](https://erfanzabeh.github.io/NeuralFieldManifold/api.html).
Report bugs or request features in [GitHub Issues](https://github.com/erfanzabeh/NeuralFieldManifold/issues).
Development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{fallah2026neuralfieldmanifold,
  title     = {{NeuralFieldManifold}: Reconstruction of {LFP} Manifold with Lag Embedding},
  author    = {Fallah, Kasra and Chen, Haoyu Novak and Singha, Rudramani and Kong, Eunji and Turi, Gergely and Losonczy, Attila and Zabeh, Erfan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## Contributors

<a href="https://github.com/erfanzabeh/NeuralFieldManifold/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erfanzabeh/NeuralFieldManifold" alt="Contributors">
</a>

## License

Released under the [MIT license](LICENSE).
