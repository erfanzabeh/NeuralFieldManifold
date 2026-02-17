<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzabeh/NeuralFieldManifold/refs/heads/main/docs/_static/logo.jpg" alt="NeuralFieldManifold" width="400">
</p>

<h1 align="center">NeuralFieldManifold</h1>
<p align="center"><em>Recover low-dimensional manifold geometry from noisy, autocorrelated time series</em></p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <!-- TODO: Replace with actual arXiv link -->
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg"></a>
  <a href="https://erfanzabeh.github.io/NeuralFieldManifold/"><img alt="Docs" src="https://img.shields.io/badge/docs-readthedocs-blue.svg"></a>
  <a href="https://github.com/your-org/NeuralFieldManifold/actions"><img alt="CI" src="https://img.shields.io/badge/CI-passing-brightgreen.svg"></a>
</p>

<!-- docs-start -->

---

## Overview

Standard manifold learning methods break down on signals with strong temporal autocorrelation — oscillations, 1/f noise, and nonstationarity corrupt the geometry that dimensionality reduction is meant to reveal.

**NeuralFieldManifold** provides a principled solution grounded in dynamical systems theory: model the signal as a (time-varying) autoregressive process, then exploit the analytical link between oscillatory spectral structure and the topology of delay embeddings. The core theoretical result is that *K* sustained oscillatory modes produce a *K*-torus in the lag-embedded state space — and aperiodic background only adds thickness, not topology.

<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzabeh/NeuralFieldManifold/refs/heads/main/docs/_static/overview.jpg" alt="Conceptual framework" width="85%">
</p>

---

## Physics-Informed Reconstruction of the Manifold

To recover the predicted toroidal geometry from real, nonstationary neural recordings, we introduce **DeepLagField** — a deep learning model that estimates time-varying autoregressive (TVAR) coefficients with physics-informed constraints. It features time-varying AR estimation via a neural network backbone with adaptive order selection and physics-informed losses — bounded energy, temporal smoothness of coefficients, and autoregressive reconstruction error.

<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzabeh/NeuralFieldManifold/refs/heads/main/docs/_static/arch.jpg" alt="Conceptual framework" width="85%">
  <sub>The input local field potential (LFP) signal is processed by two coupled modules. The order block predicts a soft selection over candidate autoregressive orders, producing a sparse mask that determines the effective lag set. Conditioned on this mask, the dynamic block outputs time-varying autoregressive coefficients $\{\phi_k(t)\}$, enabling a locally stationary TVAR representation.</sub>
</p>

---

## Installation

```bash
pip install -e .

# or with dev/docs extras
pip install -e ".[dev,docs]"
```

**Requires:** Python ≥ 3.10, PyTorch, JAX, NumPy, SciPy, scikit-learn.

---

## Quick Start

```python
from NeuralFieldManifold.generators import sinusoid, tvar
from NeuralFieldManifold.embedders import embed
from NeuralFieldManifold.models import DeepLagEmbed

# generate a synthetic time-varying AR signal
coeffs = sinusoid(T=10000, order=4)
x = tvar(coeffs, noise_std=0.1)

# delay embedding
z = embed(x, m=6, tau=15)

# learn time-varying AR coefficients + automatic order selection
import torch
model = DeepLagEmbed(seq_len=600, max_ar_order=6)
coeffs_hat, p_logits, p_hard, x_hat = model(
    torch.tensor(x[:600], dtype=torch.float32).unsqueeze(0)
)
```

## Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzabeh/NeuralFieldManifold/refs/heads/main/docs/_static/monkey.jpg" alt="Monkey LFP manifold" width="100%"><br>
  <sub>Monkey LFP manifold after lag embedding</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzabeh/NeuralFieldManifold/refs/heads/main/docs/_static/mouse1.jpg" alt="Mouse LFP manifold" width="100%"><br>
  <sub>Mouse LFP manifold reconstruction</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzabeh/NeuralFieldManifold/refs/heads/main/docs/_static/mouse2.jpg" alt="Mouse EEG manifold" width="100%"><br>
  <sub>Mouse EEG manifold reconstruction</sub>
</p>


---

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{fallah2026neuralfieldmanifold,
  title     = {{NeuralFieldManifold}: Reconstruction of {LFP} Manifold with Lag Embedding},
  author    = {Fallah, Kasra and Chen, Haoyu Novak and Singha, Rudramani and Kong, Eunji and Turi, Georgo and Losonczy, Attila and Zabeh, Erfan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/XXXX.XXXXX},
}
```

## Contributors 

<a href="https://github.com/erfanzabeh/NeuralFieldManifold/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erfanzabeh/NeuralFieldManifold" />
</a>

---