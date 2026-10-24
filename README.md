# Neural Field Manifold

NumPy and PyTorch tools for autoregressive dynamics and delay-coordinate geometry.

```python
import neuralfieldmanifold as nfm
```

Documentation: [User Guide and API](https://erfanzabeh.github.io/NeuralFieldManifold/)

## Installation

```sh
python -m pip install "neuralfieldmanifold @ git+https://github.com/erfanzabeh/NeuralFieldManifold.git@main"
```

The numerical package depends on NumPy, SciPy, and PyTorch. Plotting and
notebook dependencies are optional. There is no JAX dependency. PyPI
publication is a separate release step; the command above installs from GitHub.

For the tutorials and local development:

```sh
git clone --depth 1 https://github.com/erfanzabeh/NeuralFieldManifold.git
cd NeuralFieldManifold
python -m pip install ".[tutorials]"
```

## Recorded Signal

```python
recording = nfm.datasets.load("monkey_lfp")
config = nfm.preprocessing.PreprocessConfig(
    notch_hz=50,
    band_hz=(1, 50),
    envelope_band_hz=(1, 80),
    envelope_lowpass_hz=3,
)
processed = nfm.preprocessing.process(
    recording.signal,
    sampling_rate_hz=recording.sampling_rate_hz,
    config=config,
)
cloud = nfm.embedding.delay(
    processed.signal, dimension=3, tau_samples=30,
)
fit = nfm.geometry.fit_torus(cloud)
print(fit.score, fit.error)
```

The bundled data are small 1D raw segments: 20 seconds of monkey LFP at
500 Hz and five seconds of mouse LFP at 20 kHz. They are not full datasets.
No animal or channel identifiers are exposed in the sample metadata.

## Public API

| Module | Purpose |
|---|---|
| `datasets` | Load small recorded 1D signals and sampling rates |
| `preprocessing` | Explicit filtering, decimation, envelope normalization |
| `embedding` | Delay coordinates, lag matrices, sliding windows |
| `models` | OLS AR, threshold AR, DeepLagField |
| `geometry` | Elliptical ring and torus fits with named outputs |
| `simulation` | Lorenz, AR/TVAR, coefficient schedules, noise, OU, VAR |
| `training` | Dataset generation, losses, training, evaluation |
| `spectral` | Explicit Welch power spectra |
| `plotting` | Optional Matplotlib and rotatable Plotly figures |

Filtering, normalization, delay selection, and variance matching are never
silently performed by model or embedding calls. Fit scores measure geometric
containment, not topological recovery. Persistent homology is not part of this
package's tutorial scope.

## Reproductions

Notebooks using the public API are under `docs/tutorials/`, with saved outputs
also available in the [online documentation](https://erfanzabeh.github.io/NeuralFieldManifold/).
The [original tutorials](https://github.com/erfanzabeh/NeuralFieldManifold/tree/060ea3c60/docs/tutorials)
and research analyses remain accessible in repository history.

```sh
python reproductions/recordings.py --output results/recordings --workers 3
```

The two mouse processing recipes deliberately retain different passbands:
1--50 Hz and 1--20 Hz. Reproduction also retains the original training split,
hybrid-prediction refresh, and delay values. Float32 casting is explicit where
the original JAX embedding used float32.

NumPy and JAX do not share seeded random streams. Stochastic replay uses fixed
inputs; a NumPy seed is not advertised as reproducing JAX's exact samples.
DeepLagField evaluation is deterministic by default. Set `sample_order=True`
to reproduce stochastic Gumbel order selection during a reference comparison.

## Local Checks

Tests are manual and are not wired into pushes or documentation builds.

```sh
python -m unittest discover -s tests -v
```

The tests compare frozen tutorial arrays, AR predictions, embeddings, PSDs,
geometric fit metrics, neural-network outputs, losses, and gradients. They do
not rerun the full training benchmark. Build documentation separately:

```sh
python -m pip install ".[docs]"
python -m sphinx -b html docs docs/_build/html
```
