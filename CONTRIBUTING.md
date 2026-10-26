# Contributing

Please report bugs and discuss proposed changes in
[GitHub Issues](https://github.com/erfanzabeh/NeuralFieldManifold/issues).
Include a minimal example, package versions, and the full traceback for bugs.

## Development Setup

```sh
git clone https://github.com/erfanzabeh/NeuralFieldManifold.git
cd NeuralFieldManifold
python -m pip install -e ".[dev,docs,tutorials]"
```

## Tests

Run the focused numerical regression tests locally:

```sh
python -m unittest discover -s tests -v
```

See [tests/README.md](tests/README.md) for coverage and numerical tolerances.
Tests are run manually, not on pushes or during documentation builds.

## Documentation

```sh
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Documentation builds use saved notebook outputs and do not execute notebooks.
Document public functions with NumPy-style docstrings.

Keep changes focused and use short, descriptive commit messages.

## Releases

Erfan Zabeh maintains the package. PyPI project ownership is configured in
PyPI, independently of the author and maintainer metadata.

Before the first release, Erfan should register a pending Trusted Publisher
from his PyPI account with these settings:

| Setting | Value |
|---|---|
| PyPI project | `neuralfieldmanifold` |
| GitHub owner | `erfanzabeh` |
| Repository | `NeuralFieldManifold` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Configure the GitHub `pypi` environment with a required release approver.
Publishing a GitHub release tagged `v1.0.0` builds and checks the distributions,
then uploads them through Trusted Publishing after any required environment
approval. The tag must match the package version. Ordinary pushes only build
the package and documentation; they do not publish to PyPI.
