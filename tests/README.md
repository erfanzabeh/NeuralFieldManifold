# Numerical Regression Tests

Run from the repository root after installing the package:

```sh
python -m unittest discover -s tests -v
```

Fixtures contain reference numerical outputs. Checks
cover the monkey and two distinct mouse processing recipes, AR predictions,
delay coordinates, PSD, both geometric fit models, DeepLagField
outputs/losses/gradients, a single training step, and checkpoint round trips.

Signal arrays and neural calculations are compared exactly. Spectra tolerate
2e-12 relative error between Welch implementations. Geometric fits tolerate
1e-12 for numerical optimizer output. No test reruns the large training
benchmark or the paper's persistent-homology analyses.
