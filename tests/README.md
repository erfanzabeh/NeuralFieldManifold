# Manual Regression Checks

Run `python -m unittest discover -s tests -v` from an installed candidate.
No CI hook or documentation command runs these tests.

Fixtures record numerical outputs from the frozen source routines. Checks
cover the monkey and two distinct mouse processing recipes, AR predictions,
float32 delay coordinates, PSD, both geometric fit models, DeepLagField
outputs/losses/gradients, a single training step, and checkpoint round trips.

Signal arrays and neural calculations are compared exactly. Spectra tolerate
2e-12 relative error between Welch implementations. Geometric fits tolerate
1e-12 for numerical optimizer output. No test reruns the large training
benchmark or the paper's persistent-homology analyses.
