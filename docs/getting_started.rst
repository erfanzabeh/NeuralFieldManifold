Getting Started
===============

Install the local candidate with notebook support::

   python -m pip install ".[tutorials]"

Import the package::

   import neuralfieldmanifold as nfm

Signal conventions
------------------

Single signals are one-dimensional NumPy arrays. Delay coordinates have shape
``(time, coordinate)`` with the newest sample first. Coefficient schedules have
shape ``(time, lag)`` with lag 1 first. DeepLagField inputs have shape
``(batch, time)`` and outputs have named attributes. Orders are actual AR
orders, not zero-based class indices. Sampling rates are in Hz; embedding and
prediction delays are in samples.

Samples
-------

``nfm.datasets.load("monkey_lfp")`` returns a 20-second raw signal at 500 Hz.
``nfm.datasets.load("mouse_lfp")`` returns a five-second raw signal at 20 kHz.
Both return ``signal``, ``sampling_rate_hz``, and ``duration_seconds``.

Processing and prediction
-------------------------

The configurable processing order is notch, bandpass, optional integer IIR
decimation, then optional Hilbert-envelope normalization. Parameters must be
chosen for the signal; there is no automatic common passband or sampling rate.

``ARFit.predict`` is one-step prediction. ``ARFit.forecast`` is recursive;
``refresh_samples=1`` uses observed history each step, larger intervals are
hybrid predictions, and ``None`` is free-running. Explicit variance matching
for display or simulation must not be interpreted as held-out prediction.

DeepLagField
------------

Constructing a model gives randomly initialized weights, not a pretrained
estimator. Training is an explicit call. Evaluation uses deterministic argmax
order selection; ``sample_order=True`` requests stochastic Gumbel sampling.
Checkpoints save architecture settings and model weights.

Geometric fits
--------------

``fit_ring`` fits a planar elliptical annular band. ``fit_torus`` fits a thick
elliptical torus. Score is the contained fraction and error is the average
outside distance. Ring error is in-plane; its MSE also includes the normal
offset. A high containment score alone does not establish topology.

Reproducibility
---------------

Original tutorials are retained unchanged in the reference archive. These
new notebooks use the public API. Deterministic numerical stages are compared
to frozen reference arrays. NumPy random generators replace JAX; identical
integer seeds do not imply identical random draws across those libraries.
Fixed random inputs are used when a direct stochastic comparison is needed.

Documentation builds neither execute notebooks nor run tests.
