Getting Started
===============

This example processes a bundled recording, fits an autoregressive model,
and measures its delay-coordinate geometry. See :doc:`installation` for
installation options.

Load and Process a Signal
-------------------------

Signals are one-dimensional NumPy arrays. Sampling rates are in Hz;
window lengths and delays are in samples.

.. code-block:: python

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
   x = processed.signal

Processing applies notch filtering, bandpass filtering, optional integer
decimation, and optional Hilbert-envelope normalization, in that order.
Choose bands and an output rate appropriate to your recording; a setting
of ``None`` disables the corresponding step.

Bundled samples are ``monkey_lfp`` (20 seconds at 500 Hz) and ``mouse_lfp``
(five seconds at 20 kHz). Your own data can be used in place of
``recording.signal`` by providing its sampling rate.

Fit an AR Model
---------------

.. code-block:: python

   split = int(0.8 * len(x))
   model = nfm.models.AR(order=4).fit(x[:split])
   predicted = model.predict(x[split - model.order:])
   metrics = nfm.models.prediction_metrics(x[split:], predicted)
   print(metrics)

``predict`` uses observed lag history for each one-step prediction.
``forecast`` instead feeds predictions back into the model; its
``refresh_samples`` option periodically replaces that history with observed
samples. Use a temporal train/test split to evaluate prediction on later data.

Embed and Fit Geometry
----------------------

.. code-block:: python

   cloud = nfm.embedding.delay(x, dimension=3, tau_samples=30)
   fit = nfm.geometry.fit_torus(cloud)
   print(fit.major_radii, fit.tube_radius)
   print(fit.score, fit.error)

Each row contains the current sample followed by delayed samples:
``[x(t), x(t-tau), x(t-2*tau)]``. Select a delay that resolves the dynamics
of interest. Three-dimensional clouds can be fitted with ``fit_ring`` or
``fit_torus``; ``FitConfig`` controls regularization and hole size.

The score is the fraction of points inside the fitted geometry, and the
error is the mean distance outside it. These quantify geometric fit, not
topological recovery. Ring containment is measured in the fitted plane;
ring MSE also includes the out-of-plane distance.

Learn Time-Varying Dynamics
---------------------------

.. code-block:: python

   import torch

   model = nfm.models.DeepLagField(sequence_length=600, orders=(2, 3, 4, 5, 6))
   model.eval()
   batch = torch.tensor(x[:600], dtype=torch.float32).unsqueeze(0)
   with torch.no_grad():
       output = model(batch)
   print(output.coefficients.shape, output.order)

A new model has untrained weights. The :doc:`tutorials/deeplagfield`
tutorial covers training and checkpoint loading. Inputs have shape
``(batch, time)``; coefficients have shape ``(batch, time, lag)``.
Evaluation selects the highest-scoring AR order, while training uses
straight-through Gumbel sampling.
