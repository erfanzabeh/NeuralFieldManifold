API Reference
=============

.. contents::
   :local:
   :depth: 2

Signals and Spectra
-------------------

Load sample recordings, configure filtering and envelope normalization,
and estimate power spectral densities.

Sample Recordings
~~~~~~~~~~~~~~~~~

.. automodule:: neuralfieldmanifold.datasets
   :members:

Preprocessing
~~~~~~~~~~~~~

.. automodule:: neuralfieldmanifold.preprocessing
   :members:

Power Spectra
~~~~~~~~~~~~~

.. automodule:: neuralfieldmanifold.spectral
   :members:

Delay Embedding
---------------

Build delay-coordinate trajectories, lag matrices for autoregression, and
sliding windows. Coordinates are returned as columns, with the newest
sample first; rows advance in time.

.. automodule:: neuralfieldmanifold.embedding
   :members:

Autoregressive Models
---------------------

Fit a classical AR or threshold AR model, or learn time-varying coefficients
and order with DeepLagField. See :doc:`tutorials/model` for a
complete training example.

Linear AR
~~~~~~~~~

.. autoclass:: neuralfieldmanifold.models.AR
   :members:

.. autoclass:: neuralfieldmanifold.models.ARFit
   :members:

.. autoclass:: neuralfieldmanifold.models.PredictionMetrics
   :members:

.. autofunction:: neuralfieldmanifold.models.prediction_metrics

Threshold AR
~~~~~~~~~~~~

.. autoclass:: neuralfieldmanifold.models.TAR
   :members:

.. autoclass:: neuralfieldmanifold.models.TARFit
   :members:

DeepLagField
~~~~~~~~~~~~

.. autoclass:: neuralfieldmanifold.models.DeepLagField
   :members: forward, save, load

.. autoclass:: neuralfieldmanifold.models.DynamicsOutput
   :members:

Geometric Fitting
-----------------

Fit elliptical rings or tori to three-dimensional clouds and compare
trajectories with Procrustes disparity. Fits retain the input coordinate
scale. Containment score and outside distance quantify geometric fit;
they do not establish topological recovery.

Ring and Torus Fits
~~~~~~~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.geometry.fit_ring

.. autofunction:: neuralfieldmanifold.geometry.fit_torus

Settings and Results
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neuralfieldmanifold.geometry.FitConfig
   :members:

.. autoclass:: neuralfieldmanifold.geometry.GeometryFit
   :members:

Trajectory Comparison
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.geometry.pairwise_disparity

.. autoclass:: neuralfieldmanifold.geometry.Disparity
   :members:

Simulation
----------

Generate autoregressive signals, classical dynamical systems, noise,
and sensor mixtures. Frequency-based AR construction uses Hz; coefficient
schedules specify variation over sample indices.

Autoregressive Signals
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.simulation.coefficients_from_modes

.. autofunction:: neuralfieldmanifold.simulation.ar

.. autofunction:: neuralfieldmanifold.simulation.tvar

.. autoexception:: neuralfieldmanifold.simulation.SimulationRejected

Dynamical Systems
~~~~~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.simulation.lorenz

.. autofunction:: neuralfieldmanifold.simulation.ornstein_uhlenbeck

.. autofunction:: neuralfieldmanifold.simulation.rotating_var

Noise
~~~~~

.. autofunction:: neuralfieldmanifold.simulation.white_noise

.. autofunction:: neuralfieldmanifold.simulation.colored_noise

Source Mixing
~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.simulation.oscillatory_ar

.. autofunction:: neuralfieldmanifold.simulation.mixing_matrix

Coefficient Schedules
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: neuralfieldmanifold.simulation.schedules
   :members:

Training and Evaluation
-----------------------

Generate labeled time-varying AR data, train DeepLagField, and evaluate
coefficients, predictions, and order selection. All order labels are AR
orders rather than zero-based category indices.

Synthetic Datasets
~~~~~~~~~~~~~~~~~~

.. autoclass:: neuralfieldmanifold.training.TVARDataset
   :members:

.. autofunction:: neuralfieldmanifold.training.make_tvar_dataset

.. autofunction:: neuralfieldmanifold.training.split_dataset

Loss Functions
~~~~~~~~~~~~~~

.. autoclass:: neuralfieldmanifold.training.LossWeights
   :members:

.. autofunction:: neuralfieldmanifold.training.compute_losses

Training
~~~~~~~~

.. autoclass:: neuralfieldmanifold.training.TrainingConfig
   :members:

.. autofunction:: neuralfieldmanifold.training.train

Evaluation
~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.training.evaluate

Plotting
--------

Visualize a trajectory with Matplotlib or inspect a fitted geometry in
rotatable three-dimensional Plotly views. Install the ``plot`` or
``tutorials`` extra to use these functions.

.. automodule:: neuralfieldmanifold.plotting
   :members:

Tutorial Styling
~~~~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.plotting.set_style

.. autofunction:: neuralfieldmanifold.plotting.prettify

Training and Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neuralfieldmanifold.plotting.training_history

.. autofunction:: neuralfieldmanifold.plotting.coefficients
