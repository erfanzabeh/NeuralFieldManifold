Simulation
==========

Generate autoregressive signals, classical dynamical systems, noise,
and sensor mixtures. Frequency-based AR construction uses Hz; coefficient
schedules specify variation over sample indices.

Autoregressive Signals
----------------------

.. autofunction:: neuralfieldmanifold.simulation.coefficients_from_modes

.. autofunction:: neuralfieldmanifold.simulation.ar

.. autofunction:: neuralfieldmanifold.simulation.tvar

.. autoexception:: neuralfieldmanifold.simulation.SimulationRejected

Dynamical Systems
-----------------

.. autofunction:: neuralfieldmanifold.simulation.lorenz

.. autofunction:: neuralfieldmanifold.simulation.ornstein_uhlenbeck

.. autofunction:: neuralfieldmanifold.simulation.rotating_var

Noise
-----

.. autofunction:: neuralfieldmanifold.simulation.white_noise

.. autofunction:: neuralfieldmanifold.simulation.colored_noise

Source Mixing
-------------

.. autofunction:: neuralfieldmanifold.simulation.oscillatory_ar

.. autofunction:: neuralfieldmanifold.simulation.mixing_matrix

Coefficient Schedules
---------------------

.. automodule:: neuralfieldmanifold.simulation.schedules
   :members:
