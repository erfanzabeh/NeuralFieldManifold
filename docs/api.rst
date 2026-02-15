API Reference
=============

Loss Functions
==============

.. autofunction:: stochastic_dynamics.utils.loss_p
.. autofunction:: stochastic_dynamics.utils.loss_ar
.. autofunction:: stochastic_dynamics.utils.loss_energy
.. autofunction:: stochastic_dynamics.utils.loss_smooth
.. autofunction:: stochastic_dynamics.utils.loss_order

Models
======

.. autoclass:: stochastic_dynamics.models.AR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stochastic_dynamics.models.TAR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stochastic_dynamics.models.DeepLagEmbed
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stochastic_dynamics.models.ARMLP
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stochastic_dynamics.models.ARTransformer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: stochastic_dynamics.models.AnalyticalAR
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
=========

.. autofunction:: stochastic_dynamics.utils.train_loop
.. autofunction:: stochastic_dynamics.utils.bench_loop
.. autofunction:: stochastic_dynamics.utils.bicoherence

Generators
==========

.. autofunction:: stochastic_dynamics.generators.tvar
.. autofunction:: stochastic_dynamics.generators.tvvar
.. autofunction:: stochastic_dynamics.generators.lorenz
.. autofunction:: stochastic_dynamics.generators.ou_exact
.. autofunction:: stochastic_dynamics.generators.ou_euler
.. autofunction:: stochastic_dynamics.generators.white_noise
.. autofunction:: stochastic_dynamics.generators.pink_noise
.. autofunction:: stochastic_dynamics.generators.brown_noise
.. autofunction:: stochastic_dynamics.generators.colored_noise

Embedders
=========

.. autofunction:: stochastic_dynamics.embedders.embed


Plotting
========

.. autofunction:: stochastic_dynamics.plottings.plot_history
.. autofunction:: stochastic_dynamics.plottings.plot_confusion_matrix
.. autofunction:: stochastic_dynamics.plottings.plot_coefficients_by_p
.. autofunction:: stochastic_dynamics.plottings.plot_tvar_sample
.. autofunction:: stochastic_dynamics.pub_utils.set_pub_style
.. autofunction:: stochastic_dynamics.pub_utils.prettify
