Autoregressive Models
=====================

Fit a classical AR or threshold AR model, or learn time-varying coefficients
and order with DeepLagField. See :doc:`../tutorials/deeplagfield` for a
complete training example.

Linear AR
---------

.. autoclass:: neuralfieldmanifold.models.AR
   :members:

.. autoclass:: neuralfieldmanifold.models.ARFit
   :members:

.. autoclass:: neuralfieldmanifold.models.PredictionMetrics
   :members:

.. autofunction:: neuralfieldmanifold.models.prediction_metrics

Threshold AR
------------

.. autoclass:: neuralfieldmanifold.models.TAR
   :members:

.. autoclass:: neuralfieldmanifold.models.TARFit
   :members:

DeepLagField
------------

.. autoclass:: neuralfieldmanifold.models.DeepLagField
   :members: forward, save, load

.. autoclass:: neuralfieldmanifold.models.DynamicsOutput
   :members:
