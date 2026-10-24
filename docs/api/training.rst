Training and Evaluation
=======================

Generate labeled time-varying AR data, train DeepLagField, and evaluate
coefficients, predictions, and order selection. All order labels are AR
orders rather than zero-based category indices.

Synthetic Datasets
------------------

.. autoclass:: neuralfieldmanifold.training.TVARDataset
   :members:

.. autofunction:: neuralfieldmanifold.training.make_tvar_dataset

.. autofunction:: neuralfieldmanifold.training.split_dataset

Loss Functions
--------------

.. autoclass:: neuralfieldmanifold.training.LossWeights
   :members:

.. autofunction:: neuralfieldmanifold.training.compute_losses

Training
--------

.. autoclass:: neuralfieldmanifold.training.TrainingConfig
   :members:

.. autofunction:: neuralfieldmanifold.training.train

Evaluation
----------

.. autofunction:: neuralfieldmanifold.training.evaluate

