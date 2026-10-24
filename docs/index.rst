Neural Field Manifold
=====================

Tutorials and User Guide
-------------------------

Explore the relationship between signal dynamics and delay-coordinate
geometry, then reconstruct manifolds from synthetic and recorded signals.
The notebooks below cover the complete analyses, including diagnostics,
parameter sweeps, and comparisons across channels.

.. image:: _static/overview.jpg
   :alt: Oscillatory dynamics and delay-coordinate geometry
   :width: 100%

Start Here
----------

Install the package with :doc:`installation`, or follow
:doc:`getting_started` for a short introduction to the Python API.

Foundations
-----------

* :doc:`tutorials/intrinsic_extrinsic` compares PCA embedding dimensionality,
  oscillatory AR roots, colored noise, and time-varying dynamics.
* :doc:`tutorials/simulations` follows the Lorenz system from spectral analysis
  and AR order selection to residual diagnostics and threshold-AR reconstruction.
* :doc:`tutorials/autocorrelated_manifolds` introduces an upcoming tutorial.

Recorded Signals
----------------

* :doc:`tutorials/monkey`: monkey V1 LFP reconstruction, prediction diagnostics,
  time-window comparisons, delay sweeps, and multichannel reconstruction.
* :doc:`tutorials/mouse`: mice hippocampus reconstruction with the original
  processing, prediction, and multichannel analyses.
* :doc:`tutorials/mouse_lfp`: mice hippocampus LFP reconstruction, including
  residual analysis and window, delay, and channel comparisons.

DeepLagField
------------

.. image:: _static/arch.jpg
   :alt: DeepLagField order-selection and dynamics networks
   :width: 100%

The :doc:`tutorials/model` tutorial covers all eight coefficient-schedule
families, synthetic-data generation, training, order-classification confusion
matrices, coefficient-recovery benchmarks, and predicted coefficient trajectories.
:doc:`tutorials/upcoming` describes the planned fine-tuning tutorial.

Source Mixing
-------------

:doc:`tutorials/volume_conduction_manifold` examines genuine and mixed-source
delay geometry, mixing-strength sweeps, spatial consistency, synthetic trial
classification, and multichannel monkey LFP and mouse EEG examples.

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   getting_started

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/autocorrelated_manifolds
   tutorials/intrinsic_extrinsic
   tutorials/simulations
   tutorials/monkey
   tutorials/mouse
   tutorials/mouse_lfp
   tutorials/model
   tutorials/upcoming
   tutorials/volume_conduction_manifold

.. toctree::
   :hidden:
   :caption: Reference

   api
   citation

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
