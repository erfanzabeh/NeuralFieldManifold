Neural Field Manifold
=====================

.. raw:: html

   <p class="project-tagline">Recover low-dimensional manifold geometry from noisy, autocorrelated time series</p>
   <div class="project-badges">
     <a href="https://erfanzabeh.github.io/NeuralFieldManifold/installation.html"><img alt="Installation" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
     <a href="https://erfanzabeh.github.io/NeuralFieldManifold/"><img alt="Documentation" src="https://img.shields.io/badge/docs-online-blue.svg"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/actions/workflows/docs.yml"><img alt="Documentation build" src="https://github.com/erfanzabeh/NeuralFieldManifold/actions/workflows/docs.yml/badge.svg"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/erfanzabeh/NeuralFieldManifold"></a>
     <a href="https://erfanzabeh.github.io/NeuralFieldManifold/citation.html"><img alt="NeurIPS 2026 paper" src="https://img.shields.io/badge/NeurIPS-2026-68478D.svg"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/issues"><img alt="Open issues" src="https://img.shields.io/github/issues/erfanzabeh/NeuralFieldManifold"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/blob/main/CONTRIBUTING.md"><img alt="Contributing" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"></a>
   </div>

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

Source Mixing
-------------

:doc:`tutorials/volume_conduction_manifold` examines genuine and mixed-source
delay geometry, mixing-strength sweeps, spatial consistency, synthetic trial
classification, and multichannel monkey LFP and mouse EEG examples.

Coming Soon
-----------

* :doc:`tutorials/upcoming`
* :doc:`tutorials/autocorrelated_manifolds`

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   getting_started
   tutorials/index
   api
   citation

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
