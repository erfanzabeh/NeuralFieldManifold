Neural Field Manifold
=====================

.. raw:: html

   <p class="project-tagline">Recover low-dimensional manifold geometry from noisy, autocorrelated time series</p>
   <div class="project-badges">
     <a href="citation.html"><img alt="NeurIPS 2026 paper" src="https://img.shields.io/badge/NeurIPS-2026-68478D.svg"></a>
     <a href="installation.html"><img alt="Python 3.10 or later" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/actions/workflows/docs.yml"><img alt="Documentation" src="https://img.shields.io/github/actions/workflow/status/erfanzabeh/NeuralFieldManifold/docs.yml?branch=main&label=Documentation&logo=github"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/actions/workflows/build.yml"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/erfanzabeh/NeuralFieldManifold/build.yml?branch=main&label=Build&logo=github"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/erfanzabeh/NeuralFieldManifold"></a>
     <a href="https://github.com/erfanzabeh/NeuralFieldManifold/issues"><img alt="Open issues" src="https://img.shields.io/github/issues/erfanzabeh/NeuralFieldManifold"></a>
     <a href="installation.html"><img alt="PyPI: XXXX (forthcoming)" src="https://img.shields.io/badge/PyPI-XXXX-3775A9?logo=pypi&logoColor=white"></a>
     <a href="citation.html"><img alt="arXiv: XXXX (forthcoming)" src="https://img.shields.io/badge/arXiv-XXXX-B31B1B?logo=arxiv&logoColor=white"></a>
   </div>

Connect local signal dynamics to delay-coordinate geometry. Process neural
recordings, estimate autoregressive dynamics with classical models or
DeepLagField, and reconstruct and measure their ring and torus geometry.

.. image:: _static/overview.jpg
   :alt: Oscillatory dynamics and delay-coordinate geometry
   :width: 100%

Installation
------------

Requires Python 3.10 or later. Install from GitHub while the PyPI release
is being prepared:

.. code-block:: sh

   python -m pip install "neuralfieldmanifold @ git+https://github.com/erfanzabeh/NeuralFieldManifold.git@main"

See :doc:`installation` for plotting, notebook dependencies, and editable installs.

Getting Started
---------------

Load a bundled recording, process it, and measure its delay geometry:

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
   cloud = nfm.embedding.delay(processed.signal, dimension=3, tau_samples=30)
   fit = nfm.geometry.fit_torus(cloud)
   print(fit.score, fit.error)

Use your own one-dimensional signal and sampling rate in place of the sample.
Choose filtering bands and delays appropriate to that recording.
Continue with :doc:`getting_started` or browse the :doc:`api`.

Foundations
-----------

Explore intrinsic and embedding dimensions, oscillatory dynamics, and
synthetic manifold reconstruction in the :doc:`tutorial library <tutorials/index>`.

Demonstrations
--------------

The :doc:`tutorial library <tutorials/index>` includes monkey V1 LFP,
mice hippocampus EEG and LFP, and DeepLagField training examples, with
saved figures and complete analysis code.

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   tutorials/index
   api
   citation
