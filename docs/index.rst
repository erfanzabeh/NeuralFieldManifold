Neural Field Manifold
=====================

.. image:: _static/logo.jpg
   :alt: Neural Field Manifold
   :width: 300px
   :align: center

Autoregressive dynamics and delay-coordinate geometry for neural time series.

**Neural Field Manifold** connects local signal dynamics to the geometry of
delay embeddings. Process a recording, estimate autoregressive dynamics,
reconstruct a trajectory, and quantify its ring or torus geometry through
a single Python workflow.

.. image:: _static/overview.jpg
   :alt: From oscillatory signals to delay-coordinate geometry
   :width: 100%

Getting Started
---------------

Install the package, load a sample recording, and fit your first delay
embedding in the :doc:`getting_started` guide. The :doc:`api` documents
parameters, array shapes, and return values.

DeepLagField
------------

DeepLagField jointly estimates local autoregressive order and time-varying
coefficients. Its coupled networks are trained using reconstruction,
order-classification, target-power, and coefficient-smoothness losses.

.. image:: _static/arch.jpg
   :alt: DeepLagField order-selection and dynamics networks
   :width: 100%

Follow the :doc:`tutorials/deeplagfield` tutorial to generate training data,
train a model, and evaluate its predictions.

Tutorials
---------

* :doc:`tutorials/monkey`: processing, prediction, and delay geometry.
* :doc:`tutorials/mouse`: mouse LFP dynamics in the 1-50 Hz band.
* :doc:`tutorials/mouse_lfp`: low-frequency mouse LFP and delay embedding.
* :doc:`tutorials/lorenz`: Lorenz simulation and threshold autoregression.
* :doc:`tutorials/intrinsic_dimension`: oscillatory modes and embedding dimension.
* :doc:`tutorials/source_mixing`: sensor mixtures and geometric comparison.

Gallery
-------

.. image:: _static/monkey.jpg
   :alt: Monkey LFP reconstruction
   :width: 100%

.. image:: _static/mouse1.jpg
   :alt: Mouse neural recording reconstruction
   :width: 100%

.. image:: _static/mouse2.jpg
   :alt: Low-frequency reconstruction
   :width: 100%

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   getting_started

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/monkey
   tutorials/mouse
   tutorials/mouse_lfp
   tutorials/lorenz
   tutorials/intrinsic_dimension
   tutorials/deeplagfield
   tutorials/source_mixing

.. toctree::
   :hidden:
   :caption: Reference

   api

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
