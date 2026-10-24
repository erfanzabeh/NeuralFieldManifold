Installation
============

Neural Field Manifold requires Python 3.10 or later and depends on NumPy,
SciPy, and PyTorch.

Install from GitHub
-------------------

.. code-block:: sh

   python -m pip install "neuralfieldmanifold @ git+https://github.com/erfanzabeh/NeuralFieldManifold.git@main"

Optional Dependencies
---------------------

Use ``plot`` for Matplotlib and Plotly support, or ``tutorials`` for the
complete notebook environment:

.. code-block:: sh

   python -m pip install "neuralfieldmanifold[tutorials] @ git+https://github.com/erfanzabeh/NeuralFieldManifold.git@main"

For an editable installation:

.. code-block:: sh

   git clone https://github.com/erfanzabeh/NeuralFieldManifold.git
   cd NeuralFieldManifold
   python -m pip install -e ".[tutorials]"

Verify the Installation
-----------------------

.. code-block:: python

   import neuralfieldmanifold as nfm

   recording = nfm.datasets.load("monkey_lfp")
   print(recording.signal.shape, recording.sampling_rate_hz)

Continue with :doc:`getting_started` or open a notebook in ``docs/tutorials``.
