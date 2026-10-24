Installation
============

Neural Field Manifold requires Python 3.10 or later and depends on NumPy,
SciPy, and PyTorch.

.. toctree::
   :hidden:

   getting_started

The PyPI release is forthcoming. Install the current package from GitHub.

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

Recorded-Data Tutorials
-----------------------

The quick start uses the small one-dimensional recordings bundled with the
package. The complete recorded-data tutorials also contain multichannel
analyses that require access to the original recordings. Set their paths
before starting Jupyter:

.. code-block:: sh

   export NFM_MONKEY_DATA_DIR=/path/to/monkey/arrays
   export NFM_MOUSE_DATA_FILE=/path/to/eke003_spont_250818_175844.npy

The full DeepLagField benchmark reads its saved synthetic training dataset
from ``NFM_TVAR_DATA_FILE`` (or ``train_data.npz`` in the working directory).
The data-generation section also shows how to construct a pilot dataset.

The website displays saved notebook results. Documentation builds do not
execute notebook code or retrain models.
