Installation
============

Install from source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/erfanzabeh/NeuralFieldManifold.git
   cd NeuralFieldManifold
   pip install -e .

To include development and documentation extras:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Requirements
------------

- Python ≥ 3.10
- PyTorch, JAX, NumPy, SciPy, scikit-learn, and other dependencies
  listed in ``pyproject.toml``

Install documentation dependencies
-----------------------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
