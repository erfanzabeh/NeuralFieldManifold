# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'NeuralFieldManifold'
copyright = '2026, Erfan Zabeh'
author = 'Erfan Zabeh'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Suppress myst-nb deprecation warning with Sphinx 9.x
suppress_warnings = ["mystnb.unknown_mime_type"]
import warnings
warnings.filterwarnings("ignore", message=".*is deprecated.*Check CHANGES for Sphinx API.*")

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/erfanzabeh/NeuralFieldManifold",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "show_toc_level": 3,
    "navigation_with_keys": True,
    "toc_title": "On this page",
    "show_nav_level": 2,
}

html_title = "Neural Field Manifold"

# --- FIX 1: LOAD PLOTLY JAVASCRIPT ---
# This ensures the browser can actually draw the plots
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js", 
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- MyST-NB configuration (for Jupyter notebooks) --------------------------
# Do NOT re-run notebooks during docs build
nb_execution_mode = "off"

# Alternatively, use "cache" to run once and cache, or "auto" to run if changed
# nb_execution_mode = "cache"

# Notebook execution timeout (seconds)
nb_execution_timeout = 300

# Show notebook execution statistics
nb_execution_show_tb = True

# Source suffix for MyST
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# MyST parser extensions
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]