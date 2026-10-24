"""Sphinx configuration for Neural Field Manifold."""

project = "Neural Field Manifold"
release = "0.1.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]
html_theme = "sphinx_book_theme"
exclude_patterns = ["_build"]
nb_execution_mode = "off"
autodoc_member_order = "bysource"
html_title = "Neural Field Manifold"
html_logo = "_static/logo.jpg"
html_static_path = ["_static"]
html_baseurl = "https://erfanzabeh.github.io/NeuralFieldManifold/"
html_theme_options = {
    "repository_url": "https://github.com/erfanzabeh/NeuralFieldManifold",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "logo": {"text": "Neural Field Manifold"},
}
