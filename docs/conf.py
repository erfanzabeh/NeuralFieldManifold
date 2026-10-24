"""Package documentation; no notebook execution or tests on build."""

project = "Neural Field Manifold"
release = "0.1.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_nb"]
html_theme = "sphinx_book_theme"
exclude_patterns = ["_build"]
nb_execution_mode = "off"
autodoc_member_order = "bysource"
html_title = "Neural Field Manifold"
html_baseurl = "https://erfanzabeh.github.io/NeuralFieldManifold/"
html_theme_options = {
    "repository_url": "https://github.com/erfanzabeh/NeuralFieldManifold",
    "use_repository_button": True,
}
