"""Release-candidate documentation; no notebook execution or tests on build."""

project = "Neural Field Manifold"
release = "0.1.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_nb"]
html_theme = "sphinx_book_theme"
exclude_patterns = ["_build"]
nb_execution_mode = "off"
autodoc_member_order = "bysource"
html_title = "Neural Field Manifold"
