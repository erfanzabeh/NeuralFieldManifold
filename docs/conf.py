"""Sphinx configuration for Neural Field Manifold."""

project = "Neural Field Manifold"
author = "Erfan Zabeh and contributors"
copyright = "2026, Erfan Zabeh and contributors"
release = "1.0.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]
html_theme = "pydata_sphinx_theme"
exclude_patterns = ["_build"]
nb_execution_mode = "off"
autodoc_member_order = "bysource"
html_title = "Neural Field Manifold"
html_logo = "_static/branding/nfm-horizontal-red.svg"
html_favicon = "_static/branding/favicon.svg"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_context = {"default_mode": "light"}
html_sidebars = {
    "installation": ["installation-nav.html"],
    "getting_started": ["installation-nav.html"],
    "**": ["sidebar-nav-bs.html"],
}
html_baseurl = "https://erfanzabeh.github.io/NeuralFieldManifold/"
html_theme_options = {
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "logo": {"alt_text": "Neural Field Manifold - Home"},
    "github_url": "https://github.com/erfanzabeh/NeuralFieldManifold",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": [],
}
nb_code_prompt_show = "Show code"
nb_code_prompt_hide = "Hide code"
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "html_image"]
