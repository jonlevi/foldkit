# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))  # So autodoc can find your package

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "foldkit"
copyright = "2026, Jonathan Levine"
author = "Jonathan Levine"
release = "1.0.1"
version = "1.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # generates per-object API pages, like scanpy
    "sphinx.ext.napoleon",  # Google / NumPy-style docstrings
    "sphinx.ext.viewcode",  # "[source]" links next to API entries
    "sphinx.ext.intersphinx",  # cross-link to numpy/python docs
    "sphinx_autodoc_typehints",  # show type hints in docs
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx_design",  # grids / cards / tabs on the landing page
]

# Autosummary/autodoc behavior
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"  # move type hints out of the signature, into the text
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_custom_sections = [("Inputs", "params_style")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_title = "FoldKit"
# html_logo = "_static/logo.png"       # drop a logo file in _static/ and uncomment
# html_favicon = "_static/favicon.ico"  # drop a favicon in _static/ and uncomment

html_theme_options = {
    "github_url": "https://github.com/jonlevi/foldkit",
    "show_prev_next": False,
    "navigation_with_keys": True,
    "collapse_navigation": True,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jonlevi/foldkit",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/foldkit/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

html_context = {
    "default_mode": "auto",  # respects OS light/dark preference, like scanpy's docs
}
