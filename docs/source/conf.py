# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))  # So autodoc can find your package

project = 'foldkit'
copyright = '2025, Jonathan Levine'
author = 'Jonathan Levine'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []




extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         # For Google-style docstrings
    "sphinx_autodoc_typehints",    # Show type hints in docs
]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "navigation_depth": 2,
    "show_toc_level": 2,
    "github_url": "https://github.com/jonlevi/foldkit",
}

html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"]
}
