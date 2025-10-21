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

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
