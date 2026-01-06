# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add the project root to the Python path so autodoc can find the package
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HydroAngleAnalyzer"  # Capitalized for proper naming conventions
copyright = "2025, Gabriel"
author = "Gabriel"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.viewcode",  # Include links to source code
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.mathjax",  # Render math notation
    "sphinx.ext.coverage",  # Check documentation coverage
    "sphinx.ext.autosummary",  # Generate summary tables for modules
]

# Autosummary settings
autosummary_generate = True  # Automatically generate stub files

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "../images/logo-HydroAngleAnalyzer.png"
html_theme_options = {
    "repository_url": "https://github.com/username/hydroangleanalyzer",
    "use_repository_button": True,
}
html_static_path = ["_static"]

# -- Napoleon options (for better docstring parsing) -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_attr_annotations = True
