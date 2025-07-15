# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
import importlib.metadata
sys.path.insert(0, os.path.abspath('../../../uqlm'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'uqlm'
copyright = '2025, CVS Health'
author = 'Dylan Bouchard, Mohit Singh Chauhan'
release = '0.1'
# version = importlib.metadata.version("uqlm")
# release = ".".join(version.rsplit(".")[:-1])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # Core library for html generation from docstrings
    "sphinx_autodoc_typehints", # Automatically document type hints
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.napoleon', # NumPy and Google style docsrings parsing
    "sphinx.ext.duration", # build duration
    "sphinx.ext.doctest", # Test snippets in the documentation
    "sphinxcontrib.bibtex", # Bibliographic references
    "sphinx_favicon", # Add favicon
    "nbsphinx", # Execute Jupyter notebooks + OSX  brew install pandoc
]
nbsphinx_execute="never"

bibtex_bibfiles = ["refs.bib"]

autosummary_generate = True

templates_path = ['_templates']

html_static_path = ['_static']

html_css_files = ['custom.css']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

favicons = [
    {
        "rel": "icon",
        "sizes": "16x16",
        "href": "images/favicon/favicon-16x16.png",
        "type": "image/png"
    },
    {
        "rel": "icon",
        "sizes": "32x32",
        "href": "images/favicon/favicon-32x32.png",
        "type": "image/png"
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "images/favicon/apple-touch-icon.png",
        "type": "image/png"
    },
]

html_theme = 'pydata_sphinx_theme'

html_favicon = '_static/images/favicon/favicon.ico'

html_theme_options = {
    "github_url": "https://github.com/cvs-health/uqlm",
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://cvs-health.github.io/uqlm/versions.json",
        "version_match": release,
    },
    "logo": {
        "image_light": "_static/images/horizontal_logo.png",
        "image_dark": "_static/images/horizontal_logo_no_bg.png",
    },
}

source_suffix = [".rst"]