# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import automl_sphinx_theme
from dacbench import __version__ as version
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

name = "DACBench"
copyright = "2021, Theresa Eimer, Maximilian Reimer"
author = "Theresa Eimer, Maximilian Reimer"

# The full version, including alpha/beta/rc tags
release = "01.02.2021"

options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "versions": {
        f"v{version}": "#",
    },
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/automl/DACBench",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
    # this is here to exclude the gallery for examples
    "extensions": [
        "myst_parser",
        "sphinx.ext.autodoc",
        "sphinx.ext.viewcode",
        "sphinx.ext.napoleon",  # Enables to understand NumPy docstring
        # "numpydoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.autosectionlabel",
        "sphinx_autodoc_typehints",
        "sphinx.ext.doctest",
    ],
}
automl_sphinx_theme.set_options(globals(), options)
