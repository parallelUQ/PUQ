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
import os
import sys

sys.path.append(os.path.abspath("../PUQ"))
sys.path.append(os.path.abspath("../PUQ/designmethods"))
sys.path.append(os.path.abspath("../PUQ/surrogatemethods"))
sys.path.append(os.path.abspath("../PUQ/surrogatesupport"))

# -- Project information -----------------------------------------------------

project = "PUQ"
copyright = "2024, Özge Sürer, Matthew Plumlee, Stefan M. Wild"
author = "Özge Sürer, Matthew Plumlee, Stefan M. Wild"

# The full version, including alpha/beta/rc tags
release = "0.1"

needs_sphinx = "3.0"
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.autodoc',
#              'sphinx.ext.napoleon',
#              'sphinx.ext.autosummary',
#              'sphinx.ext.autosectionlabel',
#              'sphinx.ext.intersphinx',
#              'sphinx.ext.imgconverter',
#              'sphinx.ext.mathjax']
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    #'numpydoc'
    "sphinx.ext.napoleon",
]
autoclass_content = "both"
autosummary_generate = True
# extensions = ['sphinx.ext.autodoc',
#              'sphinx.ext.autosummary',
#              'sphinx.ext.todo',
#              'sphinx.ext.viewcode',
#              'sphinx.ext.imgmath',
#              'numpydoc',
#              'sphinx.ext.autosectionlabel']

# autosummary_generate = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
html_static_path = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
