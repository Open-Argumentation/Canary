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

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)

# -- Project information -----------------------------------------------------

project = 'Canary'
copyright = '2021, Christopher Wales'
author = 'Christopher Wales'
html_title = "Canary"
release = '0.0.1'
version = '0.0.1'
pygments_style = 'sphinx'
add_module_names = False

# -- General configuration
html_static_path = ['_static']
todo_include_todos = True
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    "sphinx.ext.viewcode",
    "sphinx_autobuild",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'furo'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_css_files = [
    'css/custom.css',
]

autodoc_mock_imports = ["pkg_resources", "numpy", "nltk", "sklearn", "pybrat", "vaderSentiment", "spacy", "benepar",
                        "pandas", "sklearn_crfsuite", "scipy", "joblib"]
