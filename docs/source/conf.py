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

# -- Project information -----------------------------------------------------

project = 'Canary'
copyright = '2021, Christopher Wales'
author = 'Christopher Wales'
# Configuration file for the Sphinx documentation builder.

release = '0.0.1'
version = '0.0.1'

add_module_names = False

# -- General configuration
html_static_path = ['_static']
todo_include_todos = True
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.viewcode",
    "sphinx_autobuild",
    "sphinx_copybutton",
    "myst_parser",
    # "sphinx_rtd_theme"
]
extlinks = {
    "pypi": ("https://pypi.org/project/%s/", ""),
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

templates_path = ['_templates']

# -- Options for HTML output
autosummary_generate = True  # Turn on sphinx.ext.autosummary

html_theme = 'furo'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_css_files = [
    'css/custom.css',
]
