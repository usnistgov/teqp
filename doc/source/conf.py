# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, subprocess, shutil
on_actions = os.getenv('GITHUB_ACTION') != None

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

here = os.path.dirname(__file__)

# -- Project information -----------------------------------------------------

project = 'teqp'
copyright = '2022, Ian Bell'
author = 'Ian Bell'

# The full version, including alpha/beta/rc tags
import teqp
release = teqp.__version__

# -- Execute all notebooks --------------------------------------------------
if not on_actions:
    import sphinx_pre_run
    sphinx_pre_run.run()

### -- Auto-generate API documentation -----------------------------------------
subprocess.check_output(f'sphinx-apidoc -f -o api {os.path.dirname(teqp.__file__)}', shell=True, cwd=here)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'nbsphinx',
'sphinx.ext.autodoc',
'sphinxcontrib.doxylink',
'sphinx.ext.githubpages',
'sphinx.ext.imgconverter'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

doxylink = {
    'teqp' : (os.path.abspath(here+'/_static/doxygen/html/teqp.tag'), '_static/doxygen/html'),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if True:
    html_theme = 'alabaster'
else:
    html_theme = 'insipid'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for TeX output -------------------------------------------------

latex_engine = 'xelatex'

latex_elements = {
    'preamble': open(os.path.abspath(here+'/../macros.tex')).read()
}


