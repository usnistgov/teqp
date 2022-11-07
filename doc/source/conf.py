# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, subprocess
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

here = os.path.dirname(__file__)

# -- Project information -----------------------------------------------------

project = 'teqp'
copyright = '2022, Ian Bell'
author = 'Ian Bell'

# The full version, including alpha/beta/rc tags
import teqp
release = teqp.__version__

# -- Execute all notebooks --------------------------------------------------

# Run doxygen
if not os.path.exists(here+'/_static/'):
    os.makedirs(here+'/_static')
subprocess.check_call('doxygen Doxyfile', cwd=here+'/../..', shell=True)

if on_rtd:
    # subprocess.check_output(f'jupyter nbconvert --version', shell=True)
    for path, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.ipynb') and '.ipynb_checkpoints' not in path:
                subprocess.check_output(f'jupyter nbconvert  --to notebook --output {file} --execute {file}', shell=True, cwd=path)
                # --ExecutePreprocessor.allow_errors=True      (this allows you to allow errors globally, but a raises-exception cell tag is better)


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
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

doxylink = {
    'teqp' : ('source/_static/doxygen/html/teqp.tag', '_static/doxygen/html'),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if on_rtd:
    html_theme = 'default'
else:
    html_theme = 'insipid'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
