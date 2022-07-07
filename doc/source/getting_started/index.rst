Introduction
============

teqp is a C++-based library with wrappers. The documentation is based on the Python wrapper because it can be readily integrated with the documentation tools and can be auto-generated at documentation build time.

Installation
============

Python
------

The library can be installed with:

.. code::

   pip install teqp

as the binary wheels for all major platforms are provided on pypi.

If you desire to build teqp yourself, it is recommended to pull from github and build a binary wheel:

.. code::

    git clone --recursive https://github.com/usnistgov/teqp
    cd teqp
    python setup.py bdist_wheel
    pip install dist/*.whl # or replace with the appropriate binary wheel