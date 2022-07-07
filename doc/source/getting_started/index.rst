Getting Started
===============

Introduction
------------

teqp is a C++-based library with wrappers. It was written because implementing EOS (particularly working out the derivatives) is a painful , error-prone, and slow process.  The advent of automatic differentiation makes the implementation of EOS as fast as hand-written derivatives, and much easier to implement without errors.

The documentation is based on the Python wrapper because it can be readily integrated with the documentation tools and can be auto-generated at documentation build time.

Installation
------------

Python
^^^^^^

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

C++
^^^

The build is cmake based.  There are targets available for an interface library, etc.  Please see ``CMakeLists.txt``