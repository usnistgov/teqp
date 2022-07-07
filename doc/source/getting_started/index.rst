Getting Started
===============

Introduction
------------

teqp (phonetically: tɛk pi) is a C++-based library with wrappers. It was written because implementing EOS (particularly working out the derivatives) is a painful, error-prone, and slow process.  The advent of open-source automatic differentiation libraries makes the implementation of EOS as fast as hand-written derivatives, and much easier to implement without errors.

There is a paper about teqp: https://doi.org/10.1021/acs.iecr.2c00237

The documentation is based on the Python wrapper because it can be readily integrated with the documentation tools (sphinx in this case) and can be auto-generated at documentation build time.

Installation
------------

Python
^^^^^^

The library can be installed with:

.. code::

   pip install teqp

because the binary wheels for all major platforms are provided on pypi.

If you desire to build teqp yourself, it is recommended to pull from github and build a binary wheel, and then subsequently install that wheel:

.. code::

    git clone --recursive https://github.com/usnistgov/teqp
    cd teqp
    python setup.py bdist_wheel
    pip install dist/*.whl # or replace with the appropriate binary wheel

C++
^^^

The build is cmake based.  There are targets available for an interface library, etc.  Please see ``CMakeLists.txt``