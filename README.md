
# Intro

This library implements advanced derivative techniques to allow for implementation of EOS without any hand-written derivatives.  The name TEQP comes from Templated Equation of State Package

So far the following EOS are implemented:

* van der Waals
* Peng-Robinson
* Soave-Redlich-Kwong
* PC-SAFT
* cubic plus association (CPA) for pure fluids
* multi-fluid model in the form of GERG

Why?

* Implementing an EOS is an error-prone and boring exercise. Automatic differentiation packages are a mature solution for calculating derivatives 
* Algorithms can be implemented in a very generic way that is model-agnostic

Docs are on ReadTheDocs: [![Documentation Status](https://readthedocs.org/projects/teqp/badge/?version=latest)](https://teqp.readthedocs.io/en/latest/?badge=latest)

Written by Ian Bell, NIST.  

## Changelog

[![PyPI version](https://badge.fury.io/py/teqp.svg)](https://badge.fury.io/py/teqp)

* 0.10.0 :

  * Add isobar tracing for VLE of binary mixtures (exposed to Python)

  * Add ``IdealHelmholtz`` class for ideal-gas Helmholtz energy contribution (exposed to Python)

  * Bugfix: Fix order of coefficients in one departure term. See [f1da57](https://github.com/usnistgov/teqp/commit/f1da57a586db9bda0a21f74c843cb263208fc110).  Had been wrong in CoolProp for many years.

* 0.9.5 :

  * Bugfix: Fix the eigenvector orientiation as well when taking the temperature derivative in critical curve tracing. See [b70178f7](https://github.com/usnistgov/teqp/commit/b70178f79fe9f4950b3075b80fec4495313ffba5)

* 0.9.4 :

  * Expose the a and b parameters of cubic EOS. See [84ebc0fb](https://github.com/usnistgov/teqp/commit/84ebc0fb258ff42af30b2521b02a2a4984b7e715)

* 0.9.3 :

  * Bugfix: Fixed stopping condition in ``mix_VLE_Tx`` (if ``dx`` was negative, automatic stop, missing ``abs``).  See [d87e91e](https://github.com/usnistgov/teqp/commit/d87e91ea10bbbd936993edd02f85a53ccd42817d)

* 0.9.2 :

  * Bugfix: ``kmat`` can be set also when specifying ``sigma`` and ``e/kB`` with PC-SAFT

* 0.9.1 :

  * Transcription error in a coefficient of PC-SAFT

* 0.9.0 :

  * Add ability to obtain ancillaries for multifluid model (``see teqp/models/multifluid_ancillaries.hpp``) or the ``build_ancillaries`` method in python

  * Enable ability to use multiprecision with PC-SAFT

* 0.8.1 :

  * Replace the ``get_Ar20`` function that was erroneously removed

* 0.8.0 Significant changes include:
  
  * kij can be set for PC-SAFT and cubics (PR & SRK)

  * Added Lennard-Jones EOS from Thol et al.

  * Partial molar volume is now an available output

  * Added solver for pure fluid critical point

  * Added 2D Chebyshev departure function

  * Starting work on a C++ wrapper in the hopes of improving compile times for C++ integration

* 0.7.0 Significant changes include:
  
  * ``get_Arxy`` generalized to allow for any derivative

  * Local stability tests for critical points can be enabled

  * Critical curve polishers much more reliable

  * Add a method for dp/dT along isopleth of phase envelope of mixture.

  * Estimation is not enabled by default by the ``estimation`` flag. If that is desired, use ``force-estimation``

* 0.6.0 Add VLLE from VLE routine based upon https://pubs.acs.org/doi/abs/10.1021/acs.iecr.1c04703

* 0.5.0 Add VLE polishing routine [50b61af0](https://github.com/usnistgov/teqp/commit/50b61af05697c01c0a2bf686b256724cc79f73d4), fix bug in critical curve tracing misalignment of eigenvectors [f01ac7be](https://github.com/usnistgov/teqp/commit/f01ac7be43fcca4e1cd3c502be0259755396436b), assorted issue fixes, especially support for javascript

* 0.4.0 Add VLE tracing, code coverage and valgrind testing, fugacity_coefficients, generalize loading of multifluid models

* 0.3.0 Add integration options to the tracing of the critical curve; fix ``__version__``

* 0.2.0 Add fluid files to the python package

## Install

For all users, you should be able to install the most recent version from PYPI with

``pip install teqp``

## Tests

[![Catch tests via Github Actions](https://github.com/usnistgov/teqp/actions/workflows/runcatch.yml/badge.svg)](https://github.com/usnistgov/teqp/actions/workflows/runcatch.yml)

## Build (cmake based)

Try it in your browser: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/usnistgov/teqp/master)

Be aware: compiling takes a while in release mode (multiple minutes per file in some cases) thanks to the use of generic typing in the models.  Working on making this faster...

For example to build the critical line tracing example in visual studio, do:

```
mkdir build
cd build
cmake .. 
cmake --build . --target multifluid_crit --config Release
Release\multifluid_crit
```
On linux/OSX, similar:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target multifluid_crit
./multifluid_crit
```
### Random notes for future readers:

* When building in WSL via VS Code, you might need to enable metadata to avoid pages of configure errors in cmake: https://github.com/microsoft/WSL/issues/4257
* Debugging in WSL via VS Code (it really works!): https://code.visualstudio.com/docs/cpp/config-wsl
