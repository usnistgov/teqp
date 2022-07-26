
# Intro

This library implements advanced derivative techniques to allow for implementation of EOS without any hand-written derivatives.  The name TEQP comes from Templated Equation of State Package

So far the following EOS are implemented:

* van der Waals
* Peng-Robinson
* Soave-Redlich-Kwong
* PC-SAFT
* cubic plus association (CPA) for pure fluids
* multi-fluid model in the form of GERG

Written by Ian Bell, NIST.  

## Changelog

[![PyPI version](https://badge.fury.io/py/teqp.svg)](https://badge.fury.io/py/teqp)

* 0.9.3 :

  * Bugfix: Fixed stopping condition in ``mix_VLE_Tx`` (if ``dx`` was negative, automatic stop, missing ``abs``).  See [50b61a](https://github.com/usnistgov/teqp/commit/50b61af05697c01c0a2bf686b256724cc79f73d4)

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

* 0.5.0 Add VLE polishing routine (https://github.com/usnistgov/teqp/commit/50b61af05697c01c0a2bf686b256724cc79f73d4), fix bug in critical curve tracing misalignment of eigenvectors(https://github.com/usnistgov/teqp/commit/f01ac7be43fcca4e1cd3c502be0259755396436b), assorted issue fixes, especially support for javascript

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
