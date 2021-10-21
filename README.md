
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

0.3.0 Add integration options to the tracing of the critical curve; fix ``__version__``

0.2.0 Add fluid files to the python package

## Install

For windows and OSX, you should be able to install the most recent version from PYPI with

``pip install teqp``

For linux users, please build yourself following the instructions below.

The windows wheels are built/pushed with the ``buildwheels.py`` script, the OSX wheels are built as part of github action

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
