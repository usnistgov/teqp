
# Intro

This library implements advanced derivative techniques to allow for implementation of EOS without any hand-written derivatives.  The name TEQP comes from Templated Equation of State Package

[![Catch tests via Github Actions](https://github.com/ianhbell/teqp/actions/workflows/runcatch.yml/badge.svg)](https://github.com/ianhbell/teqp/actions/workflows/runcatch.yml)

Written by Ian Bell, NIST.  

## Build (cmake based)

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
