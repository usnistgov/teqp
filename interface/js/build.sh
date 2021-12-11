#!/bin/bash

mkdir /bld
cd /bld
cmake /src -DCMAKE_TOOLCHAIN_FILE=/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DTEQP_NO_PYTHON=ON -DTEQP_JAVASCRIPT_MODULE=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .

node --trace-uncaught catch_tests.js
exit 0