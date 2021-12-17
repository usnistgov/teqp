#!/bin/bash

mkdir /bld
cd /bld
cmake /src -DCMAKE_TOOLCHAIN_FILE=/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DTEQP_NO_PYTHON=ON -DTEQP_JAVASCRIPT_MODULE=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Release -DTEQP_SNIPPETS=ON
cmake --build . --target emtest_catch_exception2
cmake --build . --target bench
cmake --build . --target catch_tests
cp emtest_catch_exception2.* /src
cp bench.* /src
cp catch_tests.* /src

node --experimental-wasm-eh emtest_catch_exception2.js
exit 0