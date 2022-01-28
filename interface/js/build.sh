#!/bin/bash

mkdir /bldbind
cd /bldbind
cmake /src -DCMAKE_TOOLCHAIN_FILE=/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DTEQP_NO_PYTHON=ON -DTEQP_EMBIND_MODULARIZE_ES6=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Release -DTEQP_SNIPPETS=OFF
cmake --build . --target teqpbind

cp teqpbind.* /src/interface/js

exit 0


mkdir /bld
cd /bld
cmake /src -DCMAKE_TOOLCHAIN_FILE=/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DTEQP_NO_PYTHON=ON -DTEQP_JAVASCRIPT_MODULE=ON -DTEQP_JAVASCRIPT_HTML=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Release -DTEQP_SNIPPETS=ON
cmake --build . --target emtest_catch_exception2
cmake --build . --target bench
cmake --build . --target catch_tests

#node --experimental-wasm-eh emtest_catch_exception2.js
#node --experimental-wasm-eh bench.js
#node --experimental-wasm-eh catch_tests.js

cp emtest_catch_exception2.* /src/interface/js
cp bench.* /src/interface/js
cp catch_tests.* /src/interface/js

exit 0