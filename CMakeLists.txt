set(CMAKE_OSX_ARCHITECTURES x86_64;arm64 CACHE INTERNAL "archs for osx")
cmake_minimum_required(VERSION 3.16)
project(teqp)
enable_testing()

# Enable /bigobj for MSVC builds
if (MSVC)
    add_compile_options(-bigobj)
endif()

####  SETUP
set(CMAKE_CXX_STANDARD 17)

set(ZIPFN "${CMAKE_CURRENT_SOURCE_DIR}/dev/docker/boost_bcp_docker/boost_teqp.tar.xz")
set(OUTFN "${CMAKE_CURRENT_SOURCE_DIR}/boost_teqp/boost/version.hpp")
if (EXISTS ${ZIPFN})
    if(NOT EXISTS ${OUTFN})
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xJf ${ZIPFN}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        message(STATUS "Untarred boost headers")
    endif()
else()
    message(FATAL_ERROR "tar.xz of boost sources needed for teqp cannot be found")
endif()

add_library(teqpinterface INTERFACE)

target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/multicomplex/include")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann_json")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/boost_teqp")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch2")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/json-schema-validator")

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen" CACHE INTERNAL "Path to Eigen, for autodiff")
set(EIGEN3_VERSION_OK TRUE CACHE BOOL "Yes eigen is fine")
set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/cmake" CACHE INTERNAL "Path to Eigen, for autodiff")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/cmake" CACHE INTERNAL "Path to Eigen finder module, for autodiff")
set(AUTODIFF_BUILD_TESTS FALSE CACHE BOOL "No autodiff tests")
set(AUTODIFF_BUILD_PYTHON FALSE CACHE BOOL "No autodiff python")
set(AUTODIFF_BUILD_EXAMPLES FALSE CACHE BOOL "No autodiff examples")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff")

# Turn on more useful diagnostic messages in nlohmann::json, for instance if you are accessing a field that doesn't exist
set(JSON_Diagnostics TRUE CACHE BOOL "Turn on more helpful diagnostics in nlohmann::json")



option (TEQP_NO_PYTHON
        "Enable to NOT build the Python interface"
        OFF)

option (TEQP_NO_TESTS
        "Enable to NOT include the test targets at all"
        OFF)

option (TEQP_NO_TEQPCPP
        "Enable to NOT include the teqp C++ wrapper"
        OFF)

option (TEQP_TESTTEQPCPP
        "Enable to add a target with a test of the C++ interface"
        OFF)

option (TEQP_TEQPC
        "Enable to build the shared library with extern \"C\" interface"
        OFF)

option (TEQP_COVERAGE
        "Enable to build the GCOV tests of the catch tests"
        OFF)

option (TEQP_JAVASCRIPT_HTML
        "Enable to generate HTML files rather than js with the emscripten engine"
        OFF)

option (TEQP_JAVASCRIPT_EMBIND
        "Build C++ <-> Javascript interface with embind"
        OFF)

option (TEQP_MULTIPRECISION_ENABLED
        "Enable the use of boost::multiprecision"
        OFF)
        
option (TEQP_MULTICOMPLEX_ENABLED
        "Enable the use of multi-complex arithmetic for taking derivatives"
        OFF)

if (NOT TEQP_NO_TEQPCPP)
  # Add a static library with the C++ interface that uses only STL
  # types so that recompilation of a library that uses teqp 
  # doesn't require a full compile for a single LOC change
  file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP/*.cpp")
  add_library(teqpcpp STATIC ${sources})
  target_link_libraries(teqpcpp PUBLIC nlohmann_json_schema_validator PUBLIC teqpinterface PUBLIC autodiff)
  target_include_directories(teqpcpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP")
  set_property(TARGET teqpcpp PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(teqpcpp PRIVATE -DMULTICOMPLEX_NO_MULTIPRECISION)
  target_compile_definitions(teqpcpp PUBLIC -DUSE_AUTODIFF)
  
  # Populate the model schema cpp file with the contents
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/notebooks/schemas.json" MODEL_SCHEMA_CONTENTS)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/dev/templates/model_schema.cpp.in" MODEL_JSON_SCHEMA_TEMPLATE)
  file(CONFIGURE
       OUTPUT model_schemas.cpp
       CONTENT ${MODEL_JSON_SCHEMA_TEMPLATE}
       @ONLY)
  target_sources(teqpcpp PRIVATE model_schemas.cpp)

  if (TEQP_TESTTEQPCPP)
    add_executable(test_teqpcpp "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP/test/test_teqpcpp.cpp")
    target_link_libraries(test_teqpcpp PUBLIC teqpcpp)
    add_executable(bench_teqpcpp "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP/test/bench_teqpcpp.cpp")
    target_link_libraries(bench_teqpcpp PUBLIC teqpcpp PRIVATE Catch2WithMain)
  endif()
endif()

if (TEQP_JAVASCRIPT_MODULE)
  # cmake -DTEQP_JAVASCRIPT_MODULE=ON
  #       -DCMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN}/cmake/Platform/Emscripten.cmake
  #       

  add_compile_options( -sDISABLE_EXCEPTION_CATCHING=0)
  add_link_options( -sDISABLE_EXCEPTION_CATCHING=0 -sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=1)

  #add_compile_options( -gsource-map -fsanitize=address)
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -gsource-map -fsanitize=address ")
endif()

if (TEQP_MULTIPRECISION_ENABLED)
    add_definitions(-DTEQP_MULTIPRECISION_ENABLED)
endif()

if (TEQP_MULTICOMPLEX_ENABLED)
    add_definitions(-DTEQP_MULTICOMPLEX_ENABLED)
endif()


if (NOT TEQP_NO_PYTHON)
    
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11" "pybind11")
    file(GLOB pybind11_files "${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp")
    pybind11_add_module(teqp "${pybind11_files}")
    target_include_directories(teqp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11_json/include")
    target_link_libraries(teqp PUBLIC nlohmann_json_schema_validator PRIVATE autodiff PRIVATE teqpinterface PRIVATE teqpcpp)
    target_compile_definitions(teqp PUBLIC -DUSE_AUTODIFF)
    if (MSVC)
      target_compile_options(teqp PRIVATE "/Zm1000")
    endif()
endif()

if (NOT TEQP_NO_TESTS)
  file(GLOB catch_tests_files "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cxx")

  # Make all the catch and benchmarking tests
  add_executable(catch_tests "${catch_tests_files}" "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
  if (MSVC)
      target_sources(catch_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()
  target_compile_definitions(catch_tests PRIVATE -DTEQPC_CATCH)
  target_compile_definitions(catch_tests PRIVATE -DTEQP_MULTICOMPLEX_ENABLED)
  target_compile_definitions(catch_tests PRIVATE -DTEQP_MULTIPRECISION_ENABLED)
  target_link_libraries(catch_tests PUBLIC nlohmann_json_schema_validator PRIVATE autodiff PRIVATE teqpinterface PRIVATE Catch2WithMain PUBLIC teqpcpp)
  add_test(normal_tests catch_tests)
endif()

if (TEQP_TEQPC)
  # Make a shared extern "C" library
  add_library(teqpc SHARED "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
  if (MSVC)
      target_sources(teqpc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()
  target_compile_definitions(teqpc PRIVATE -DUSE_AUTODIFF)
  target_link_libraries(teqpc PRIVATE autodiff PRIVATE teqpinterface PUBLIC teqpcpp)
  if (NOT UNIX)
    target_compile_definitions(teqpc PRIVATE -DEXTERN_C_DLLEXPORT)
  else()
    target_compile_definitions(teqpc PRIVATE -DEXTERN_C)
  endif()
  if(MSVC)
    add_custom_command(
      TARGET teqpc
      POST_BUILD
      COMMAND dumpbin /EXPORTS $<TARGET_FILE:teqpc> > ${CMAKE_CURRENT_BINARY_DIR}/exports.txt)
  endif()

  # And a little testing harness for extern "C" library
  add_executable(teqpc_demo "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/demo.cpp")
  target_link_libraries(teqpc_demo PRIVATE teqpc PRIVATE teqpinterface)

  # And a little benchmarking harness for extern "C" library
  add_executable(teqpc_bench "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc_bench.cpp")
  target_link_libraries(teqpc_bench PRIVATE teqpc PRIVATE teqpinterface PRIVATE Catch2WithMain)
endif()

### TARGETS from src folder
if (TEQP_SNIPPETS)
  add_definitions(-DUSE_TEQP_HMX)
  # Collect all the snippets
  file(GLOB_RECURSE snippets "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

  message(STATUS "snippets found = ${snippets}")
  foreach (snippet ${snippets})

    get_filename_component(snippet_name ${snippet} NAME)
    get_filename_component(snippet_exe ${snippet} NAME_WE)
    message(STATUS "snippet_name = ${snippet_name}")

    add_executable(${snippet_exe} ${snippet})
    if (MSVC)
      target_sources(${snippet_exe} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
    endif()

    target_link_libraries(${snippet_exe} PRIVATE nlohmann_json_schema_validator PRIVATE autodiff PRIVATE teqpinterface PRIVATE Catch2WithMain PRIVATE teqpcpp)
    if(UNIX)
      target_link_libraries (${snippet_exe} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    target_compile_definitions(${snippet_exe} PRIVATE -DTEQP_MULTICOMPLEX_ENABLED)
    target_compile_definitions(${snippet_exe} PRIVATE -DUSE_AUTODIFF)

    if(TEQP_JAVASCRIPT_HTML)
      # All the generated executables will compile to HTML with no prefix and file extension of HTML
      set_target_properties(${snippet_exe} PROPERTIES PREFIX "" SUFFIX .html)
    endif()
  endforeach()
else()
  message(STATUS "No snippets will be compiled, pass -DTEQP_SNIPPETS=ON to build them")
endif()

if (TEQP_EMBIND)
    # If you want a monolithic file with no async memory loading, define EMSCRIPTEN_NO_MEMORY_INIT_FILE
    if(EMSCRIPTEN_NO_MEMORY_INIT_FILE)
      set(EMSCRIPTEN_INIT_FLAG "--memory-init-file 0")
    else()
      set(EMSCRIPTEN_INIT_FLAG "--memory-init-file 1")
    endif()
    add_definitions( -sDISABLE_EXCEPTION_CATCHING=0)
    set(CMAKE_EXE_LINKER_FLAGS "--bind ${EMSCRIPTEN_INIT_FLAG} -sASSERTIONS=1 -sDISABLE_EXCEPTION_CATCHING=0")
    set(CMAKE_BUILD_TYPE Release)
    set(APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/interface/js/emscripten_interface.cxx")
    add_executable(teqpbind ${APP_SOURCES})
    target_link_libraries(teqpbind PRIVATE autodiff PRIVATE teqpinterface)
    SET_TARGET_PROPERTIES(teqpbind PROPERTIES PREFIX "" SUFFIX .js)
endif()

if (TEQP_EMBIND_MODULARIZE_ES6)

    # If you want a monolithic file with no async memory loading, define EMSCRIPTEN_NO_MEMORY_INIT_FILE
    if(EMSCRIPTEN_NO_MEMORY_INIT_FILE)
      set(EMSCRIPTEN_INIT_FLAG "--memory-init-file 0")
    else()
      set(EMSCRIPTEN_INIT_FLAG "--memory-init-file 1")
    endif()
    add_definitions( -sDISABLE_EXCEPTION_CATCHING=0)
    set(CMAKE_EXE_LINKER_FLAGS "--bind ${EMSCRIPTEN_INIT_FLAG} -sASSERTIONS=1 -sDISABLE_EXCEPTION_CATCHING=0 -s EXPORTED_RUNTIME_METHODS=['isotherm'] -s EXPORT_ES6=1 -s MODULARIZE=1 -s USE_ES6_IMPORT_META=0")
    set(CMAKE_BUILD_TYPE Release)
    set(APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/interface/js/emscripten_interface.cxx")
    add_executable(teqpbind ${APP_SOURCES})
    target_link_libraries(teqpbind PRIVATE autodiff PRIVATE teqpinterface)
    SET_TARGET_PROPERTIES(teqpbind PROPERTIES PREFIX "" SUFFIX .js)
endif()


if (TEQP_COVERAGE)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "This path")
  # See also http://stackoverflow.com/a/16536401 (a detailed guide on using gcov with cmake)
  include(CodeCoverage)
  append_coverage_compiler_flags()
  setup_target_for_coverage_gcovr_html(
    NAME teqp_coverage
    EXECUTABLE catch_tests
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  )
endif()
