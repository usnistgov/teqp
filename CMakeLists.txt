if(CMAKE_GENERATOR STREQUAL Xcode)
  set(CMAKE_OSX_ARCHITECTURES arm64 CACHE INTERNAL "archs for osx")
endif()

cmake_minimum_required(VERSION 3.16)
project(teqp)
enable_testing()

# Enable /bigobj for MSVC builds
if (MSVC)
    add_compile_options(/bigobj)
    add_compile_options(/permissive-)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

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
        
option (TEQP_COMPLEXSTEP_ENABLED
        "Enable the use of complex-step derivatives for taking first derivatives"
        ON)

option (TEQP_ASAN
        "Enable to pull in the flags needed to use address sanitizer"
        OFF)

# Define -DTEQP_ASAN=ON to enable the option of using address sanitizer of clang
if (TEQP_ASAN)

  # https://stackoverflow.com/a/64294837 (CC BY-SA 4.0)
  if(isMultiConfig)
      if(NOT "Asan" IN_LIST CMAKE_CONFIGURATION_TYPES)
          list(APPEND CMAKE_CONFIGURATION_TYPES Asan)
      endif()
  else()
      set(allowedBuildTypes Asan Debug Release RelWithDebInfo MinSizeRel)
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${allowedBuildTypes}")

      if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
          message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
      endif()
  endif()

  set(CMAKE_C_FLAGS_ASAN
      "${CMAKE_C_FLAGS_RelWithDebInfo} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
      "Flags used by the C compiler for Asan build type or configuration." FORCE)

  set(CMAKE_CXX_FLAGS_ASAN
      "${CMAKE_CXX_FLAGS_RelWithDebInfo} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
      "Flags used by the C++ compiler for Asan build type or configuration." FORCE)

  set(CMAKE_EXE_LINKER_FLAGS_ASAN
      "${CMAKE_EXE_LINKER_FLAGS_RelWithDebInfo} -fsanitize=address" CACHE STRING
      "Linker flags to be used to create executables for Asan build type." FORCE)

  set(CMAKE_SHARED_LINKER_FLAGS_ASAN
      "${CMAKE_SHARED_LINKER_FLAGS_RelWithDebInfo} -fsanitize=address" CACHE STRING
      "Linker lags to be used to create shared libraries for Asan build type." FORCE)

endif()


####  SETUP
set(CMAKE_CXX_STANDARD 20)

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

set(ZIPFN "${CMAKE_CURRENT_SOURCE_DIR}/dev/model_schemas.tar.xz")
if (EXISTS ${ZIPFN})
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xJf ${ZIPFN} 
                  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP")
  message(STATUS "Untarred model schemas")
else()
  message(FATAL_ERROR "tar.xz of thermodynamic model schemas needed for teqp cannot be found")
endif()

add_library(teqpinterface INTERFACE)

target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/multicomplex/include")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann_json")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/boost_teqp")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/REFPROP-interop/include")
target_include_directories(teqpinterface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/externals/stbrumme-hashing/include")

if ((NOT TEQP_NO_TESTS) AND PROJECT_IS_TOP_LEVEL)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch2")
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/json-schema-validator")
set_property(TARGET nlohmann_json_schema_validator PROPERTY POSITION_INDEPENDENT_CODE ON)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen" CACHE INTERNAL "Path to Eigen, for autodiff")
set(EIGEN3_VERSION_OK TRUE CACHE BOOL "Yes eigen is fine")
set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/cmake" CACHE INTERNAL "Path to Eigen, for autodiff")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/cmake" CACHE INTERNAL "Path to Eigen finder module, for autodiff")
set(AUTODIFF_BUILD_TESTS FALSE CACHE BOOL "No autodiff tests")
set(AUTODIFF_BUILD_PYTHON FALSE CACHE BOOL "No autodiff python")
set(AUTODIFF_BUILD_EXAMPLES FALSE CACHE BOOL "No autodiff examples")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff" EXCLUDE_FROM_ALL)

# Turn on more useful diagnostic messages in nlohmann::json, for instance if you are accessing a field that doesn't exist
set(JSON_Diagnostics TRUE CACHE BOOL "Turn on more helpful diagnostics in nlohmann::json")

# Prepare preprocessor macros to disable models
set(preprocessors)
# A comma-delimited, case-insensitive, set of factories to disable"
## set(TEQP_DISABLED_FACTORIES "CPA,SAFTVRMIE,GENERICSAFT,SQUAREWELL,EXP6,2CLJF,MIE,MULTIFLUIDACTIVITY,PCSAFT,GERG200X,MULTIFLUIDASSOCIATION,LKP") # as an example
if (TEQP_DISABLED_FACTORIES)
    string(REGEX MATCHALL "[^,]+" disabled_factories "${TEQP_DISABLED_FACTORIES}")
    message(STATUS "These model factories are disabled: ${disabled_factories}")
    foreach (factory ${disabled_factories})
      string(TOUPPER "${factory}" factory)
      string(STRIP "${factory}" factory)
      list(APPEND preprocessors "#define DISABLE_${factory}")
    endforeach()
    set(PREPROCESSORS_SEMICOLON "${preprocessors}") # CMake requires strings to not contain newlines, so we store with ; instead of newlines
    string (REPLACE ";" "\n" preprocessors "${preprocessors}")
    if (NOT ("${PREPROCESSORS_SEMICOLON}" STREQUAL "$CACHE{TEQP_DISABLED_FACTORIES_PREPROCESSORS}"))
      # Generate the model_flags.hpp header
      message(STATUS "Writing to ${CMAKE_CURRENT_BINARY_DIR}/generated_headers/model_flags.hpp:\n${preprocessors}")
      file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated_headers/model_flags.hpp" "${preprocessors}")
    endif()
    SET(TEQP_DISABLED_FACTORIES_PREPROCESSORS ${PREPROCESSORS_SEMICOLON} CACHE INTERNAL "" FORCE)
else()
    # Only write the HPP file the if it doesn't exist; thereafter, do nothing
    if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/generated_headers/model_flags.hpp")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated_headers/model_flags.hpp" "// No models have been disabled; see the flag TEQP_DISABLED_FACTORIES in CMakeLists.txt for more information")
            message(STATUS "Writing no-op disabling header to ${CMAKE_CURRENT_BINARY_DIR}/generated_headers/model_flags.hpp")
    endif()
endif()


# Single-source the version, either from scikit, or from parsing the pyproject.toml
if (SKBUILD)
    add_definitions("-DTEQPVERSION=\"${SKBUILD_PROJECT_VERSION_FULL}\"")
else()
    file(READ "pyproject.toml" TOML_CONTENT)
    set(REG "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"")
    string(REGEX MATCH "${REG}" VERSION_MATCH "${TOML_CONTENT}")
    if (NOT VERSION_MATCH)
        message(FATAL_ERROR "Can't parse the version")
    else()
        string(REGEX REPLACE "${REG}" "\\1" PROJECT_VERSION_FULL "${VERSION_MATCH}")
        message(STATUS "Version: ${PROJECT_VERSION_FULL}")
        add_definitions("-DTEQPVERSION=\"${PROJECT_VERSION_FULL}\"")
    endif()
endif()

if (NOT TEQP_NO_TEQPCPP)
  # Add a static library with the C++ interface that uses only STL
  # types so that recompilation of a library that uses teqp 
  # doesn't require a full compile for a single LOC change
  file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP/*.cpp")
  add_library(teqpcpp STATIC ${sources})
  target_link_libraries(teqpcpp PUBLIC nlohmann_json_schema_validator PUBLIC teqpinterface PUBLIC autodiff)
  target_include_directories(teqpcpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP")
  target_include_directories(teqpcpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_include_directories(teqpcpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
  target_include_directories(teqpcpp PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/generated_headers")

  set_property(TARGET teqpcpp PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(teqpcpp PRIVATE -DMULTICOMPLEX_NO_MULTIPRECISION)
  target_compile_definitions(teqpcpp PUBLIC -DUSE_AUTODIFF)
  
  # Add the schema file
  target_sources(teqpcpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/interface/CPP/model_schemas.cpp")
  
  # The data files too
  target_sources(teqpcpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/data/FEANN.cpp")
  target_sources(teqpcpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/data/Dufal_assoc.cpp")
  target_sources(teqpcpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/data/PCSAFT.cpp")
  
  target_compile_definitions(teqpcpp PUBLIC -DTEQP_MULTIPRECISION_ENABLED)

  if (WIN32)
    target_compile_options(teqpcpp PUBLIC
       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wno-deprecated-anon-enum-enum-conversion -Wno-deprecated-enum-enum-conversion -Wno-parentheses -Wno-c++98-compat -Wno-c++98-compat-pedantic>
       $<$<CXX_COMPILER_ID:MSVC>:>)
  endif()

  # target_compile_options(teqpcpp PRIVATE
  #    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
  #       -Wall -Wno-c++98-compat -Wno-c++98-compat-pedantic>
  #    $<$<CXX_COMPILER_ID:MSVC>:
  #         /W4 /wd4127 /wd4389 /wd4100>)

  target_compile_options(teqpcpp PUBLIC
     $<$<OR:$<CXX_COMPILER_ID:GNU>>:
         -Wno-class-memaccess>)

  if(MSVC)
    target_compile_options(teqpcpp PRIVATE /W4 ) # or add /WX to turn warnings into errors
    target_compile_options(teqpcpp PUBLIC /wd4127 /wd4389 /wd4100 /wd5054 /wd4702)
  else()
    # Turn on all warnings, and then disable selectively
    target_compile_options(teqpcpp PRIVATE -Wall -Wextra -Wpedantic  -Wno-deprecated-copy -Wno-language-extension-token) # or add -Werror to turn warnings into errors
    target_compile_options(teqpcpp PUBLIC -Wno-deprecated-anon-enum-enum-conversion -Wno-deprecated-enum-enum-conversion -Wno-parentheses)
  endif()

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


if ((NOT TEQP_NO_PYTHON) AND PROJECT_IS_TOP_LEVEL)
    
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11" "pybind11")
    file(GLOB pybind11_files "${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp")
    pybind11_add_module(teqp "${pybind11_files}")
    target_include_directories(teqp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11_json/include")
    target_link_libraries(teqp PRIVATE teqpcpp PRIVATE autodiff PRIVATE teqpinterface )
    target_compile_definitions(teqp PUBLIC -DUSE_AUTODIFF)
    if (MSVC)
      target_compile_options(teqp PRIVATE "/Zm1000")
    endif()
    install(TARGETS teqp LIBRARY DESTINATION teqp)
endif()

if ((NOT TEQP_NO_TESTS) AND PROJECT_IS_TOP_LEVEL)
  file(GLOB catch_tests_files "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cxx")

  # Make all the catch and benchmarking tests
  add_executable(catch_tests "${catch_tests_files}" "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
  if (MSVC)
      target_sources(catch_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()

  # Add all the headers to the project for XCode use
  file(GLOB_RECURSE headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
  target_sources(catch_tests PUBLIC ${headers})

  target_compile_definitions(catch_tests PRIVATE -DTEQPC_CATCH)
  target_compile_definitions(catch_tests PRIVATE -DTEQP_MULTICOMPLEX_ENABLED)
  target_compile_definitions(catch_tests PRIVATE -DTEQP_COMPLEXSTEP_ENABLED)
  target_compile_definitions(catch_tests PRIVATE -DTEQP_MULTIPRECISION_ENABLED)
  target_link_libraries(catch_tests PUBLIC teqpcpp PRIVATE autodiff PRIVATE teqpinterface PRIVATE Catch2WithMain)
  add_test(normal_tests catch_tests)
endif()

if (TEQP_TEQPC)
  # Make a shared extern "C" library
  add_library(teqpc SHARED "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
  if (MSVC)
      target_sources(teqpc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()
  target_compile_definitions(teqpc PRIVATE -DUSE_AUTODIFF)
  target_link_libraries(teqpc PUBLIC teqpcpp PRIVATE autodiff PRIVATE teqpinterface )
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

  if (PROJECT_IS_TOP_LEVEL)

    # And a little testing harness for extern "C" library
    add_executable(teqpc_demo "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/demo.cpp")
    target_link_libraries(teqpc_demo PRIVATE teqpc PRIVATE teqpinterface)

    # And a little benchmarking harness for extern "C" library
    add_executable(teqpc_bench "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc_bench.cpp")
    target_link_libraries(teqpc_bench PRIVATE Catch2WithMain PRIVATE teqpc)
    if(UNIX)
        target_link_libraries (teqpc_bench PRIVATE ${CMAKE_DL_LIBS})
    endif()
  endif()
endif()

### TARGETS from src folder
if (TEQP_SNIPPETS AND PROJECT_IS_TOP_LEVEL)
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

    target_link_libraries(${snippet_exe} PRIVATE teqpcpp PRIVATE autodiff PRIVATE teqpinterface PRIVATE Catch2WithMain )
    if(UNIX)
      target_link_libraries (${snippet_exe} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    target_compile_definitions(${snippet_exe} PRIVATE -DTEQP_MULTICOMPLEX_ENABLED)
    target_compile_definitions(${snippet_exe} PRIVATE -DTEQP_COMPLEXSTEP_ENABLED)
    target_compile_definitions(${snippet_exe} PRIVATE -DTEQP_MULTIPRECISION_ENABLED)
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


if (TEQP_COVERAGE AND PROJECT_IS_TOP_LEVEL)
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
