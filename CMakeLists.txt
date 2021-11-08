cmake_minimum_required(VERSION 3.16)
project(teqp)
enable_testing()

# Enable /bigobj for MSVC builds
if (MSVC)
    add_compile_options(-bigobj)
endif()

####  SETUP
set(CMAKE_CXX_STANDARD 17)

set(ZIPFN "${CMAKE_CURRENT_SOURCE_DIR}/dev/docker/boost_bcp_docker/boost_teqp.tar.gz")
set(OUTFN "${CMAKE_CURRENT_SOURCE_DIR}/boost_teqp/boost/version.hpp")
if (EXISTS ${ZIPFN})
    if(NOT EXISTS ${OUTFN})
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${ZIPFN}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        message(STATUS "Untarred boost headers")
    endif()
else()
    message(FATAL_ERROR "tar.gz of boost sources needed for teqp cannot be found")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/multicomplex/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann_json")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/boost_teqp")

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen" CACHE INTERNAL "Path to Eigen, for autodiff")
set(EIGEN3_VERSION_OK TRUE CACHE BOOL "Yes eigen is fine")
set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/cmake" CACHE INTERNAL "Path to Eigen, for autodiff")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/cmake" CACHE INTERNAL "Path to Eigen finder module, for autodiff")
set(AUTODIFF_BUILD_TESTS FALSE CACHE BOOL "No autodiff tests")
set(AUTODIFF_BUILD_PYTHON FALSE CACHE BOOL "No autodiff python")
set(AUTODIFF_BUILD_EXAMPLES FALSE CACHE BOOL "No autodiff examples")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff")

# Find eigen library
find_package(Eigen3 REQUIRED)

# Turn on more useful diagnostic messages in nlohmann::json, for instance if you are accessing a field that doesn't exist
set(JSON_Diagnostics TRUE CACHE BOOL "Turn on more helpful diagnostics in nlohmann::json")

option (TEQP_NO_PYTHON
        "Enable to NOT build the Python interface"
        OFF)

option (TEQP_TEQPC
        "Enable to build the shared library with extern \"C\" interface"
        OFF)

option (TEQP_COVERAGE
        "Enable to build the GCOV tests of the catch tests"
        OFF)

if (NOT TEQP_NO_PYTHON)
    add_definitions(-DUSE_AUTODIFF)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11" "pybind11")
    file(GLOB pybind11_files "${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp")
    pybind11_add_module(teqp "${pybind11_files}")
    target_include_directories(teqp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11_json/include")
    target_link_libraries(teqp PRIVATE autodiff)
endif()

file(GLOB catch_tests_files "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cxx")

# Make all the catch and benchmarking tests
add_executable(catch_tests "${catch_tests_files}" "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
if (MSVC)
    target_sources(catch_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
target_compile_definitions(catch_tests PRIVATE -DTEQPC_CATCH)
target_link_libraries(catch_tests autodiff)
add_test(normal_tests catch_tests)

if (TEQP_TEQPC)
  # Make a shared extern "C" library
  add_library(teqpc SHARED "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
  if (MSVC)
      target_sources(teqpc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()
  target_link_libraries(teqpc autodiff)
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
  target_link_libraries(teqpc_demo teqpc)

  # And a little benchmarking harness for extern "C" library
  add_executable(teqpc_bench "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc_bench.cpp")
  target_link_libraries(teqpc_bench teqpc)
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

    target_link_libraries(${snippet_exe} autodiff)
    if(UNIX)
      target_link_libraries (${snippet_exe} ${CMAKE_DL_LIBS})
    endif()
  endforeach()
else()
  message(STATUS "No snippets will be compiled, pass -DTEQP_SNIPPETS=ON to build them")
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