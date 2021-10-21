cmake_minimum_required(VERSION 3.16)
project(teqp)
enable_testing()

# Enable /bigobj for MSVC builds
if (MSVC)
    add_compile_options(-bigobj)
endif()

####  SETUP
set(CMAKE_CXX_STANDARD 17)

set(ZIPFN "${CMAKE_CURRENT_SOURCE_DIR}/dev/boost_bcp_docker/boost_teqp.tar.gz")
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

#set(EIGEN_BUILD_DOC FALSE CACHE BOOL "Turn off Eigen snippets in their docs")
#set(EIGEN_TEST_NOQT TRUE CACHE BOOL "Turn off Eigen tests of Qt")
#set(BUILD_TESTING FALSE CACHE BOOL "Turn off Eigen tests")
#set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen" CACHE INTERNAL "Path to Eigen, for autodiff")
# add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")

set(AUTODIFF_BUILD_TESTS FALSE CACHE BOOL "No autodiff tests")
set(AUTODIFF_BUILD_PYTHON FALSE CACHE BOOL "No autodiff python")
set(AUTODIFF_BUILD_EXAMPLES FALSE CACHE BOOL "No autodiff examples")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff")

# Turn on more useful diagnostic messages in nlohmann::json, for instance if you are accessing a field that doesn't exist
set(JSON_Diagnostics TRUE CACHE BOOL "Turn on more helpful diagnostics in nlohmann::json")


set(TEQP_NO_PYTHON false)
if (NOT TEQP_NO_PYTHON)
    add_definitions(-DUSE_AUTODIFF)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11" "pybind11")
    file(GLOB pybind11_files "${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp")
    pybind11_add_module(teqp "${pybind11_files}")
    target_include_directories(teqp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11_json/include")
    target_link_libraries(teqp PRIVATE autodiff)
endif()

file(GLOB catch_tests_files "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cxx")

add_executable(catch_tests "${catch_tests_files}")
if (MSVC)
    target_sources(catch_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
target_link_libraries(catch_tests autodiff)
add_test(normal_tests catch_tests)

add_executable(teqpc "${CMAKE_CURRENT_SOURCE_DIR}/interface/C/teqpc.cpp")
if (MSVC)
    target_sources(teqpc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
target_link_libraries(teqpc autodiff)
target_compile_definitions(teqpc PRIVATE -DTEQPC_CATCH)
add_test(teqpc_tests teqpc)

### TARGETS from src folder
if (TEQP_SNIPPETS)
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
