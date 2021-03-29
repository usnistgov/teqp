cmake_minimum_required(VERSION 3.19)
project(teqp)

####  SETUP
set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/pymcx/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann_json")

set(BUILD_TESTING FALSE CACHE BOOL "Turn off Eigen tests")
set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen" CACHE INTERNAL "Path to Eigen, for autodiff")

set(AUTODIFF_BUILD_TESTS FALSE CACHE BOOL "No autodiff tests")
set(AUTODIFF_BUILD_PYTHON FALSE CACHE BOOL "No autodiff python")
set(AUTODIFF_BUILD_EXAMPLES FALSE CACHE BOOL "No autodiff examples")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff")

### TARGETS
add_executable(main "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(main autodiff)

add_executable(trace "${CMAKE_CURRENT_SOURCE_DIR}/src/trace_critical.cpp")
if (MSVC)
    target_sources(trace PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
target_link_libraries(trace autodiff)

add_executable(multifluid "${CMAKE_CURRENT_SOURCE_DIR}/src/multifluid.cpp")
if (MSVC)
    target_sources(multifluid PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
target_link_libraries(multifluid autodiff)

# add_executable(test_autodiff "${CMAKE_CURRENT_SOURCE_DIR}/src/test_autodiff.cpp")
# if (MSVC)
#     target_sources(test_autodiff PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
# endif()
# target_link_libraries(test_autodiff autodiff)

add_executable(catch_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/catch_tests.cxx")
if (MSVC)
    target_sources(catch_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()
target_link_libraries(catch_tests autodiff)