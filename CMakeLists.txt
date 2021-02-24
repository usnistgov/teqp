project(teqp)

set(CMAKE_CXX_STANDARD 17)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/pymcx/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")

add_executable(main "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_executable(trace "${CMAKE_CURRENT_SOURCE_DIR}/src/trace_critical.cpp")
if (MSVC)
    target_sources(trace PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()