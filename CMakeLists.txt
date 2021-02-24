project(teqp)

set(CMAKE_CXX_STANDARD 17)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/pymcx/include")

add_executable(main "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

