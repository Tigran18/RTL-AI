add_library(network STATIC network.cpp)
target_compile_options(network PRIVATE -fPIC)
target_include_directories(network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
