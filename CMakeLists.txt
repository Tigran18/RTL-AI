cmake_minimum_required(VERSION 3.14)
project(ai_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# Add subdirectories
add_subdirectory(network)
add_subdirectory(json_parser)

# Create Python module from bindings.cpp
pybind11_add_module(ai_module bindings.cpp)

# Link C++ components
target_link_libraries(ai_module PRIVATE network json_parser)
