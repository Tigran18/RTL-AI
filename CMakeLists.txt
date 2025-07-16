cmake_minimum_required(VERSION 3.14)
project(ai_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use pip-installed pybind11's CMake config
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# Add subdirectories
add_subdirectory(network)
add_subdirectory(json_parser)

# Create Python module from bindings.cpp
pybind11_add_module(ai_module bindings.cpp)

# Link C++ components
target_link_libraries(ai_module PRIVATE network json_parser)
