cmake_minimum_required(VERSION 3.14...3.28)
project(ai_module LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure Visual Studio headers are used
include_directories(BEFORE "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/include")

# Find Python and pybind11
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

# -----------------------------
# Temporarily disable Python module creation
# -----------------------------
# pybind11_add_module(ai_module bindings.cpp)
# target_include_directories(ai_module BEFORE PRIVATE "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/include")
# target_link_libraries(ai_module PRIVATE network json_parser)
