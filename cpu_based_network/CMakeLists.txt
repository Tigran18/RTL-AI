cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure default build type is Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Tell CMake where to find Python 3.11
set(PYTHON_EXECUTABLE "C:/Users/HP/AppData/Local/Programs/Python/Python311/python.exe")

# Find pybind11
find_package(pybind11 REQUIRED)

# Include headers
include_directories(include)

# Source files
set(SOURCES
    src/network.cpp
    python_bindings/py_network.cpp
)

# Build Python module
pybind11_add_module(py_network ${SOURCES})

# Set output directory for the .pyd
set_target_properties(py_network PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
)
