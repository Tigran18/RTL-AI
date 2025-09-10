cmake_minimum_required(VERSION 3.22)
project(GPU_Network LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 (and instruct it to find Python too)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Prefer modern CUDAToolkit; fall back to legacy FindCUDA if necessary
find_package(CUDAToolkit QUIET)
if (NOT CUDAToolkit_FOUND)
    find_package(CUDA REQUIRED)
    set(USE_OLD_FINDCUDA TRUE)
else()
    set(USE_OLD_FINDCUDA FALSE)
endif()

# Network library (CPU part)
add_library(network STATIC
    src/network.cpp
)
target_include_directories(network PUBLIC src)

# Build Python module and compile CUDA kernels directly into it
pybind11_add_module(py_network
    python_bindings/py_network.cpp
    cuda/cuda_kernels.cu        # compile the .cu directly into the module
)

# Enable separable compilation so device code links properly across objects
set_target_properties(py_network PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Force the output filename to "py_network" (so import py_network works)
set_target_properties(py_network PROPERTIES OUTPUT_NAME "py_network")

# Link against our CPU network code and CUDA libraries (cuBLAS and cudart)
if (USE_OLD_FINDCUDA)
    target_link_libraries(py_network PRIVATE network cublas cudart)
else()
    # modern CUDAToolkit provides imported targets
    target_link_libraries(py_network PRIVATE network CUDA::cublas CUDA::cudart)
endif()

# Post-build: copy the .pyd into the tests/ folder so tests can import without path fiddling
add_custom_command(TARGET py_network POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:py_network> ${CMAKE_SOURCE_DIR}/tests/
)

# Post-build: copy core CUDA DLLs to tests/ (adjust CUDA_BIN_DIR if your CUDA is elsewhere)
set(CUDA_BIN_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin" CACHE PATH "Path to CUDA bin directory")
foreach(_dll IN ITEMS cudart64_12.dll cublas64_12.dll cublasLt64_12.dll)
    add_custom_command(TARGET py_network POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDA_BIN_DIR}/${_dll}"
            ${CMAKE_SOURCE_DIR}/tests/
    )
endforeach()
