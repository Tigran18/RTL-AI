cmake_minimum_required(VERSION 3.18)
project(network_lib LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.Embed)

# Add pybind11 CMake path if installed in virtual environment
list(APPEND CMAKE_PREFIX_PATH "C:/Users/HP/Documents/Synopsys/Diploma_project/RTL-AI/venv311/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# Ignore MinGW paths
set(CMAKE_IGNORE_PATH "C:/msys64/mingw64;C:/msys64/usr")

# Define source files
set(CUDA_SOURCES cuda_kernels.cu)
set(CPP_SOURCES network.cpp bindings.cpp)

# Create the Python module
pybind11_add_module(network_lib ${CUDA_SOURCES} ${CPP_SOURCES})

# Set include directories for the target
target_include_directories(network_lib PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Set C++ and CUDA standards
set_target_properties(network_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75 # Adjust for your GPU (e.g., 75 for Turing, 86 for Ampere)
)

# Link libraries
target_link_libraries(network_lib PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    Python::Python
)

# Platform-specific settings
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/MD,/EHsc,/NODEFAULTLIB:LIBCMT")
    target_compile_definitions(network_lib PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        NOMINMAX
    )
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --disable-warnings")
endif()

# Set output directories and suffix
set_target_properties(network_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    SUFFIX ".pyd"
)

# Ensure MSVC is used as the CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64/cl.exe")