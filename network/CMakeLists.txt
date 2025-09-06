cmake_minimum_required(VERSION 3.8...3.28)
project(neural_network CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Add include directories
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}  # include current dir for network.h
)

# Create the executable with all source files
add_executable(neural_network
    test.cpp
    network.cpp
    cuda_kernels.cu
)



# Important: let CMake know that network.cpp is a C++ source, cuda_kernels.cu is CUDA
set_target_properties(neural_network PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 11
    CUDA_STANDARD 11
)

# Link CUDA libraries
target_link_libraries(neural_network
    CUDA::cudart
    CUDA::cublas
)
