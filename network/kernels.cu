#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void apply_activation(double* input, double* output, int size, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        switch (act_type) {
            case 0: // Sigmoid
                output[idx] = 1.0 / (1.0 + exp(-input[idx]));
                break;
            case 1: // ReLU
                output[idx] = max(0.0, input[idx]);
                break;
            case 2: // Tanh
                output[idx] = tanh(input[idx]);
                break;
        }
    }
}

__global__ void batch_norm(double* input, double* output, double mean, double variance,
                           double gamma, double beta, double epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gamma * (input[idx] - mean) / sqrt(variance + epsilon) + beta;
    }
}

extern "C" void compute_activation_on_gpu(double* input, double* output, int size, int act_type) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    apply_activation<<<gridSize, blockSize>>>(input, output, size, act_type);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_activation_on_gpu at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    }
}

extern "C" void compute_batch_norm_on_gpu(double* input, double* output, double mean, double variance, double gamma, double beta, double epsilon, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    batch_norm<<<gridSize, blockSize>>>(input, output, mean, variance, gamma, beta, epsilon, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_batch_norm_on_gpu at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    }
}