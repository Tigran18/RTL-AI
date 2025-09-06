#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

extern "C" __global__ void compute_activation_on_gpu(double* input, double* output, int size, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) { // Print from one thread to avoid clutter
        printf("compute_activation_on_gpu: size=%d, act_type=%d\n", size, act_type);
    }
    if (idx < size) {
        double x = input[idx];
        switch (act_type) {
            case 0: // Sigmoid
                output[idx] = 1.0 / (1.0 + exp(-x));
                break;
            case 1: // ReLU
                output[idx] = x > 0.0 ? x : 0.0;
                break;
            case 2: // Tanh
                output[idx] = tanh(x);
                break;
            default:
                printf("Invalid act_type %d at idx %d\n", act_type, idx);
                output[idx] = x; // Fallback
                break;
        }
    }
}

extern "C" __global__ void compute_batch_norm_on_gpu(double* input, double* output, double mean, 
                                                    double variance, double gamma, double beta, 
                                                    double epsilon, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("compute_batch_norm_on_gpu: size=%d, mean=%f, variance=%f\n", size, mean, variance);
    }
    if (idx < size) {
        output[idx] = gamma * (input[idx] - mean) / sqrt(variance + epsilon) + beta;
    }
}