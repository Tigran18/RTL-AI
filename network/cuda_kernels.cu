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

__global__ void compute_layer_output(double* weights, double* inputs, double* biases,
                                    double* output, int num_neurons, int num_inputs,
                                    int act_type) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons) {
        double sum = 0.0;
        for (int i = 0; i < num_inputs; ++i) {
            sum += weights[neuron_idx * num_inputs + i] * inputs[i];
        }
        sum += biases[neuron_idx];
        output[neuron_idx] = sum; // Store pre-activation value
        switch (act_type) {
            case 0: // Sigmoid
                output[neuron_idx] = 1.0 / (1.0 + exp(-sum));
                break;
            case 1: // ReLU
                output[neuron_idx] = max(0.0, sum);
                break;
            case 2: // Tanh
                output[neuron_idx] = tanh(sum);
                break;
        }
    }
}

__global__ void compute_batch_norm_stats(double* input, double* mean, double* variance,
                                        int size, double epsilon) {
    extern __shared__ double shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load input to shared memory
    shared_data[tid] = (idx < size) ? input[idx] : 0.0;
    __syncthreads();

    // Compute mean
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < blockDim.x && i < size; ++i) {
            sum += shared_data[i];
        }
        *mean = size > 0 ? sum / size : 0.0;
    }
    __syncthreads();

    // Compute variance
    if (tid == 0) {
        double var = 0.0;
        for (int i = 0; i < blockDim.x && i < size; ++i) {
            var += (shared_data[i] - *mean) * (shared_data[i] - *mean);
        }
        *variance = size > 1 ? var / (size - 1) : 1.0;
    }
}

extern "C" void compute_activation_on_gpu(double* input, double* output, int size, int act_type) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    apply_activation<<<gridSize, blockSize>>>(input, output, size, act_type);
}

extern "C" void compute_batch_norm_on_gpu(double* input, double* output, double mean, double variance,
                                         double gamma, double beta, double epsilon, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    batch_norm<<<gridSize, blockSize>>>(input, output, mean, variance, gamma, beta, epsilon, size);
}

extern "C" void compute_layer_output_on_gpu(double* weights, double* inputs, double* biases,
                                           double* output, int num_neurons, int num_inputs, int act_type) {
    int blockSize = 256;
    int gridSize = (num_neurons + blockSize - 1) / blockSize;
    compute_layer_output<<<gridSize, blockSize>>>(weights, inputs, biases, output, num_neurons, num_inputs, act_type);
}

extern "C" void compute_batch_norm_stats_on_gpu(double* input, double* mean, double* variance,
                                               int size, double epsilon) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    compute_batch_norm_stats<<<gridSize, blockSize, blockSize * sizeof(double)>>>(input, mean, variance, size, epsilon);
}