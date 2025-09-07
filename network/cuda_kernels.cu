#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <cmath>

// Fused bias addition and activation kernel
__global__ void fused_bias_activation_kernel(float* input, float* biases, float* output, int batch_size, int num, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch_size * num;
    if (idx < size) {
        int row = idx / num; // Batch index
        int col = idx % num; // Neuron index
        float val = input[idx];
        if (biases) val += biases[col];
        switch (act_type) {
            case 0: output[idx] = 1.0f / (1.0f + expf(-val)); break; // Sigmoid
            case 1: output[idx] = fmaxf(0.0f, val); break; // ReLU
            case 2: output[idx] = tanhf(val); break; // Tanh
        }
    }
}

extern "C" void fused_bias_activation_on_gpu(float* input, float* biases, float* output, int batch_size, int num, int act_type) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    fused_bias_activation_kernel<<<gridSize, blockSize>>>(input, biases, output, batch_size, num, act_type);
}

// Optimized batch norm stats with shared memory
__global__ void compute_batch_norm_stats_batched(float* input, float* mean, float* variance, 
                                                int batch_size, int num_features) {
    extern __shared__ float shared[];
    int j = blockIdx.x;
    int tid = threadIdx.x;
    float* sdata = shared;

    if (j < num_features) {
        float sum = 0.0f;
        for (int i = tid; i < batch_size; i += blockDim.x) {
            sum += input[i * num_features + j];
        }
        sdata[tid] = sum;
        __syncthreads();

        // Reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) mean[j] = sdata[0] / batch_size;

        float var = 0.0f;
        for (int i = tid; i < batch_size; i += blockDim.x) {
            float diff = input[i * num_features + j] - mean[j];
            var += diff * diff;
        }
        sdata[tid] = var;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) variance[j] = (batch_size > 1) ? sdata[0] / (batch_size - 1) : 0.0f;
    }
}

extern "C" void compute_batch_norm_stats_batched_on_gpu(float* input, float* mean, float* variance, 
                                                       int batch_size, int num_features, int block_size) {
    int gridSize = num_features;
    compute_batch_norm_stats_batched<<<gridSize, block_size, block_size * sizeof(float)>>>(input, mean, variance, batch_size, num_features);
}

// Batch normalization application
__global__ void batch_norm_batched(float* input, float* output, float* mean, float* variance, 
                                   float* gamma, float* beta, float epsilon, int batch_size, 
                                   int num_features, float* hat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch_size * num_features;
    if (idx < size) {
        int i = idx / num_features; // Batch index
        int j = idx % num_features; // Feature index
        float stddev = sqrtf(variance[j] + epsilon);
        float h = (input[idx] - mean[j]) / stddev;
        if (hat) hat[idx] = h;
        output[idx] = gamma[j] * h + beta[j];
    }
}

extern "C" void compute_batch_norm_batched_on_gpu(float* input, float* output, float* mean, 
                                                 float* variance, float* gamma, float* beta, 
                                                 float epsilon, int batch_size, int num_features, 
                                                 float* hat) {
    int blockSize = 256;
    int gridSize = (batch_size * num_features + blockSize - 1) / blockSize;
    batch_norm_batched<<<gridSize, blockSize>>>(input, output, mean, variance, gamma, beta, 
                                                epsilon, batch_size, num_features, hat);
}

// Parameter update kernel
__global__ void update_parameters_kernel(float* weights, float* weight_gradients, float* weight_velocity,
                                         float* biases, float* bias_gradients, float* bias_velocity,
                                         float* gammas, float* gamma_gradients,
                                         float* betas, float* beta_gradients,
                                         float learning_rate, float momentum,
                                         int num_weights, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_weights) {
        weight_velocity[idx] = momentum * weight_velocity[idx] - learning_rate * weight_gradients[idx];
        weights[idx] += weight_velocity[idx];
    }
    if (idx < num) {
        bias_velocity[idx] = momentum * bias_velocity[idx] - learning_rate * bias_gradients[idx];
        biases[idx] += bias_velocity[idx];
        if (gammas) {
            gammas[idx] -= learning_rate * gamma_gradients[idx];
            betas[idx] -= learning_rate * beta_gradients[idx];
        }
    }
}

extern "C" void update_parameters_on_gpu(float* weights, float* weight_gradients, float* weight_velocity,
                                         float* biases, float* bias_gradients, float* bias_velocity,
                                         float* gammas, float* gamma_gradients,
                                         float* betas, float* beta_gradients,
                                         float learning_rate, float momentum,
                                         int num_weights, int num, bool use_bn) {
    int blockSize = 256;
    int gridSize = (max(num_weights, num) + blockSize - 1) / blockSize;
    update_parameters_kernel<<<gridSize, blockSize>>>(weights, weight_gradients, weight_velocity,
                                                      biases, bias_gradients, bias_velocity,
                                                      gammas, gamma_gradients,
                                                      betas, beta_gradients,
                                                      learning_rate, momentum,
                                                      num_weights, num);
}

// Other kernels remain similar, updated to float
__global__ void compute_output_delta_batched(float* outputs, float* pre_acts, float* targets, 
                                            float* deltas, int batch_size, int num, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        int i = idx / num;
        int j = idx % num;
        float o = outputs[idx];
        float p = pre_acts[idx];
        float error = o - targets[idx];
        float deriv;
        switch (act_type) {
            case 0: {
                float sig = 1.0f / (1.0f + expf(-p));
                deriv = sig * (1.0f - sig);
                break;
            }
            case 1: deriv = p > 0 ? 1.0f : 0.0f; break;
            case 2: {
                float t = tanhf(p);
                deriv = 1.0f - t * t;
                break;
            }
        }
        deltas[idx] = error * deriv;
    }
}

extern "C" void compute_output_delta_batched_on_gpu(float* outputs, float* pre_acts, float* targets, 
                                                   float* deltas, int batch_size, int num, int act_type) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    compute_output_delta_batched<<<gridSize, blockSize>>>(outputs, pre_acts, targets, deltas, 
                                                          batch_size, num, act_type);
}

__global__ void compute_hidden_delta_batched(float* delta_next, float* weights_next, float* errors, 
                                            int batch_size, int num_current, int num_next) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch_size * num_current;
    if (idx < size) {
        int i = idx / num_current;
        int j = idx % num_current;
        float sum = 0.0f;
        for (int k = 0; k < num_next; ++k) {
            sum += delta_next[i * num_next + k] * weights_next[j * num_next + k];
        }
        errors[idx] = sum;
    }
}

extern "C" void compute_hidden_delta_batched_on_gpu(float* delta_next, float* weights_next, 
                                                   float* errors, int batch_size, int num_current, 
                                                   int num_next) {
    int blockSize = 256;
    int gridSize = (batch_size * num_current + blockSize - 1) / blockSize;
    compute_hidden_delta_batched<<<gridSize, blockSize>>>(delta_next, weights_next, errors, 
                                                          batch_size, num_current, num_next);
}

__global__ void compute_bn_grad_batched(float* error, float* hat, float* gamma_grad, 
                                       float* beta_grad, int batch_size, int num) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num) {
        float sum_gamma = 0.0f;
        float sum_beta = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            float e = error[i * num + j];
            sum_gamma += e * hat[i * num + j];
            sum_beta += e;
        }
        gamma_grad[j] = sum_gamma;
        beta_grad[j] = sum_beta;
    }
}

extern "C" void compute_bn_grad_batched_on_gpu(float* error, float* hat, float* gamma_grad, 
                                              float* beta_grad, int batch_size, int num) {
    int blockSize = 256;
    int gridSize = (num + blockSize - 1) / blockSize;
    compute_bn_grad_batched<<<gridSize, blockSize>>>(error, hat, gamma_grad, beta_grad, 
                                                     batch_size, num);
}

__global__ void update_error_for_bn(float* error, float* gamma, float* variance, float epsilon, 
                                    int batch_size, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        int j = idx % num;
        float stddev = sqrtf(variance[j] + epsilon);
        error[idx] *= gamma[j] / stddev;
    }
}

extern "C" void update_error_for_bn_on_gpu(float* error, float* gamma, float* variance, 
                                          float epsilon, int batch_size, int num) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    update_error_for_bn<<<gridSize, blockSize>>>(error, gamma, variance, epsilon, batch_size, num);
}

__global__ void compute_delta_from_error(float* error, float* pre_act, float* delta, 
                                        int batch_size, int num, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        float e = error[idx];
        float p = pre_act[idx];
        float deriv;
        switch (act_type) {
            case 0: {
                float sig = 1.0f / (1.0f + expf(-p));
                deriv = sig * (1.0f - sig);
                break;
            }
            case 1: deriv = p > 0 ? 1.0f : 0.0f; break;
            case 2: {
                float t = tanhf(p);
                deriv = 1.0f - t * t;
                break;
            }
        }
        delta[idx] = e * deriv;
    }
}

extern "C" void compute_delta_from_error_on_gpu(float* error, float* pre_act, float* delta, 
                                               int batch_size, int num, int act_type) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    compute_delta_from_error<<<gridSize, blockSize>>>(error, pre_act, delta, batch_size, num, act_type);
}