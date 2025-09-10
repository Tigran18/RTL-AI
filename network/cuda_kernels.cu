#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_activation_kernel(float* input, float* biases, float* output, int batch_size, int num, int act_type, float* pre_act) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        int j = idx % num;
        float val = input[idx] + biases[j];
        if (pre_act) pre_act[idx] = val;
        if (act_type == 0) { // Sigmoid
            output[idx] = 1.0f / (1.0f + expf(-val));
        } else if (act_type == 1) { // ReLU
            output[idx] = fmaxf(val, 0.0f);
        } else if (act_type == 2) { // Tanh
            output[idx] = tanhf(val);
        }
    }
}

__global__ void compute_batch_norm_stats_batched_kernel(float* input, float* mean, float* variance, int batch_size, int num_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num_features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += input[i * num_features + j];
        }
        mean[j] = sum / batch_size;
        float var_sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            float diff = input[i * num_features + j] - mean[j];
            var_sum += diff * diff;
        }
        variance[j] = var_sum / batch_size;
    }
}

__global__ void compute_batch_norm_batched_kernel(float* input, float* output, float* mean, float* variance, float* gamma, float* beta, float epsilon, int batch_size, int num_features, float* hat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_features) {
        int j = idx % num_features;
        float stddev = sqrtf(variance[j] + epsilon);
        float normalized = (input[idx] - mean[j]) / stddev;
        if (hat) hat[idx] = normalized;
        output[idx] = gamma[j] * normalized + beta[j];
    }
}

__global__ void compute_output_delta_batched_kernel(float* outputs, float* pre_acts, float* targets, float* deltas, int batch_size, int num, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        float diff = outputs[idx] - targets[idx];
        float act_deriv;
        if (act_type == 0) { // Sigmoid
            act_deriv = outputs[idx] * (1.0f - outputs[idx]);
        } else if (act_type == 1) { // ReLU
            act_deriv = pre_acts[idx] > 0 ? 1.0f : 0.0f;
        } else { // Tanh
            act_deriv = 1.0f - outputs[idx] * outputs[idx];
        }
        deltas[idx] = diff * act_deriv;
    }
}

__global__ void compute_hidden_delta_batched_kernel(float* delta_next, float* weights_next, float* errors, int batch_size, int num_current, int num_next) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_current) {
        int j = idx % num_current;
        float sum = 0.0f;
        for (int k = 0; k < num_next; ++k) {
            sum += delta_next[(idx / num_current) * num_next + k] * weights_next[k * num_current + j];
        }
        errors[idx] = sum;
    }
}

__global__ void compute_bn_grad_batched_kernel(float* error, float* hat, float* gamma_grad, float* beta_grad, int batch_size, int num) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num) {
        float g_grad = 0.0f;
        float b_grad = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            int idx = i * num + j;
            g_grad += error[idx] * hat[idx];
            b_grad += error[idx];
        }
        gamma_grad[j] = g_grad;
        beta_grad[j] = b_grad;
    }
}

__global__ void update_error_for_bn_kernel(float* error, float* gamma, float* variance, float epsilon, int batch_size, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        int j = idx % num;
        float stddev = sqrtf(variance[j] + epsilon);
        error[idx] *= gamma[j] / stddev;
    }
}

__global__ void compute_delta_from_error_kernel(float* error, float* pre_act, float* delta, int batch_size, int num, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        float act_deriv;
        if (act_type == 0) { // Sigmoid
            float sigmoid = 1.0f / (1.0f + expf(-pre_act[idx]));
            act_deriv = sigmoid * (1.0f - sigmoid);
        } else if (act_type == 1) { // ReLU
            act_deriv = pre_act[idx] > 0 ? 1.0f : 0.0f;
        } else { // Tanh
            float tanh_val = tanhf(pre_act[idx]);
            act_deriv = 1.0f - tanh_val * tanh_val;
        }
        delta[idx] = error[idx] * act_deriv;
    }
}

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

__global__ void apply_bn_backprop_correction(float* deltas, float* hats, float* gammas, float* gamma_grads, float* beta_grads, float* variances, float epsilon, int batch_size, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num) {
        int j = idx % num;
        float stddev = sqrtf(variances[j] + epsilon);
        float correction1 = gammas[j] * beta_grads[j] / stddev;
        float correction2 = gammas[j] * gamma_grads[j] / stddev;
        deltas[idx] -= correction1 + hats[idx] * correction2;
    }
}

__global__ void adam_update_kernel(float* weights, float* weight_gradients, float* m_weights, float* v_weights,
                                   float* biases, float* bias_gradients, float* m_biases, float* v_biases,
                                   float* gammas, float* gamma_gradients, float* m_gammas, float* v_gammas,
                                   float* betas, float* beta_gradients, float* m_betas, float* v_betas,
                                   float learning_rate, float beta1, float beta2, float epsilon,
                                   int num_weights, int num, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float beta1_t = powf(beta1, t);
    float beta2_t = powf(beta2, t);

    if (idx < num_weights) {
        m_weights[idx] = beta1 * m_weights[idx] + (1.0f - beta1) * weight_gradients[idx];
        v_weights[idx] = beta2 * v_weights[idx] + (1.0f - beta2) * weight_gradients[idx] * weight_gradients[idx];
        float m_hat = m_weights[idx] / (1.0f - beta1_t);
        float v_hat = v_weights[idx] / (1.0f - beta2_t);
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
    if (idx < num) {
        m_biases[idx] = beta1 * m_biases[idx] + (1.0f - beta1) * bias_gradients[idx];
        v_biases[idx] = beta2 * v_biases[idx] + (1.0f - beta2) * bias_gradients[idx] * bias_gradients[idx];
        float m_hat = m_biases[idx] / (1.0f - beta1_t);
        float v_hat = v_biases[idx] / (1.0f - beta2_t);
        biases[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);

        if (gammas) {
            m_gammas[idx] = beta1 * m_gammas[idx] + (1.0f - beta1) * gamma_gradients[idx];
            v_gammas[idx] = beta2 * v_gammas[idx] + (1.0f - beta2) * gamma_gradients[idx] * gamma_gradients[idx];
            float m_hat_gamma = m_gammas[idx] / (1.0f - beta1_t);
            float v_hat_gamma = v_gammas[idx] / (1.0f - beta2_t);
            gammas[idx] -= learning_rate * m_hat_gamma / (sqrtf(v_hat_gamma) + epsilon);

            m_betas[idx] = beta1 * m_betas[idx] + (1.0f - beta1) * beta_gradients[idx];
            v_betas[idx] = beta2 * v_betas[idx] + (1.0f - beta2) * beta_gradients[idx] * beta_gradients[idx];
            float m_hat_beta = m_betas[idx] / (1.0f - beta1_t);
            float v_hat_beta = v_betas[idx] / (1.0f - beta2_t);
            betas[idx] -= learning_rate * m_hat_beta / (sqrtf(v_hat_beta) + epsilon);
        }
    }
}

extern "C" void fused_bias_activation_on_gpu(float* input, float* biases, float* output, int batch_size, int num, int act_type, float* pre_act) {
    int blockSize = min(256, max(32, (batch_size * num + 31) / 32 * 32)); // Align to warp size
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    fused_bias_activation_kernel<<<gridSize, blockSize>>>(input, biases, output, batch_size, num, act_type, pre_act);
}

extern "C" void compute_batch_norm_stats_batched_on_gpu(float* input, float* mean, float* variance, int batch_size, int num_features, int block_size) {
    int gridSize = (num_features + block_size - 1) / block_size;
    compute_batch_norm_stats_batched_kernel<<<gridSize, block_size>>>(input, mean, variance, batch_size, num_features);
}

extern "C" void compute_batch_norm_batched_on_gpu(float* input, float* output, float* mean, float* variance, float* gamma, float* beta, float epsilon, int batch_size, int num_features, float* hat) {
    int blockSize = 256;
    int gridSize = (batch_size * num_features + blockSize - 1) / blockSize;
    compute_batch_norm_batched_kernel<<<gridSize, blockSize>>>(input, output, mean, variance, gamma, beta, epsilon, batch_size, num_features, hat);
}

extern "C" void compute_output_delta_batched_on_gpu(float* outputs, float* pre_acts, float* targets, float* deltas, int batch_size, int num, int act_type) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    compute_output_delta_batched_kernel<<<gridSize, blockSize>>>(outputs, pre_acts, targets, deltas, batch_size, num, act_type);
}

extern "C" void compute_hidden_delta_batched_on_gpu(float* delta_next, float* weights_next, float* errors, int batch_size, int num_current, int num_next) {
    int blockSize = 256;
    int gridSize = (batch_size * num_current + blockSize - 1) / blockSize;
    compute_hidden_delta_batched_kernel<<<gridSize, blockSize>>>(delta_next, weights_next, errors, batch_size, num_current, num_next);
}

extern "C" void compute_bn_grad_batched_on_gpu(float* error, float* hat, float* gamma_grad, float* beta_grad, int batch_size, int num) {
    int blockSize = 256;
    int gridSize = (num + blockSize - 1) / blockSize;
    compute_bn_grad_batched_kernel<<<gridSize, blockSize>>>(error, hat, gamma_grad, beta_grad, batch_size, num);
}

extern "C" void update_error_for_bn_on_gpu(float* error, float* gamma, float* variance, float epsilon, int batch_size, int num) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    update_error_for_bn_kernel<<<gridSize, blockSize>>>(error, gamma, variance, epsilon, batch_size, num);
}

extern "C" void compute_delta_from_error_on_gpu(float* error, float* pre_act, float* delta, int batch_size, int num, int act_type) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    compute_delta_from_error_kernel<<<gridSize, blockSize>>>(error, pre_act, delta, batch_size, num, act_type);
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

extern "C" void apply_bn_backprop_correction_on_gpu(float* deltas, float* hats, float* gammas, float* gamma_grads, float* beta_grads, float* variances, float epsilon, int batch_size, int num) {
    int blockSize = 256;
    int gridSize = (batch_size * num + blockSize - 1) / blockSize;
    apply_bn_backprop_correction<<<gridSize, blockSize>>>(deltas, hats, gammas, gamma_grads, beta_grads, variances, epsilon, batch_size, num);
}

extern "C" void adam_update_on_gpu(float* weights, float* weight_gradients, float* m_weights, float* v_weights,
                                   float* biases, float* bias_gradients, float* m_biases, float* v_biases,
                                   float* gammas, float* gamma_gradients, float* m_gammas, float* v_gammas,
                                   float* betas, float* beta_gradients, float* m_betas, float* v_betas,
                                   float learning_rate, float beta1, float beta2, float epsilon,
                                   int num_weights, int num, int t, bool use_bn) {
    int blockSize = 256;
    int gridSize = (max(num_weights, num) + blockSize - 1) / blockSize;
    adam_update_kernel<<<gridSize, blockSize>>>(weights, weight_gradients, m_weights, v_weights,
                                                biases, bias_gradients, m_biases, v_biases,
                                                gammas, gamma_gradients, m_gammas, v_gammas,
                                                betas, beta_gradients, m_betas, v_betas,
                                                learning_rate, beta1, beta2, epsilon,
                                                num_weights, num, t);
}