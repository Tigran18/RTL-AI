#include "network.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <numeric>
#include <cmath>

// Error checking macros
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl << std::flush; \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl << std::flush; \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

// Neuron implementation
network::neuron::neuron(size_t number_of_weights, std::mt19937& gen) 
    : m_number_of_weights(number_of_weights), m_bias(generate_random_value(gen)) {
    std::cout << "Creating neuron with " << number_of_weights << " weights" << std::endl << std::flush;
    for (size_t i = 0; i < number_of_weights; ++i) {
        m_weights.push_back(generate_random_value(gen));
        m_weight_updates.push_back(0.0f);
    }
}

float network::neuron::generate_random_value(std::mt19937& gen) {
    float range = (m_number_of_weights > 0) ? 1.0f / sqrtf(static_cast<float>(m_number_of_weights)) : 1.0f;
    std::uniform_real_distribution<float> dis(-range, range);
    return dis(gen);
}

// Network implementation
network::network(std::vector<size_t> number_of_neurons_per_layer,
                std::vector<ActivationType> activations,
                float learning_rate, size_t epochs, size_t batch_size, 
                float momentum, bool use_batch_norm)
    : m_number_of_neurons_per_layer(number_of_neurons_per_layer),
      m_learning_rate(learning_rate), m_epochs(epochs), m_batch_size(batch_size),
      m_momentum(momentum), m_activations(activations), m_use_batch_norm(use_batch_norm) {
    std::cout << "Initializing network with " << number_of_neurons_per_layer.size() 
              << " layers..." << std::endl << std::flush;
    m_layers = number_of_neurons_per_layer.size();
    
    // Validate parameters
    if (m_batch_size == 0) throw std::invalid_argument("Batch size must be greater than 0");
    if (m_momentum < 0.0f || m_momentum > 1.0f) throw std::invalid_argument("Momentum must be between 0 and 1");
    if (activations.size() != m_layers - 1) throw std::invalid_argument("Number of activations must match number of non-input layers");
    
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize neurons
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_network.emplace_back();
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
            m_network[layer].emplace_back(prev_layer_size, gen);
        }
    }

    // Initialize GPU memory for weights, biases, and Adam moments
    d_weights.resize(m_layers, nullptr);
    d_biases.resize(m_layers, nullptr);
    d_gammas.resize(m_layers, nullptr);
    d_betas.resize(m_layers, nullptr);
    d_hats.resize(m_layers, nullptr);
    d_mean.resize(m_layers, nullptr);
    d_variance.resize(m_layers, nullptr);
    d_gamma_grad.resize(m_layers, nullptr);
    d_beta_grad.resize(m_layers, nullptr);
    d_error.resize(m_layers, nullptr);
    d_weight_gradients.resize(m_layers, nullptr);
    d_bias_gradients.resize(m_layers, nullptr);
    d_weight_m.resize(m_layers, nullptr);
    d_weight_v.resize(m_layers, nullptr);
    d_bias_m.resize(m_layers, nullptr);
    d_bias_v.resize(m_layers, nullptr);
    d_gamma_m.resize(m_layers, nullptr);
    d_gamma_v.resize(m_layers, nullptr);
    d_beta_m.resize(m_layers, nullptr);
    d_beta_v.resize(m_layers, nullptr);

    for (size_t layer = 1; layer < m_layers; ++layer) {
        size_t num = m_number_of_neurons_per_layer[layer];
        size_t prev = m_number_of_neurons_per_layer[layer - 1];

        // Allocate and copy weights
        float* h_weights = new float[prev * num];
        float* h_biases = new float[num];
        float* h_gammas = new float[num];
        float* h_betas = new float[num];

        for (size_t out = 0; out < num; ++out) {
            const auto& n = m_network[layer][out];
            for (size_t in = 0; in < prev; ++in) {
                h_weights[in * num + out] = n.m_weights[in];
            }
            h_biases[out] = n.m_bias;
            h_gammas[out] = n.m_bn_gamma;
            h_betas[out] = n.m_bn_beta;
        }

        CUDA_CHECK(cudaMalloc(&d_weights[layer], prev * num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_weights[layer], h_weights, prev * num * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_biases[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_biases[layer], h_biases, num * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_gammas[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gammas[layer], h_gammas, num * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_betas[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_betas[layer], h_betas, num * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_mean[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_variance[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma_grad[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta_grad[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_error[layer], m_batch_size * num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_gradients[layer], prev * num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_gradients[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_m[layer], prev * num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_v[layer], prev * num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_m[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_v[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma_m[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma_v[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta_m[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta_v[layer], num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_weight_m[layer], 0, prev * num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_weight_v[layer], 0, prev * num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias_m[layer], 0, num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias_v[layer], 0, num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gamma_m[layer], 0, num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gamma_v[layer], 0, num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_beta_m[layer], 0, num * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_beta_v[layer], 0, num * sizeof(float)));

        delete[] h_weights;
        delete[] h_biases;
        delete[] h_gammas;
        delete[] h_betas;
    }

    initialize_gpu();
}

network::~network() {
    cleanup_gpu();
}

void network::initialize_gpu() {
    std::cout << "Initializing GPU resources..." << std::endl << std::flush;
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << free_mem / (1024.0 * 1024.0) << " MB free of " 
              << total_mem / (1024.0 * 1024.0) << " MB total" << std::endl << std::flush;
    
    CUBLAS_CHECK(cublasCreate(&m_cublas_handle));
    
    // Allocate memory for batched operations
    d_layer_outputs.resize(m_layers, nullptr);
    d_pre_acts.resize(m_layers, nullptr);
    d_layer_deltas.resize(m_layers, nullptr);

    for (size_t layer = 0; layer < m_layers; ++layer) {
        size_t size = m_batch_size * m_number_of_neurons_per_layer[layer];
        CUDA_CHECK(cudaMalloc(&d_layer_outputs[layer], size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_acts[layer], size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_layer_deltas[layer], size * sizeof(float)));
        if (layer > 0 && m_use_batch_norm && layer < m_layers - 1) {
            CUDA_CHECK(cudaMalloc(&d_hats[layer], size * sizeof(float)));
        }
    }

    // Pre-allocate d_ones
    CUDA_CHECK(cudaMalloc(&d_ones, m_batch_size * sizeof(float)));
    float* h_ones = new float[m_batch_size];
    for (size_t i = 0; i < m_batch_size; ++i) {
        h_ones[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_ones, h_ones, m_batch_size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_ones;

    // Get optimal block size
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    m_block_size = prop.warpSize * 4; // Example: 128 for many GPUs
}

void network::cleanup_gpu() {
    std::cout << "Cleaning up GPU resources..." << std::endl << std::flush;
    CUBLAS_CHECK(cublasDestroy(m_cublas_handle));
    for (size_t layer = 0; layer < m_layers; ++layer) {
        if (d_layer_outputs[layer]) CUDA_CHECK(cudaFree(d_layer_outputs[layer]));
        if (d_pre_acts[layer]) CUDA_CHECK(cudaFree(d_pre_acts[layer]));
        if (d_layer_deltas[layer]) CUDA_CHECK(cudaFree(d_layer_deltas[layer]));
        if (d_weights[layer]) CUDA_CHECK(cudaFree(d_weights[layer]));
        if (d_biases[layer]) CUDA_CHECK(cudaFree(d_biases[layer]));
        if (d_gammas[layer]) CUDA_CHECK(cudaFree(d_gammas[layer]));
        if (d_betas[layer]) CUDA_CHECK(cudaFree(d_betas[layer]));
        if (d_hats[layer]) CUDA_CHECK(cudaFree(d_hats[layer]));
        if (d_mean[layer]) CUDA_CHECK(cudaFree(d_mean[layer]));
        if (d_variance[layer]) CUDA_CHECK(cudaFree(d_variance[layer]));
        if (d_gamma_grad[layer]) CUDA_CHECK(cudaFree(d_gamma_grad[layer]));
        if (d_beta_grad[layer]) CUDA_CHECK(cudaFree(d_beta_grad[layer]));
        if (d_error[layer]) CUDA_CHECK(cudaFree(d_error[layer]));
        if (d_weight_gradients[layer]) CUDA_CHECK(cudaFree(d_weight_gradients[layer]));
        if (d_bias_gradients[layer]) CUDA_CHECK(cudaFree(d_bias_gradients[layer]));
        if (d_weight_m[layer]) CUDA_CHECK(cudaFree(d_weight_m[layer]));
        if (d_weight_v[layer]) CUDA_CHECK(cudaFree(d_weight_v[layer]));
        if (d_bias_m[layer]) CUDA_CHECK(cudaFree(d_bias_m[layer]));
        if (d_bias_v[layer]) CUDA_CHECK(cudaFree(d_bias_v[layer]));
        if (d_gamma_m[layer]) CUDA_CHECK(cudaFree(d_gamma_m[layer]));
        if (d_gamma_v[layer]) CUDA_CHECK(cudaFree(d_gamma_v[layer]));
        if (d_beta_m[layer]) CUDA_CHECK(cudaFree(d_beta_m[layer]));
        if (d_beta_v[layer]) CUDA_CHECK(cudaFree(d_beta_v[layer]));
    }
    if (d_ones) CUDA_CHECK(cudaFree(d_ones));
}

void network::forward_propagate(const std::vector<std::vector<float>>& batch_inputs, bool training) {
    size_t B = batch_inputs.size();
    if (B == 0) return;
    size_t input_dim = m_number_of_neurons_per_layer[0];
    if (batch_inputs[0].size() != input_dim) 
        throw std::out_of_range("Input size does not match input layer size.");

    // Copy batched inputs to GPU
    std::vector<float> h_batch(B * input_dim);
    for (size_t i = 0; i < B; ++i) {
        memcpy(&h_batch[i * input_dim], batch_inputs[i].data(), input_dim * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpy(d_layer_outputs[0], h_batch.data(), B * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Forward pass through layers
    for (size_t layer = 1; layer < m_layers; ++layer) {
        size_t num = m_number_of_neurons_per_layer[layer];
        size_t prev = m_number_of_neurons_per_layer[layer - 1];
        int act_type = static_cast<int>(m_activations[layer - 1]);

        // Linear transformation using cuBLAS
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(m_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, num, B, prev, 
                                 &alpha, d_weights[layer], num, d_layer_outputs[layer - 1], prev, 
                                 &beta, d_pre_acts[layer], num));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (m_use_batch_norm && layer < m_layers - 1) {
            // Compute batch statistics
            compute_batch_norm_stats_batched_on_gpu(d_pre_acts[layer], d_mean[layer], 
                                                   d_variance[layer], B, num, m_block_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (training) {
                // Update running statistics (on CPU for simplicity, low overhead)
                std::vector<float> h_mean(num), h_variance(num);
                CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean[layer], num * sizeof(float), 
                                     cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_variance.data(), d_variance[layer], num * sizeof(float), 
                                     cudaMemcpyDeviceToHost));

                for (size_t j = 0; j < num; ++j) {
                    m_network[layer][j].m_bn_mean = (1 - m_bn_momentum) * h_mean[j] + 
                                                   m_bn_momentum * m_network[layer][j].m_bn_mean;
                    m_network[layer][j].m_bn_variance = (1 - m_bn_momentum) * h_variance[j] + 
                                                       m_bn_momentum * m_network[layer][j].m_bn_variance;
                }
            } else {
                // Use running statistics for inference
                std::vector<float> h_mean(num), h_variance(num);
                for (size_t j = 0; j < num; ++j) {
                    h_mean[j] = m_network[layer][j].m_bn_mean;
                    h_variance[j] = m_network[layer][j].m_bn_variance;
                }
                CUDA_CHECK(cudaMemcpy(d_mean[layer], h_mean.data(), num * sizeof(float), 
                                     cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_variance[layer], h_variance.data(), num * sizeof(float), 
                                     cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Apply batch normalization
            compute_batch_norm_batched_on_gpu(d_pre_acts[layer], d_layer_outputs[layer], 
                                             d_mean[layer], d_variance[layer], 
                                             d_gammas[layer], d_betas[layer], 
                                             m_bn_epsilon, B, num, d_hats[layer]);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            // Apply bias and activation
            fused_bias_activation_on_gpu(d_pre_acts[layer], d_biases[layer], 
                                        d_layer_outputs[layer], B, num, act_type, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

void network::backpropagate(const std::vector<std::vector<float>>& batch_targets) {
    size_t B = batch_targets.size();
    size_t output_dim = m_number_of_neurons_per_layer[m_layers - 1];
    if (batch_targets[0].size() != output_dim) 
        throw std::out_of_range("Target size does not match output layer size.");

    // Copy batched targets to GPU
    std::vector<float> h_targets(B * output_dim);
    for (size_t i = 0; i < B; ++i) {
        memcpy(&h_targets[i * output_dim], batch_targets[i].data(), output_dim * sizeof(float));
    }

    float* d_targets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_targets, B * output_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), B * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute output layer deltas
    int act_type = static_cast<int>(m_activations[m_layers - 2]);
    compute_output_delta_batched_on_gpu(d_layer_outputs[m_layers - 1], d_pre_acts[m_layers - 1], 
                                       d_targets, d_layer_deltas[m_layers - 1], B, output_dim, 
                                       act_type);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_targets));

    // Backpropagate through hidden layers
    for (int layer = m_layers - 2; layer >= 1; --layer) {
        size_t num = m_number_of_neurons_per_layer[layer];
        size_t num_next = m_number_of_neurons_per_layer[layer + 1];
        act_type = static_cast<int>(m_activations[layer - 1]);

        // Compute errors for current layer
        compute_hidden_delta_batched_on_gpu(d_layer_deltas[layer + 1], d_weights[layer + 1], 
                                           d_error[layer], B, num, num_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (m_use_batch_norm && layer < static_cast<int>(m_layers) - 1) {
            // Compute dL/dy = dL/da * act'(y) where y is the input to the activation (post-BN)
            compute_delta_from_error_on_gpu(d_error[layer], d_pre_acts[layer], 
                                           d_layer_deltas[layer], B, num, act_type);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Compute batch norm gradients using dL/dy
            compute_bn_grad_batched_on_gpu(d_layer_deltas[layer], d_hats[layer], 
                                          d_gamma_grad[layer], d_beta_grad[layer], B, num);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Start with the main term: dL/dz ≈ dL/dy * (gamma / stddev)
            update_error_for_bn_on_gpu(d_layer_deltas[layer], d_gammas[layer], 
                                      d_variance[layer], m_bn_epsilon, B, num);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Apply corrections for exact BN backprop (accounting for mean/var dependencies)
            apply_bn_backprop_correction_on_gpu(d_layer_deltas[layer], d_hats[layer], 
                                               d_gammas[layer], d_gamma_grad[layer], 
                                               d_beta_grad[layer], d_variance[layer], 
                                               m_bn_epsilon, B, num);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Now d_layer_deltas = exact dL/dx for weight updates
        } else {
            // Non-BN: dL/dpre = dL/da * act'(pre)
            compute_delta_from_error_on_gpu(d_error[layer], d_pre_acts[layer], 
                                           d_layer_deltas[layer], B, num, act_type);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

void network::train(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    const std::vector<std::vector<float>>& val_inputs,
                    const std::vector<std::vector<float>>& val_targets) {
    std::cout << "Starting training with " << inputs.size() << " samples..." << std::endl << std::flush;
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (!val_inputs.empty() && val_inputs.size() != val_targets.size()) {
        throw std::invalid_argument("Number of validation inputs must match number of validation targets");
    }
    if (inputs.empty()) {
        std::cout << "No training data provided." << std::endl << std::flush;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indices(inputs.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Adam iteration counter
    int t = 0;

    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        float total_error = 0.0f;

        for (size_t batch_start = 0; batch_start < inputs.size(); batch_start += m_batch_size) {
            size_t current_batch_size = std::min(m_batch_size, inputs.size() - batch_start);
            std::vector<std::vector<float>> batch_inputs(current_batch_size);
            std::vector<std::vector<float>> batch_targets(current_batch_size);

            // Prepare batch
            for (size_t b = 0; b < current_batch_size; ++b) {
                size_t idx = indices[batch_start + b];
                batch_inputs[b] = inputs[idx];
                batch_targets[b] = targets[idx];
            }

            // Forward and backward pass
            forward_propagate(batch_inputs, true);
            CUDA_CHECK(cudaDeviceSynchronize());
            backpropagate(batch_targets);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Increment Adam iteration counter
            t++;

            // Compute gradients on GPU
            for (size_t layer = 1; layer < m_layers; ++layer) {
                size_t num = m_number_of_neurons_per_layer[layer];
                size_t prev = m_number_of_neurons_per_layer[layer - 1];

                // Compute weight gradients using cuBLAS (deltas * prev_acts^T)
                float alpha = 1.0f / current_batch_size, beta = 0.0f;
                CUBLAS_CHECK(cublasSgemm(m_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                         num, prev, current_batch_size, 
                                         &alpha, d_layer_deltas[layer], num, 
                                         d_layer_outputs[layer - 1], prev, 
                                         &beta, d_weight_gradients[layer], num));
                CUDA_CHECK(cudaDeviceSynchronize());

                // Compute bias gradients (sum deltas over batch)
                CUBLAS_CHECK(cublasSgemv(m_cublas_handle, CUBLAS_OP_N, num, current_batch_size,
                                         &alpha, d_layer_deltas[layer], num, d_ones, 1,
                                         &beta, d_bias_gradients[layer], 1));
                CUDA_CHECK(cudaDeviceSynchronize());

                // Update parameters using Adam on GPU
                adam_update_on_gpu(d_weights[layer], d_weight_gradients[layer], d_weight_m[layer], d_weight_v[layer],
                                   d_biases[layer], d_bias_gradients[layer], d_bias_m[layer], d_bias_v[layer],
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_gammas[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_gamma_grad[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_gamma_m[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_gamma_v[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_betas[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_beta_grad[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_beta_m[layer] : nullptr,
                                   (m_use_batch_norm && layer < m_layers - 1) ? d_beta_v[layer] : nullptr,
                                   m_learning_rate, m_beta1, m_beta2, m_epsilon, prev * num, num, t,
                                   m_use_batch_norm && layer < m_layers - 1);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Compute batch error (on CPU for simplicity)
            std::vector<float> h_output(current_batch_size * m_number_of_neurons_per_layer[m_layers - 1]);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_layer_outputs[m_layers - 1], 
                                 current_batch_size * m_number_of_neurons_per_layer[m_layers - 1] * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());

            for (size_t b = 0; b < current_batch_size; ++b) {
                size_t idx = indices[batch_start + b];
                for (size_t j = 0; j < targets[idx].size(); ++j) {
                    float diff = h_output[b * targets[idx].size() + j] - targets[idx][j];
                    total_error += diff * diff;
                }
            }
        }

        total_error /= inputs.size();
        float val_error = val_inputs.empty() ? 0.0f : evaluate(val_inputs, val_targets);
        if (epoch % 100 == 0 || epoch == m_epochs - 1) {
            std::cout << "Epoch " << epoch << ", Train MSE: " << total_error;
            if (!val_inputs.empty()) {
                std::cout << ", Validation MSE: " << val_error;
            }
            std::cout << std::endl << std::flush;
        }
    }
}

float network::evaluate(const std::vector<std::vector<float>>& inputs,
                       const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (inputs.empty()) {
        std::cout << "No evaluation data provided." << std::endl << std::flush;
        return 0.0f;
    }
    float total_error = 0.0f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = predict(inputs[i]);
        for (size_t j = 0; j < targets[i].size(); ++j) {
            float diff = targets[i][j] - output[j];
            total_error += diff * diff;
        }
    }
    float mse = total_error / inputs.size();
    std::cout << "Evaluation MSE: " << mse << std::endl << std::flush;
    return mse;
}

std::vector<float> network::predict(const std::vector<float>& input) {
    forward_propagate({input}, false);
    size_t output_dim = m_number_of_neurons_per_layer[m_layers - 1];
    std::vector<float> output(output_dim);
    CUDA_CHECK(cudaMemcpy(output.data(), d_layer_outputs[m_layers - 1], 
                         output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

void network::display_outputs() const {
    for (size_t layer = 0; layer < m_layers; ++layer) {
        std::cout << "Layer " << layer + 1 << " outputs: ";
        size_t num = m_number_of_neurons_per_layer[layer];
        std::vector<float> h_outputs(num);
        CUDA_CHECK(cudaMemcpy(h_outputs.data(), d_layer_outputs[layer], 
                             num * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < num; ++j) {
            std::cout << h_outputs[j] << " ";
        }
        std::cout << std::endl << std::flush;
    }
}

void network::save_model(const std::string& filename) const {
    std::cout << "Saving model to " << filename << "..." << std::endl << std::flush;
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open file for saving model: " << filename << std::endl << std::flush;
        throw std::runtime_error("Failed to open file for saving model.");
    }
    out << m_layers << "\n";
    for (size_t n : m_number_of_neurons_per_layer) {
        out << n << " ";
    }
    out << "\n";
    for (const auto& act : m_activations) {
        out << static_cast<int>(act) << " ";
    }
    out << "\n";
    out << m_use_batch_norm << "\n";
    for (const auto& layer : m_network) {
        for (const auto& neuron : layer) {
            out << neuron.m_bias << " ";
            for (float w : neuron.m_weights) {
                out << w << " ";
            }
            out << neuron.m_bn_gamma << " " << neuron.m_bn_beta << " ";
            out << neuron.m_bn_mean << " " << neuron.m_bn_variance << "\n";
        }
    }
    out.close();
    std::cout << "Model saved successfully." << std::endl << std::flush;
}

void network::load_model(const std::string& filename) {
    std::cout << "Loading model from " << filename << "..." << std::endl << std::flush;
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Failed to open file for loading model: " << filename << std::endl << std::flush;
        throw std::runtime_error("Failed to open file for loading model.");
    }
    in >> m_layers;
    m_number_of_neurons_per_layer.resize(m_layers);
    for (size_t i = 0; i < m_layers; ++i) {
        in >> m_number_of_neurons_per_layer[i];
    }
    m_activations.resize(m_layers - 1);
    for (size_t i = 0; i < m_layers - 1; ++i) {
        int act;
        in >> act;
        if (act < 0 || act > static_cast<int>(ActivationType::Tanh)) {
            throw std::runtime_error("Invalid activation type in model file");
        }
        m_activations[i] = static_cast<ActivationType>(act);
    }
    int use_bn;
    in >> use_bn;
    m_use_batch_norm = use_bn;
    m_network.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t layer = 0; layer < m_layers; ++layer) {
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        std::vector<neuron> layer_neurons;
        for (size_t j = 0; j < m_number_of_neurons_per_layer[layer]; ++j) {
            neuron n(prev_layer_size, gen);
            in >> n.m_bias;
            for (size_t k = 0; k < prev_layer_size; ++k) {
                in >> n.m_weights[k];
            }
            in >> n.m_bn_gamma >> n.m_bn_beta >> n.m_bn_mean >> n.m_bn_variance;
            layer_neurons.push_back(n);
        }
        m_network.push_back(layer_neurons);
    }
    in.close();
    std::cout << "Model loaded successfully." << std::endl << std::flush;
    initialize_gpu();
}
