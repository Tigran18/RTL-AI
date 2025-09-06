#include "network.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <numeric>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

extern "C" void compute_activation_on_gpu(double* input, double* output, int size, int act_type);
extern "C" void compute_batch_norm_on_gpu(double* input, double* output, double mean, double variance, double gamma, double beta, double epsilon, int size);

network::neuron::neuron(size_t number_of_weights, std::mt19937& gen)
    : m_number_of_weights(number_of_weights), m_bias(generate_random_value(gen)) {
    for (size_t i = 0; i < number_of_weights; ++i) {
        m_weights.push_back(generate_random_value(gen));
        m_weight_updates.push_back(0.0);
    }
    allocate_gpu_memory();
}

network::neuron::~neuron() {
    free_gpu_memory();
}

void network::neuron::allocate_gpu_memory() {
    if (m_number_of_weights > 0) {
        CUDA_CHECK(cudaMalloc(&d_weights, m_number_of_weights * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_inputs, m_number_of_weights * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_weight_updates, m_number_of_weights * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(double)));
        sync_weights_to_device();
        CUDA_CHECK(cudaMemset(d_weight_updates, 0, m_number_of_weights * sizeof(double)));
    }
}

void network::neuron::free_gpu_memory() {
    if (d_weights) CUDA_CHECK(cudaFree(d_weights));
    if (d_inputs) CUDA_CHECK(cudaFree(d_inputs));
    if (d_weight_updates) CUDA_CHECK(cudaFree(d_weight_updates));
    if (d_output) CUDA_CHECK(cudaFree(d_output));
}

void network::neuron::sync_weights_to_device() {
    if (m_number_of_weights > 0) {
        CUDA_CHECK(cudaMemcpy(d_weights, m_weights.data(), m_number_of_weights * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void network::neuron::sync_weights_to_host() const {
    if (m_number_of_weights > 0) {
        CUDA_CHECK(cudaMemcpy(const_cast<double*>(m_weights.data()), d_weights, m_number_of_weights * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

double network::neuron::generate_random_value(std::mt19937& gen) {
    double range = (m_number_of_weights > 0) ? 1.0 / m_number_of_weights : 1.0;
    std::uniform_real_distribution<double> dis(-range, range);
    return dis(gen);
}

void network::neuron::set_inputs(const std::vector<double>& inputs) {
    m_inputs = inputs;
    if (m_number_of_weights > 0) {
        CUDA_CHECK(cudaMemcpy(d_inputs, m_inputs.data(), m_number_of_weights * sizeof(double), cudaMemcpyHostToDevice));
    }
}

double network::neuron::activate(double x, ActivationType type) const {
    switch (type) {
        case ActivationType::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU: return std::max(0.0, x);
        case ActivationType::Tanh: return std::tanh(x);
        default: throw std::invalid_argument("Unknown activation type");
    }
}

double network::neuron::activate_derivative(double x, ActivationType type) const {
    switch (type) {
        case ActivationType::Sigmoid: { double sig = activate(x, ActivationType::Sigmoid); return sig * (1.0 - sig); }
        case ActivationType::ReLU: return x > 0 ? 1.0 : 0.0;
        case ActivationType::Tanh: { double t = std::tanh(x); return 1.0 - t * t; }
        default: throw std::invalid_argument("Unknown activation type");
    }
}

double network::neuron::compute_output(ActivationType type, bool use_bn, double bn_mean,
                                      double bn_variance, double bn_gamma, double bn_beta,
                                      bool training, double epsilon, cublasHandle_t cublas_handle) {
    double h_output = 0.0;
    if (m_number_of_weights > 0) {
        double one = 1.0;
        CUBLAS_CHECK(cublasDdot(cublas_handle, m_number_of_weights, d_weights, 1, d_inputs, 1, &h_output));
        h_output += m_bias;
    }
    m_z = h_output;

    CUDA_CHECK(cudaMemcpy(d_output, &h_output, sizeof(double), cudaMemcpyHostToDevice));
    compute_activation_on_gpu(d_output, d_output, 1, static_cast<int>(type));
    CUDA_CHECK(cudaMemcpy(&m_output, d_output, sizeof(double), cudaMemcpyDeviceToHost));

    if (use_bn && training) {
        CUDA_CHECK(cudaMemcpy(d_output, &m_output, sizeof(double), cudaMemcpyHostToDevice));
        compute_batch_norm_on_gpu(d_output, d_output, bn_mean, bn_variance, bn_gamma, bn_beta, epsilon, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&m_output, d_output, sizeof(double), cudaMemcpyDeviceToHost));
        m_bn_normalized = m_output;
    } else if (use_bn) {
        CUDA_CHECK(cudaMemcpy(d_output, &m_output, sizeof(double), cudaMemcpyHostToDevice));
        compute_batch_norm_on_gpu(d_output, d_output, m_bn_mean, m_bn_variance, bn_gamma, bn_beta, epsilon, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&m_output, d_output, sizeof(double), cudaMemcpyDeviceToHost));
    }

    return m_output;
}

network::network(std::vector<size_t> number_of_neurons_per_layer,
                std::vector<ActivationType> activations,
                double learning_rate, size_t epochs, size_t batch_size, double momentum, bool use_batch_norm)
    : m_number_of_neurons_per_layer(number_of_neurons_per_layer),
      m_learning_rate(learning_rate), m_epochs(epochs), m_batch_size(batch_size),
      m_momentum(momentum), m_activations(activations), m_use_batch_norm(use_batch_norm) {
    m_layers = number_of_neurons_per_layer.size();
    if (m_batch_size == 0) throw std::invalid_argument("Batch size must be greater than 0");
    if (m_momentum < 0.0 || m_momentum > 1.0) throw std::invalid_argument("Momentum must be between 0 and 1");
    if (activations.size() != m_layers - 1) throw std::invalid_argument("Number of activations must match number of non-input layers");
    std::mt19937 gen(std::random_device{}());
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_network.push_back(std::vector<neuron>());
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
            m_network[layer].emplace_back(prev_layer_size, gen);
        }
    }
    initialize_gpu();
}

network::~network() {
    cleanup_gpu();
}

void network::initialize_gpu() {
    CUBLAS_CHECK(cublasCreate(&m_cublas_handle));
    d_layer_outputs.resize(m_layers);
    d_layer_deltas.resize(m_layers);
    for (size_t layer = 0; layer < m_layers; ++layer) {
        CUDA_CHECK(cudaMalloc(&d_layer_outputs[layer], m_number_of_neurons_per_layer[layer] * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_layer_deltas[layer], m_number_of_neurons_per_layer[layer] * sizeof(double)));
    }
}

void network::cleanup_gpu() {
    CUBLAS_CHECK(cublasDestroy(m_cublas_handle));
    for (auto ptr : d_layer_outputs) CUDA_CHECK(cudaFree(ptr));
    for (auto ptr : d_layer_deltas) CUDA_CHECK(cudaFree(ptr));
}

void network::forward_propagate(const std::vector<double>& input_values, bool training) {
    if (input_values.size() != m_number_of_neurons_per_layer[0]) {
        throw std::out_of_range("Input size does not match input layer size.");
    }
    CUDA_CHECK(cudaMemcpy(d_layer_outputs[0], input_values.data(), input_values.size() * sizeof(double), cudaMemcpyHostToDevice));

    for (size_t layer = 1; layer < m_layers; ++layer) {
        std::vector<double> prev_outputs(m_number_of_neurons_per_layer[layer - 1]);
        CUDA_CHECK(cudaMemcpy(prev_outputs.data(), d_layer_outputs[layer - 1], 
                             m_number_of_neurons_per_layer[layer - 1] * sizeof(double), cudaMemcpyDeviceToHost));
        
        std::vector<double> layer_outputs(m_number_of_neurons_per_layer[layer]);
        double batch_mean = 0.0, batch_variance = 0.0;
        
        if (m_use_batch_norm && layer < m_layers - 1 && training) {
            for (size_t i = 0; i < m_network[layer].size(); ++i) {
                auto& neuron = m_network[layer][i];
                neuron.set_inputs(prev_outputs);
                layer_outputs[i] = neuron.compute_output(m_activations[layer - 1], false, 0.0, 1.0, 1.0, 0.0, false, m_bn_epsilon, m_cublas_handle);
            }
            if (!layer_outputs.empty()) {
                batch_mean = std::accumulate(layer_outputs.begin(), layer_outputs.end(), 0.0) / layer_outputs.size();
                batch_variance = 0.0;
                for (const auto& output : layer_outputs) {
                    batch_variance += std::pow(output - batch_mean, 2);
                }
                batch_variance = layer_outputs.size() > 1 ? batch_variance / (layer_outputs.size() - 1) : 1.0;
            } else {
                batch_variance = 1.0;
            }
            for (auto& neuron : m_network[layer]) {
                neuron.m_bn_mean = 0.1 * batch_mean + 0.9 * neuron.m_bn_mean;
                neuron.m_bn_variance = 0.1 * batch_variance + 0.9 * neuron.m_bn_variance;
            }
        }
        
        for (size_t i = 0; i < m_network[layer].size(); ++i) {
            auto& neuron = m_network[layer][i];
            neuron.set_inputs(prev_outputs);
            layer_outputs[i] = neuron.compute_output(m_activations[layer - 1], m_use_batch_norm && layer < m_layers - 1,
                                                   batch_mean, batch_variance, neuron.m_bn_gamma, neuron.m_bn_beta, 
                                                   training, m_bn_epsilon, m_cublas_handle);
        }
        
        CUDA_CHECK(cudaMemcpy(d_layer_outputs[layer], layer_outputs.data(), 
                             m_number_of_neurons_per_layer[layer] * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void network::backpropagate(const std::vector<double>& target_values) {
    if (target_values.size() != m_network.back().size()) {
        throw std::out_of_range("Target size does not match output layer size.");
    }
    auto& output_layer = m_network.back();
    std::vector<double> output_deltas(output_layer.size());
    for (size_t i = 0; i < output_layer.size(); ++i) {
        double error = target_values[i] - output_layer[i].m_output;
        output_deltas[i] = error * output_layer[i].activate_derivative(
            output_layer[i].m_z, m_activations[m_layers - 2]);
    }
    CUDA_CHECK(cudaMemcpy(d_layer_deltas[m_layers - 1], output_deltas.data(),
                         output_deltas.size() * sizeof(double), cudaMemcpyHostToDevice));

    for (int layer = static_cast<int>(m_layers) - 2; layer > 0; --layer) {
        std::vector<double> layer_outputs;
        for (const auto& neuron : m_network[layer]) {
            layer_outputs.push_back(neuron.m_output);
        }
        double batch_mean = 0.0, batch_variance = 0.0;
        if (m_use_batch_norm && layer < static_cast<int>(m_layers) - 1) {
            if (!layer_outputs.empty()) {
                batch_mean = std::accumulate(layer_outputs.begin(), layer_outputs.end(), 0.0) / layer_outputs.size();
                batch_variance = 0.0;
                for (const auto& output : layer_outputs) {
                    batch_variance += std::pow(output - batch_mean, 2);
                }
                batch_variance = layer_outputs.size() > 1 ? batch_variance / (layer_outputs.size() - 1) : 1.0;
            } else {
                batch_variance = 1.0;
            }
        }
        std::vector<double> layer_deltas(m_network[layer].size());
        for (size_t i = 0; i < m_network[layer].size(); ++i) {
            double error = 0.0;
            for (size_t j = 0; j < m_network[layer + 1].size(); ++j) {
                double next_delta;
                CUDA_CHECK(cudaMemcpy(&next_delta, d_layer_deltas[layer + 1] + j, sizeof(double), cudaMemcpyDeviceToHost));
                error += m_network[layer + 1][j].m_weights[i] * next_delta;
            }
            if (m_use_batch_norm && layer < static_cast<int>(m_layers) - 1) {
                double variance_sqrt = std::sqrt(batch_variance + m_bn_epsilon);
                m_network[layer][i].m_bn_gamma_gradient = error * m_network[layer][i].m_bn_normalized;
                m_network[layer][i].m_bn_beta_gradient = error;
                error = error * m_network[layer][i].m_bn_gamma / variance_sqrt;
            }
            layer_deltas[i] = error * m_network[layer][i].activate_derivative(
                m_network[layer][i].m_z, m_activations[layer - 1]);
        }
        CUDA_CHECK(cudaMemcpy(d_layer_deltas[layer], layer_deltas.data(),
                             layer_deltas.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void network::train(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets,
                    const std::vector<std::vector<double>>& val_inputs,
                    const std::vector<std::vector<double>>& val_targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (!val_inputs.empty() && val_inputs.size() != val_targets.size()) {
        throw std::invalid_argument("Number of validation inputs must match number of validation targets");
    }
    if (inputs.empty()) {
        return;
    }
    std::mt19937 gen(std::random_device{}());
    std::vector<size_t> indices(inputs.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        double total_error = 0.0;
        for (size_t batch_start = 0; batch_start < inputs.size(); batch_start += m_batch_size) {
            size_t current_batch_size = std::min(m_batch_size, inputs.size() - batch_start);
            std::vector<std::vector<std::vector<double>>> weight_gradients(
                m_layers, std::vector<std::vector<double>>());
            std::vector<std::vector<double>> bias_gradients(m_layers);
            std::vector<std::vector<double>> bn_gamma_gradients(m_layers);
            std::vector<std::vector<double>> bn_beta_gradients(m_layers);
            for (size_t layer = 1; layer < m_layers; ++layer) {
                weight_gradients[layer].resize(m_network[layer].size());
                bias_gradients[layer].resize(m_network[layer].size(), 0.0);
                bn_gamma_gradients[layer].resize(m_network[layer].size(), 0.0);
                bn_beta_gradients[layer].resize(m_network[layer].size(), 0.0);
                for (size_t j = 0; j < m_network[layer].size(); ++j) {
                    weight_gradients[layer][j].resize(m_network[layer][j].m_number_of_weights, 0.0);
                }
            }
            for (size_t b = 0; b < current_batch_size; ++b) {
                size_t idx = indices[batch_start + b];
                forward_propagate(inputs[idx], true);
                backpropagate(targets[idx]);
                for (size_t layer = 1; layer < m_layers; ++layer) {
                    for (size_t j = 0; j < m_network[layer].size(); ++j) {
                        auto& neuron = m_network[layer][j];
                        for (size_t k = 0; k < neuron.m_number_of_weights; ++k) {
                            weight_gradients[layer][j][k] += neuron.m_delta * neuron.m_inputs[k];
                        }
                        bias_gradients[layer][j] += neuron.m_delta;
                        if (m_use_batch_norm && layer < m_layers - 1) {
                            bn_gamma_gradients[layer][j] += neuron.m_bn_gamma_gradient;
                            bn_beta_gradients[layer][j] += neuron.m_bn_beta_gradient;
                        }
                    }
                }
                for (size_t j = 0; j < targets[idx].size(); ++j) {
                    double diff = targets[idx][j] - m_network.back()[j].m_output;
                    total_error += diff * diff;
                }
            }
            for (size_t layer = 1; layer < m_layers; ++layer) {
                for (size_t j = 0; j < m_network[layer].size(); ++j) {
                    auto& neuron = m_network[layer][j];
                    for (size_t k = 0; k < neuron.m_number_of_weights; ++k) {
                        double gradient = weight_gradients[layer][j][k] / current_batch_size;
                        neuron.m_weight_updates[k] = m_momentum * neuron.m_weight_updates[k] +
                                                    m_learning_rate * gradient;
                        neuron.m_weights[k] += neuron.m_weight_updates[k];
                        CUDA_CHECK(cudaMemcpy(neuron.d_weights, neuron.m_weights.data(),
                                             neuron.m_number_of_weights * sizeof(double), cudaMemcpyHostToDevice));
                    }
                    double bias_gradient = bias_gradients[layer][j] / current_batch_size;
                    neuron.m_bias_update = m_momentum * neuron.m_bias_update +
                                          m_learning_rate * bias_gradient;
                    neuron.m_bias += neuron.m_bias_update;
                    if (m_use_batch_norm && layer < m_layers - 1) {
                        double gamma_gradient = bn_gamma_gradients[layer][j] / current_batch_size;
                        double beta_gradient = bn_beta_gradients[layer][j] / current_batch_size;
                        neuron.m_bn_gamma += m_learning_rate * gamma_gradient;
                        neuron.m_bn_beta += m_learning_rate * beta_gradient;
                    }
                }
            }
        }
        total_error /= inputs.size();
        double val_error = val_inputs.empty() ? 0.0 : evaluate(val_inputs, val_targets);
        if (epoch % 100 == 0 || epoch == m_epochs - 1) {
            std::cout << "Epoch " << epoch << ", Train MSE: " << total_error;
            if (!val_inputs.empty()) {
                std::cout << ", Validation MSE: " << val_error;
            }
            std::cout << std::endl;
        }
    }
}

double network::evaluate(const std::vector<std::vector<double>>& inputs,
                        const std::vector<std::vector<double>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (inputs.empty()) {
        return 0.0;
    }
    double total_error = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = predict(inputs[i]);
        for (size_t j = 0; j < targets[i].size(); ++j) {
            double diff = targets[i][j] - output[j];
            total_error += diff * diff;
        }
    }
    return total_error / inputs.size();
}

std::vector<double> network::predict(const std::vector<double>& input) {
    forward_propagate(input, false);
    std::vector<double> outputs;
    for (const auto& neuron : m_network.back()) {
        outputs.push_back(neuron.m_output);
    }
    return outputs;
}

void network::display_outputs() const {
    for (size_t layer = 0; layer < m_layers; ++layer) {
        std::cout << "Layer " << layer + 1 << " outputs: ";
        for (const auto& neuron : m_network[layer]) {
            std::cout << neuron.m_output << " ";
        }
        std::cout << std::endl;
    }
}

void network::save_model(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) {
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
            neuron.sync_weights_to_host();
            out << neuron.m_bias << " ";
            for (double w : neuron.m_weights) {
                out << w << " ";
            }
            out << neuron.m_bn_gamma << " " << neuron.m_bn_beta << " ";
            out << neuron.m_bn_mean << " " << neuron.m_bn_variance << "\n";
        }
    }
    out.close();
}

void network::load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
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
    std::mt19937 gen(std::random_device{}());
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
            n.sync_weights_to_device();
            layer_neurons.push_back(n);
        }
        m_network.push_back(layer_neurons);
    }
    in.close();
    initialize_gpu();
}