#include "network.hpp"

network::neuron::neuron(size_t number_of_weights, std::mt19937& gen)
    : m_number_of_weights(number_of_weights), m_bias(generate_random_value(gen)) {
    for (size_t i = 0; i < number_of_weights; ++i) {
        m_weights.push_back(generate_random_value(gen));
        m_weight_updates.push_back(0.0);
    }
}

double network::neuron::generate_random_value(std::mt19937& gen) {
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    return dis(gen);
}

void network::neuron::set_inputs(const std::vector<double>& inputs) {
    m_inputs = inputs;
}

double network::neuron::activate(double x, ActivationType type) const {
    switch (type) {
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::ReLU:
            return std::max(0.0, x);
        case ActivationType::Tanh:
            return std::tanh(x);
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double network::neuron::activate_derivative(double x, ActivationType type) const {
    switch (type) {
        case ActivationType::Sigmoid: {
            double sig = activate(x, ActivationType::Sigmoid);
            return sig * (1.0 - sig);
        }
        case ActivationType::ReLU:
            return x > 0 ? 1 : 0.0;
        case ActivationType::Tanh: {
            double tanh_x = std::tanh(x);
            return 1.0 - tanh_x * tanh_x;
        }
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double network::neuron::compute_output(ActivationType type) {
    double weighted_sum = 0.0;
    for (size_t i = 0; i < m_number_of_weights; ++i) {
        weighted_sum += m_weights[i] * m_inputs[i];
    }
    weighted_sum += m_bias;
    m_z = weighted_sum;
    m_output = activate(weighted_sum, type);
    return m_output;
}

network::network(std::vector<size_t> number_of_neurons_per_layer,
                std::vector<int> activations,
                double learning_rate, size_t epochs, size_t batch_size, double momentum)
    : m_number_of_neurons_per_layer(number_of_neurons_per_layer),
      m_learning_rate(learning_rate), m_epochs(epochs), m_batch_size(batch_size),
      m_momentum(momentum) {
    m_layers = number_of_neurons_per_layer.size();
    if (m_batch_size == 0) {
        throw std::invalid_argument("Batch size must be greater than 0");
    }
    if (m_momentum < 0.0 || m_momentum > 1.0) {
        throw std::invalid_argument("Momentum must be between 0 and 1");
    }
    if (activations.size() != m_layers - 1) {
        throw std::invalid_argument("Number of activations must match number of non-input layers");
    }
    for(size_t layer = 0; layer<m_layers-1; ++layer){
        m_activations.push_back(static_cast<ActivationType>(activations[layer]));
    }
    std::mt19937 gen(std::random_device{}());
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_network.push_back(std::vector<neuron>());
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
            m_network[layer].emplace_back(prev_layer_size, gen);
        }
    }
}

void network::forward_propagate(const std::vector<double>& input_values) {
    if (input_values.size() != m_number_of_neurons_per_layer[0]) {
        throw std::out_of_range("Input size does not match input layer size.");
    }
    for (size_t i = 0; i < m_network[0].size(); ++i) {
        m_network[0][i].m_output = input_values[i];
    }
    for (size_t layer = 1; layer < m_layers; ++layer) {
        std::vector<double> prev_outputs;
        for (const auto& neuron : m_network[layer - 1]) {
            prev_outputs.push_back(neuron.m_output);
        }
        for (auto& neuron : m_network[layer]) {
            neuron.set_inputs(prev_outputs);
            neuron.compute_output(m_activations[layer - 1]);
        }
    }
}

void network::backpropagate(const std::vector<double>& target_values) {
    if (target_values.size() != m_network.back().size()) {
        throw std::out_of_range("Target size does not match output layer size.");
    }
    auto& output_layer = m_network.back();
    for (size_t i = 0; i < output_layer.size(); ++i) {
        double error = target_values[i] - output_layer[i].m_output;
        output_layer[i].m_delta = error * output_layer[i].activate_derivative(
            output_layer[i].m_z, m_activations[m_layers - 2]);
    }
    for (int layer = static_cast<int>(m_layers) - 2; layer >= 0; --layer) {
        for (size_t i = 0; i < m_network[layer].size(); ++i) {
            double error = 0.0;
            for (const auto& next_neuron : m_network[layer + 1]) {
                error += next_neuron.m_weights[i] * next_neuron.m_delta;
            }
            m_network[layer][i].m_delta = error * m_network[layer][i].activate_derivative(
                m_network[layer][i].m_z, m_activations[layer - 1]);
        }
    }
}

void network::train(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
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
            for (size_t layer = 1; layer < m_layers; ++layer) {
                weight_gradients[layer].resize(m_network[layer].size());
                bias_gradients[layer].resize(m_network[layer].size(), 0.0);
                for (size_t j = 0; j < m_network[layer].size(); ++j) {
                    weight_gradients[layer][j].resize(m_network[layer][j].m_number_of_weights, 0.0);
                }
            }
            for (size_t b = 0; b < current_batch_size; ++b) {
                size_t idx = indices[batch_start + b];
                forward_propagate(inputs[idx]);
                backpropagate(targets[idx]);
                for (size_t layer = 1; layer < m_layers; ++layer) {
                    for (size_t j = 0; j < m_network[layer].size(); ++j) {
                        auto& neuron = m_network[layer][j];
                        for (size_t k = 0; k < neuron.m_number_of_weights; ++k) {
                            weight_gradients[layer][j][k] += neuron.m_delta * neuron.m_inputs[k];
                        }
                        bias_gradients[layer][j] += neuron.m_delta;
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
                    }
                    double bias_gradient = bias_gradients[layer][j] / current_batch_size;
                    neuron.m_bias_update = m_momentum * neuron.m_bias_update +
                                          m_learning_rate * bias_gradient;
                    neuron.m_bias += neuron.m_bias_update;
                }
            }
        }
        if (epoch % 100 == 0 || epoch == m_epochs - 1) {
            std::cout << "Epoch " << epoch << ", MSE: " << total_error / inputs.size() << std::endl;
        }
    }
}

std::vector<double> network::predict(const std::vector<double>& input) {
    forward_propagate(input);
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
    for (const auto& layer : m_network) {
        for (const auto& neuron : layer) {
            out << neuron.m_bias << " ";
            for (double w : neuron.m_weights) {
                out << w << " ";
            }
            out << "\n";
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
        m_activations[i] = static_cast<ActivationType>(act);
    }
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
            layer_neurons.push_back(n);
        }
        m_network.push_back(layer_neurons);
    }
    in.close();
}