#include "network.hpp"

network::neuron::neuron(size_t number_of_weights, std::mt19937& gen)
    : m_number_of_weights(number_of_weights), m_bias(generate_random_value(gen)) {
    for (size_t i = 0; i < number_of_weights; ++i) {
        m_weights.push_back(generate_random_value(gen));
    }
}

double network::neuron::generate_random_value(std::mt19937& gen) {
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    return dis(gen);
}

void network::neuron::set_inputs(const std::vector<double>& inputs) {
    m_inputs = inputs;
}

double network::neuron::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double network::neuron::sigmoid() {
    double weighted_sum = 0.0;
    for (size_t i = 0; i < m_number_of_weights; ++i) {
        weighted_sum += m_weights[i] * m_inputs[i];
    }
    weighted_sum += m_bias;
    m_output = sigmoid(weighted_sum);
    return m_output;
}

double network::neuron::sigmoid_derivative() const {
    return m_output * (1.0 - m_output);
}

network::network(std::vector<size_t> number_of_neurons_per_layer, double learning_rate, size_t epochs)
    : m_number_of_neurons_per_layer(number_of_neurons_per_layer),
      m_learning_rate(learning_rate), m_epochs(epochs) {
    m_layers = m_number_of_neurons_per_layer.size();
    std::mt19937 gen(std::random_device{}());

    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_neurons.push_back(std::vector<neuron>());
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
            m_neurons[layer].emplace_back(prev_layer_size, gen);
        }
    }
}

void network::forward_propagate(const std::vector<double>& input_values) {
    if (input_values.size() != m_number_of_neurons_per_layer[0]) {
        throw std::out_of_range("Input size does not match input layer size.");
    }

    for (size_t i = 0; i < m_neurons[0].size(); ++i) {
        m_neurons[0][i].m_output = input_values[i];
    }

    for (size_t layer = 1; layer < m_layers; ++layer) {
        std::vector<double> prev_outputs;
        for (const auto& neuron : m_neurons[layer - 1]) {
            prev_outputs.push_back(neuron.m_output);
        }

        for (auto& neuron : m_neurons[layer]) {
            neuron.set_inputs(prev_outputs);
            neuron.sigmoid();
        }
    }
}

void network::backpropagate(const std::vector<double>& target_values) {
    auto& output_layer = m_neurons.back();
    for (size_t i = 0; i < output_layer.size(); ++i) {
        double error = target_values[i] - output_layer[i].m_output;
        output_layer[i].m_delta = error * output_layer[i].sigmoid_derivative();
    }

    for (int layer = static_cast<int>(m_layers) - 2; layer >= 0; --layer) {
        for (size_t i = 0; i < m_neurons[layer].size(); ++i) {
            double error = 0.0;
            for (const auto& next_neuron : m_neurons[layer + 1]) {
                error += next_neuron.m_weights[i] * next_neuron.m_delta;
            }
            m_neurons[layer][i].m_delta = error * m_neurons[layer][i].sigmoid_derivative();
        }
    }

    for (size_t layer = 1; layer < m_layers; ++layer) {
        for (auto& neuron : m_neurons[layer]) {
            for (size_t j = 0; j < neuron.m_number_of_weights; ++j) {
                neuron.m_weights[j] += m_learning_rate * neuron.m_delta * neuron.m_inputs[j];
            }
            neuron.m_bias += m_learning_rate * neuron.m_delta;
        }
    }
}

void network::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets) {
    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            forward_propagate(inputs[i]);
            backpropagate(targets[i]);

            for (size_t j = 0; j < targets[i].size(); ++j) {
                double diff = targets[i][j] - m_neurons.back()[j].m_output;
                total_error += diff * diff;
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
    for (const auto& neuron : m_neurons.back()) {
        outputs.push_back(neuron.m_output);
    }
    return outputs;
}

void network::display_outputs() const {
    for (size_t layer = 0; layer < m_layers; ++layer) {
        std::cout << "Layer " << layer + 1 << " outputs: ";
        for (const auto& neuron : m_neurons[layer]) {
            std::cout << neuron.m_output << " ";
        }
        std::cout << std::endl;
    }
}
