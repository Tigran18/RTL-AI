#include "network.hpp"

network::neuron::neuron(size_t number_of_weights, double bias, unsigned seed)
    : m_number_of_weights(number_of_weights), m_bias(generate_random_value(seed)) {
        for (size_t weight = 0; weight < number_of_weights; ++weight) {
        m_weights.push_back(generate_random_value(seed));
    }
}
    
double network::neuron::generate_random_value(unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    return dis(gen);
}
    
void network::neuron::set_inputs(const std::vector<double>& inputs) {
    m_inputs = inputs;
}
    
double network::neuron::sigmoid() {
    double weighted_sum = 0.0;
    for (size_t i = 0; i < m_number_of_weights; ++i) {
        weighted_sum += m_weights[i] * m_inputs[i];
    }
    weighted_sum += m_bias;
    return 1.0 / (1.0 + exp(-weighted_sum));
}

double network::neuron::sigmoid_derivative() const {
    return m_output * (1.0 - m_output);
}

network::network(std::vector<size_t> number_of_neurons_per_layer, double learning_rate, size_t epochs)
    : m_number_of_neurons_per_layer(number_of_neurons_per_layer), m_learning_rate(learning_rate), m_epochs(epochs) {
    m_layers=number_of_neurons_per_layer.size();
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_neurons.push_back(std::vector<neuron>());
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
            m_neurons[layer].push_back(neuron(prev_layer_size));
        }
    }
}  

void network::forward_propagate(const std::vector<double>& input_values) {
    if (input_values.size() != m_number_of_neurons_per_layer[0]) {
        throw std::out_of_range("Sizes don't match.\n");
    }
  
    for (size_t i = 0; i < m_neurons[0].size(); ++i) {
        m_neurons[0][i].m_output = input_values[i];
    }
  
    for (size_t layer = 1; layer < m_layers; ++layer) {
        std::vector<double> prev_layer_outputs;
        for (const auto& prev_neuron : m_neurons[layer - 1]) {
            prev_layer_outputs.push_back(prev_neuron.m_output);
        }
  
        for (size_t i = 0; i < m_neurons[layer].size(); ++i) {
            m_neurons[layer][i].set_inputs(prev_layer_outputs);
            m_neurons[layer][i].m_output = m_neurons[layer][i].sigmoid();
        }
    }
}

void network::backpropagate(const std::vector<double>& target_values) {
    for (size_t i = 0; i < m_neurons.back().size(); ++i) {
        neuron& n = m_neurons.back()[i];
        double error = target_values[i] - n.m_output;
        n.m_delta = error * n.sigmoid_derivative();
    }

    for (int layer = m_layers - 2; layer >= 0; --layer) {
        for (size_t i = 0; i < m_neurons[layer].size(); ++i) {
            neuron& n = m_neurons[layer][i];
            double error = 0.0;
            for (const auto& next_neuron : m_neurons[layer + 1]) {
                error += next_neuron.m_weights[i] * next_neuron.m_delta;
            }
            n.m_delta = error * n.sigmoid_derivative();
        }
    }

    for (size_t layer = 1; layer < m_layers; ++layer) {
        for (size_t i = 0; i < m_neurons[layer].size(); ++i) {
            neuron& n = m_neurons[layer][i];
            for (size_t j = 0; j < n.m_number_of_weights; ++j) {
                n.m_weights[j] += m_learning_rate * n.m_delta * n.m_inputs[j];
            }
            n.m_bias += m_learning_rate * n.m_delta;
        }
    }
}

void network::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets) {
    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward_propagate(inputs[i]);
            backpropagate(targets[i]);
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " completed" << std::endl;
        }
    }
}

void network::display_outputs() const {
    for (size_t layer = 0; layer < m_layers; ++layer) {
        std::cout << "Layer " << layer + 1 << " outputs: ";
        for (size_t i = 0; i < m_neurons[layer].size(); ++i) {
            std::cout << m_neurons[layer][i].m_output << " ";
        }
        std::cout << std::endl;
    }
}