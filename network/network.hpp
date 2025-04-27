#pragma once
#include <vector>
#include <iostream>

class network {
private:
    struct neuron {
        double m_input = 0.0;
        double m_output = 0.0;
    };

    size_t m_layers; 
    std::vector<size_t> m_number_of_neurons_per_layer;
    std::vector<std::vector<neuron>> m_neurons;

public:
    network(size_t layers, std::vector<size_t> number_of_neurons_per_layer)
        : m_layers(layers), m_number_of_neurons_per_layer(number_of_neurons_per_layer) {
        for (size_t layer = 0; layer < m_layers; ++layer) {
            m_neurons.push_back(std::vector<neuron>());
            for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
                m_neurons[layer].push_back(neuron());
            }
        }
    }

    size_t get_layers() const {
        return m_layers;
    }

    void get_neurons() const {
        for (size_t layer = 0; layer < m_layers; ++layer) {
            std::cout << "Layer " << layer + 1 << " has " << m_number_of_neurons_per_layer[layer] << " neurons: ";
            for (const auto& n : m_neurons[layer]) {
                std::cout << n.m_output << " ";
            }
            std::cout << "\n";
        }
    }
};
