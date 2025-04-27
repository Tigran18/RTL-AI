#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>

class network {
private:
    struct neuron {
        double m_output = 0.0;
        double m_bias = 0.0;
        size_t m_number_of_weights = 0;
        std::vector<double> m_weights;
        std::vector<double> m_inputs;
        double m_delta = 0.0;

        neuron(size_t number_of_weights=0, double bias = 0.0, unsigned seed = std::random_device{}());

        double generate_random_value(unsigned seed);
    
        void set_inputs(const std::vector<double>& inputs);

        double sigmoid();

        double sigmoid_derivative() const;
    };

    size_t m_layers;
    std::vector<size_t> m_number_of_neurons_per_layer;
    std::vector<std::vector<neuron>> m_neurons;
    double m_learning_rate; 
    size_t m_epochs;

public:
    network(std::vector<size_t> number_of_neurons_per_layer={}, double learning_rate = 0.9, size_t epochs = 4000);
  
    void forward_propagate(const std::vector<double>& input_values);

    void backpropagate(const std::vector<double>& target_values);

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets);

    void display_outputs() const;
};
