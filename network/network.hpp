#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <stdexcept>

class network {
private:
    struct neuron {
        double m_output = 0.0;
        double m_bias = 0.0;
        size_t m_number_of_weights = 0;
        std::vector<double> m_weights;
        std::vector<double> m_inputs;
        double m_delta = 0.0;

        neuron(size_t number_of_weights, std::mt19937& gen);
        double generate_random_value(std::mt19937& gen);
        void set_inputs(const std::vector<double>& inputs);
        double sigmoid(double x) const;
        double sigmoid();
        double sigmoid_derivative() const;
    };

    size_t m_layers;
    std::vector<size_t> m_number_of_neurons_per_layer;
    std::vector<std::vector<neuron>> m_network;
    double m_learning_rate;
    size_t m_epochs;

public:
    network(std::vector<size_t> number_of_neurons_per_layer = {}, double learning_rate = 0.1, size_t epochs = 4000);
    void forward_propagate(const std::vector<double>& input_values);
    void backpropagate(const std::vector<double>& target_values);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets);
    std::vector<double> predict(const std::vector<double>& input);
    void display_outputs() const;
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};
