#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>

class network {
public:
    enum class ActivationType : int {
        Sigmoid = 0,
        ReLU = 1,
        Tanh = 2
    };
private:
    struct neuron {
        double m_output = 0.0;
        double m_bias = 0.0;
        size_t m_number_of_weights = 0;
        std::vector<double> m_weights;
        std::vector<double> m_inputs;
        double m_delta = 0.0;
        double m_z = 0.0;
        std::vector<double> m_weight_updates;
        double m_bias_update = 0.0;

        neuron(size_t number_of_weights, std::mt19937& gen);
        double generate_random_value(std::mt19937& gen);
        void set_inputs(const std::vector<double>& inputs);
        double activate(double x, ActivationType type) const;
        double activate_derivative(double x, ActivationType type) const;
        double compute_output(ActivationType type);
    };

    size_t m_layers;
    std::vector<size_t> m_number_of_neurons_per_layer;
    std::vector<std::vector<neuron>> m_network;
    double m_learning_rate;
    size_t m_epochs;
    size_t m_batch_size;
    double m_momentum;
    std::vector<ActivationType> m_activations;

public:
    network(std::vector<size_t> number_of_neurons_per_layer,
            std::vector<ActivationType> activations,
            double learning_rate, size_t epochs,
            size_t batch_size, double momentum);
    void forward_propagate(const std::vector<double>& input_values);
    void backpropagate(const std::vector<double>& target_values);
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               const std::vector<std::vector<double>>& val_inputs = {},
               const std::vector<std::vector<double>>& val_targets = {});
    std::vector<double> predict(const std::vector<double>& input);
    double evaluate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets);
    void display_outputs() const;
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};