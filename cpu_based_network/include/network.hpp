#pragma once
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>

class network {
public:
    enum class ActivationType { Sigmoid, ReLU, Tanh };

private:
    class neuron {
    public:
        size_t m_number_of_weights;
        float m_bias;
        float m_bias_update;
        float m_bn_gamma, m_bn_beta;
        float m_bn_mean, m_bn_variance;
        std::vector<float> m_weights;
        std::vector<float> m_weight_updates;
        std::vector<float> m_weight_m; // First moment (Adam)
        std::vector<float> m_weight_v; // Second moment (Adam)
        float m_bias_m, m_bias_v;
        float m_gamma_m, m_gamma_v;
        float m_beta_m, m_beta_v;

        neuron(size_t number_of_weights, std::mt19937& gen);
        float generate_random_value(std::mt19937& gen);
    };

    std::vector<std::vector<neuron>> m_network;
    std::vector<size_t> m_number_of_neurons_per_layer;
    std::vector<ActivationType> m_activations;
    float m_learning_rate;
    size_t m_epochs;
    size_t m_batch_size;
    float m_momentum;
    bool m_use_batch_norm;
    float m_bn_momentum = 0.9f;
    float m_bn_epsilon = 1e-5f;
    size_t m_layers;

    // Adam optimizer parameters
    float m_beta1 = 0.9f;
    float m_beta2 = 0.999f;
    float m_epsilon = 1e-8f;

    // CPU storage for layer outputs and intermediate values
    std::vector<std::vector<float>> m_layer_outputs;
    std::vector<std::vector<float>> m_pre_acts;
    std::vector<std::vector<float>> m_layer_deltas;
    std::vector<std::vector<float>> m_hats;
    std::vector<std::vector<float>> m_mean;
    std::vector<std::vector<float>> m_variance;
    std::vector<std::vector<float>> m_gamma_grad;
    std::vector<std::vector<float>> m_beta_grad;

    void forward_propagate(const std::vector<std::vector<float>>& batch_inputs, bool training);
    void backpropagate(const std::vector<std::vector<float>>& batch_targets);
    void apply_activation(std::vector<float>& output, const std::vector<float>& input, size_t layer, bool training);
    void compute_batch_norm_stats(const std::vector<float>& input, std::vector<float>& mean, std::vector<float>& variance, size_t batch_size, size_t num_features);
    void apply_batch_norm(std::vector<float>& output, const std::vector<float>& input, std::vector<float>& mean, std::vector<float>& variance, std::vector<float>& gamma, std::vector<float>& beta, std::vector<float>& hat, size_t batch_size, size_t num_features);
    void compute_output_delta(std::vector<float>& deltas, const std::vector<float>& outputs, const std::vector<float>& pre_acts, const std::vector<float>& targets, size_t batch_size, size_t num, int act_type);
    void compute_hidden_delta(std::vector<float>& errors, const std::vector<float>& delta_next, const std::vector<std::vector<float>>& weights_next, size_t batch_size, size_t num_current, size_t num_next);
    void compute_bn_grad(std::vector<float>& gamma_grad, std::vector<float>& beta_grad, const std::vector<float>& error, const std::vector<float>& hat, size_t batch_size, size_t num);
    void update_error_for_bn(std::vector<float>& error, const std::vector<float>& gamma, const std::vector<float>& variance, size_t batch_size, size_t num);
    void compute_delta_from_error(std::vector<float>& delta, const std::vector<float>& error, const std::vector<float>& pre_act, size_t batch_size, size_t num, int act_type);
    void apply_bn_backprop_correction(std::vector<float>& deltas, const std::vector<float>& hats, const std::vector<float>& gammas, const std::vector<float>& gamma_grads, const std::vector<float>& beta_grads, const std::vector<float>& variances, size_t batch_size, size_t num);
    void adam_update(std::vector<std::vector<float>>& weights, std::vector<float>& biases, std::vector<float>& gammas, std::vector<float>& betas, const std::vector<std::vector<float>>& weight_gradients, const std::vector<float>& bias_gradients, const std::vector<float>& gamma_gradients, const std::vector<float>& beta_gradients, size_t layer, size_t t);

public:
    network(std::vector<size_t> number_of_neurons_per_layer,
            std::vector<ActivationType> activations,
            float learning_rate = 0.01f,
            size_t epochs = 1000,
            size_t batch_size = 32,
            float momentum = 0.9f,
            bool use_batch_norm = false);
    ~network() = default;

    void train(const std::vector<std::vector<float>>& inputs,
               const std::vector<std::vector<float>>& targets,
               const std::vector<std::vector<float>>& val_inputs = {},
               const std::vector<std::vector<float>>& val_targets = {});
    float evaluate(const std::vector<std::vector<float>>& inputs,
                   const std::vector<std::vector<float>>& targets);
    std::vector<float> predict(const std::vector<float>& input);
    void display_outputs() const;
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};