#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declare CUDA kernels
extern "C" void compute_activation_on_gpu(double* input, double* output, int size, int act_type);
extern "C" void compute_batch_norm_on_gpu(double* input, double* output, double mean, double variance, double gamma, double beta, double epsilon, int size);

class network {
public:
    enum class ActivationType : int {
        Sigmoid = 0,
        ReLU = 1,
        Tanh = 2
    };
private:
    struct alignas(64) neuron {
        double m_output = 0.0;
        double m_bias = 0.0;
        size_t m_number_of_weights = 0;
        std::vector<double> m_weights;
        std::vector<double> m_inputs;
        double m_delta = 0.0;
        double m_z = 0.0;
        std::vector<double> m_weight_updates;
        double m_bias_update = 0.0;
        double m_bn_gamma = 1.0;
        double m_bn_beta = 0.0;
        double m_bn_mean = 0.0;
        double m_bn_variance = 1.0;
        double m_bn_normalized = 0.0;
        double m_bn_gamma_gradient = 0.0;
        double m_bn_beta_gradient = 0.0;
        double* d_weight_updates = nullptr;
        double* d_output = nullptr;

        neuron(size_t number_of_weights, std::mt19937& gen);
        ~neuron();
        double generate_random_value(std::mt19937& gen);
        void set_inputs(const std::vector<double>& inputs);
        double activate(double x, ActivationType type) const;
        double activate_derivative(double x, ActivationType type) const;
        double compute_output(ActivationType type, bool use_bn, double bn_mean, double bn_variance,
                             double bn_gamma, double bn_beta, bool training, double epsilon,
                             cublasHandle_t cublas_handle);
        void allocate_gpu_memory();
        void free_gpu_memory();
        void sync_weights_to_device();
        void sync_weights_to_host() const;
    };

    size_t m_layers;
    std::vector<size_t> m_number_of_neurons_per_layer;
    std::vector<std::vector<neuron>> m_network;
    double m_learning_rate;
    size_t m_epochs;
    size_t m_batch_size;
    double m_momentum;
    std::vector<ActivationType> m_activations;
    bool m_use_batch_norm = true;
    double m_bn_momentum = 0.9;
    double m_bn_epsilon = 1e-5;
    cublasHandle_t m_cublas_handle;
    std::vector<double*> d_layer_outputs;
    std::vector<double*> d_layer_deltas;

public:
    network(std::vector<size_t> number_of_neurons_per_layer,
            std::vector<ActivationType> activations,
            double learning_rate, size_t epochs,
            size_t batch_size, double momentum, bool use_batch_norm = true);
    ~network();
    void forward_propagate(const std::vector<double>& input_values, bool training = true);
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
    void initialize_gpu();
    void cleanup_gpu();
};


