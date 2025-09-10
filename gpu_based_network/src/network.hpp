#pragma once
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <cublas_v2.h>

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

    // GPU-related members
    cublasHandle_t m_cublas_handle;
    std::vector<float*> d_weights;
    std::vector<float*> d_biases;
    std::vector<float*> d_layer_outputs;
    std::vector<float*> d_pre_acts;
    std::vector<float*> d_layer_deltas;
    std::vector<float*> d_gammas;
    std::vector<float*> d_betas;
    std::vector<float*> d_hats;
    std::vector<float*> d_mean;
    std::vector<float*> d_variance;
    std::vector<float*> d_gamma_grad;
    std::vector<float*> d_beta_grad;
    std::vector<float*> d_error;
    std::vector<float*> d_weight_gradients;
    std::vector<float*> d_bias_gradients;
    std::vector<float*> d_weight_m;
    std::vector<float*> d_weight_v;
    std::vector<float*> d_bias_m;
    std::vector<float*> d_bias_v;
    std::vector<float*> d_gamma_m;
    std::vector<float*> d_gamma_v;
    std::vector<float*> d_beta_m;
    std::vector<float*> d_beta_v;
    float* d_ones = nullptr;
    int m_block_size;

    void initialize_gpu();
    void cleanup_gpu();

public:
    network(std::vector<size_t> number_of_neurons_per_layer,
            std::vector<ActivationType> activations,
            float learning_rate = 0.01f,
            size_t epochs = 1000,
            size_t batch_size = 32,
            float momentum = 0.9f,
            bool use_batch_norm = false);
    ~network();

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
    void forward_propagate(const std::vector<std::vector<float>>& batch_inputs, bool training);
    void backpropagate(const std::vector<std::vector<float>>& batch_targets);
    const std::vector<size_t>& get_layer_sizes() const { return m_number_of_neurons_per_layer; }
};

// CUDA kernel declarations
extern "C" void fused_bias_activation_on_gpu(float* input, float* biases, float* output, int batch_size, int num, int act_type, float* pre_act);
extern "C" void compute_batch_norm_stats_batched_on_gpu(float* input, float* mean, float* variance, int batch_size, int num_features, int block_size);
extern "C" void compute_batch_norm_batched_on_gpu(float* input, float* output, float* mean,
                                                 float* variance, float* gamma, float* beta,
                                                 float epsilon, int batch_size, int num_features,
                                                 float* hat);
extern "C" void compute_output_delta_batched_on_gpu(float* outputs, float* pre_acts, float* targets,
                                                   float* deltas, int batch_size, int num, int act_type);
extern "C" void compute_hidden_delta_batched_on_gpu(float* delta_next, float* weights_next,
                                                   float* errors, int batch_size, int num_current,
                                                   int num_next);
extern "C" void compute_bn_grad_batched_on_gpu(float* error, float* hat, float* gamma_grad,
                                              float* beta_grad, int batch_size, int num);
extern "C" void update_error_for_bn_on_gpu(float* error, float* gamma, float* variance,
                                          float epsilon, int batch_size, int num);
extern "C" void compute_delta_from_error_on_gpu(float* error, float* pre_act, float* delta,
                                               int batch_size, int num, int act_type);
extern "C" void update_parameters_on_gpu(float* weights, float* weight_gradients, float* weight_velocity,
                                        float* biases, float* bias_gradients, float* bias_velocity,
                                        float* gammas, float* gamma_gradients,
                                        float* betas, float* beta_gradients,
                                        float learning_rate, float momentum,
                                        int num_weights, int num, bool use_bn);
extern "C" void apply_bn_backprop_correction_on_gpu(float* deltas, float* hats, float* gammas, float* gamma_grads, float* beta_grads, float* variances, float epsilon, int batch_size, int num);
extern "C" void adam_update_on_gpu(float* weights, float* weight_gradients, float* m_weights, float* v_weights,
                                   float* biases, float* bias_gradients, float* m_biases, float* v_biases,
                                   float* gammas, float* gamma_gradients, float* m_gammas, float* v_gammas,
                                   float* betas, float* beta_gradients, float* m_betas, float* v_betas,
                                   float learning_rate, float beta1, float beta2, float epsilon,
                                   int num_weights, int num, int t, bool use_bn);