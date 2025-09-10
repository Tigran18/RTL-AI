#include "network.hpp"
#include <stdexcept>
#include <numeric>
#include <algorithm> // Added for std::shuffle

// Neuron implementation
network::neuron::neuron(size_t number_of_weights, std::mt19937& gen)
    : m_number_of_weights(number_of_weights), m_bias(generate_random_value(gen)),
      m_bn_gamma(1.0f), m_bn_beta(0.0f), m_bn_mean(0.0f), m_bn_variance(1.0f),
      m_bias_m(0.0f), m_bias_v(0.0f), m_gamma_m(0.0f), m_gamma_v(0.0f),
      m_beta_m(0.0f), m_beta_v(0.0f) {
    std::cout << "Creating neuron with " << number_of_weights << " weights" << std::endl;
    for (size_t i = 0; i < number_of_weights; ++i) {
        m_weights.push_back(generate_random_value(gen));
        m_weight_updates.push_back(0.0f);
        m_weight_m.push_back(0.0f);
        m_weight_v.push_back(0.0f);
    }
}

float network::neuron::generate_random_value(std::mt19937& gen) {
    float range = (m_number_of_weights > 0) ? 1.0f / sqrtf(static_cast<float>(m_number_of_weights)) : 1.0f;
    std::uniform_real_distribution<float> dis(-range, range);
    return dis(gen);
}

// Network implementation
network::network(std::vector<size_t> number_of_neurons_per_layer,
                std::vector<ActivationType> activations,
                float learning_rate, size_t epochs, size_t batch_size,
                float momentum, bool use_batch_norm)
    : m_number_of_neurons_per_layer(number_of_neurons_per_layer),
      m_learning_rate(learning_rate), m_epochs(epochs), m_batch_size(batch_size),
      m_momentum(momentum), m_activations(activations), m_use_batch_norm(use_batch_norm) {
    std::cout << "Initializing network with " << number_of_neurons_per_layer.size()
              << " layers..." << std::endl;
    m_layers = number_of_neurons_per_layer.size();

    // Validate parameters
    if (m_batch_size == 0) throw std::invalid_argument("Batch size must be greater than 0");
    if (m_momentum < 0.0f || m_momentum > 1.0f) throw std::invalid_argument("Momentum must be between 0 and 1");
    if (activations.size() != m_layers - 1) throw std::invalid_argument("Number of activations must match number of non-input layers");

    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize neurons
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_network.emplace_back();
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        for (size_t neuron_idx = 0; neuron_idx < m_number_of_neurons_per_layer[layer]; ++neuron_idx) {
            m_network[layer].emplace_back(prev_layer_size, gen);
        }
    }

    // Initialize storage for layer outputs and intermediates
    m_layer_outputs.resize(m_layers);
    m_pre_acts.resize(m_layers);
    m_layer_deltas.resize(m_layers);
    m_hats.resize(m_layers);
    m_mean.resize(m_layers);
    m_variance.resize(m_layers);
    m_gamma_grad.resize(m_layers);
    m_beta_grad.resize(m_layers);
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_layer_outputs[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
        m_pre_acts[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
        m_layer_deltas[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
        if (m_use_batch_norm && layer > 0 && layer < m_layers - 1) {
            m_hats[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
            m_mean[layer].resize(m_number_of_neurons_per_layer[layer]);
            m_variance[layer].resize(m_number_of_neurons_per_layer[layer]);
            m_gamma_grad[layer].resize(m_number_of_neurons_per_layer[layer]);
            m_beta_grad[layer].resize(m_number_of_neurons_per_layer[layer]);
        }
    }
}

void network::compute_batch_norm_stats(const std::vector<float>& input, std::vector<float>& mean, std::vector<float>& variance, size_t batch_size, size_t num_features) {
    for (size_t j = 0; j < num_features; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            sum += input[i * num_features + j];
        }
        mean[j] = sum / batch_size;
        float var_sum = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            float diff = input[i * num_features + j] - mean[j];
            var_sum += diff * diff;
        }
        variance[j] = var_sum / batch_size;
    }
}

void network::apply_batch_norm(std::vector<float>& output, const std::vector<float>& input, std::vector<float>& mean, std::vector<float>& variance, std::vector<float>& gamma, std::vector<float>& beta, std::vector<float>& hat, size_t batch_size, size_t num_features) {
    for (size_t idx = 0; idx < batch_size * num_features; ++idx) {
        size_t j = idx % num_features;
        float stddev = std::sqrt(variance[j] + m_bn_epsilon);
        float normalized = (input[idx] - mean[j]) / stddev;
        hat[idx] = normalized;
        output[idx] = gamma[j] * normalized + beta[j];
    }
}

void network::apply_activation(std::vector<float>& output, const std::vector<float>& input, size_t layer, bool training) {
    size_t num = m_number_of_neurons_per_layer[layer];
    size_t batch_size = input.size() / num;
    int act_type = static_cast<int>(m_activations[layer - 1]);

    for (size_t idx = 0; idx < batch_size * num; ++idx) {
        size_t j = idx % num;
        float val = input[idx] + m_network[layer][j].m_bias;
        if (training) m_pre_acts[layer][idx] = val;
        if (act_type == 0) { // Sigmoid
            output[idx] = 1.0f / (1.0f + std::exp(-val));
        } else if (act_type == 1) { // ReLU
            output[idx] = std::fmax(val, 0.0f);
        } else { // Tanh
            output[idx] = std::tanh(val);
        }
    }
}

void network::forward_propagate(const std::vector<std::vector<float>>& batch_inputs, bool training) {
    size_t B = batch_inputs.size();
    if (B == 0) return;
    size_t input_dim = m_number_of_neurons_per_layer[0];
    if (batch_inputs[0].size() != input_dim)
        throw std::out_of_range("Input size does not match input layer size.");

    // Copy batched inputs
    m_layer_outputs[0].resize(B * input_dim);
    for (size_t i = 0; i < B; ++i) {
        std::copy(batch_inputs[i].begin(), batch_inputs[i].end(), m_layer_outputs[0].begin() + i * input_dim);
    }

    // Forward pass through layers
    for (size_t layer = 1; layer < m_layers; ++layer) {
        size_t num = m_number_of_neurons_per_layer[layer];
        size_t prev = m_number_of_neurons_per_layer[layer - 1];
        std::vector<float> pre_acts(B * num, 0.0f);

        // Linear transformation
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < num; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < prev; ++k) {
                    sum += m_layer_outputs[layer - 1][i * prev + k] * m_network[layer][j].m_weights[k];
                }
                pre_acts[i * num + j] = sum;
            }
        }

        if (m_use_batch_norm && layer < m_layers - 1) {
            compute_batch_norm_stats(pre_acts, m_mean[layer], m_variance[layer], B, num);
            for (size_t j = 0; j < num; ++j) {
                m_network[layer][j].m_bn_mean = m_bn_momentum * m_network[layer][j].m_bn_mean + (1.0f - m_bn_momentum) * m_mean[layer][j];
                m_network[layer][j].m_bn_variance = m_bn_momentum * m_network[layer][j].m_bn_variance + (1.0f - m_bn_momentum) * m_variance[layer][j];
            }
            std::vector<float> gamma(num), beta(num);
            for (size_t j = 0; j < num; ++j) {
                gamma[j] = m_network[layer][j].m_bn_gamma;
                beta[j] = m_network[layer][j].m_bn_beta;
            }
            if (training) {
                apply_batch_norm(m_layer_outputs[layer], pre_acts, m_mean[layer], m_variance[layer], gamma, beta, m_hats[layer], B, num);
            } else {
                // Use running mean and variance for inference
                apply_batch_norm(m_layer_outputs[layer], pre_acts, m_mean[layer], m_variance[layer], gamma, beta, m_hats[layer], B, num);
            }
            apply_activation(m_layer_outputs[layer], m_layer_outputs[layer], layer, training);
        } else {
            apply_activation(m_layer_outputs[layer], pre_acts, layer, training);
        }
    }
}

void network::compute_output_delta(std::vector<float>& deltas, const std::vector<float>& outputs, const std::vector<float>& pre_acts, const std::vector<float>& targets, size_t batch_size, size_t num, int act_type) {
    for (size_t idx = 0; idx < batch_size * num; ++idx) {
        float diff = outputs[idx] - targets[idx];
        float act_deriv;
        if (act_type == 0) { // Sigmoid
            act_deriv = outputs[idx] * (1.0f - outputs[idx]);
        } else if (act_type == 1) { // ReLU
            act_deriv = pre_acts[idx] > 0 ? 1.0f : 0.0f;
        } else { // Tanh
            act_deriv = 1.0f - outputs[idx] * outputs[idx];
        }
        deltas[idx] = diff * act_deriv;
    }
}

void network::compute_hidden_delta(std::vector<float>& errors, const std::vector<float>& delta_next, const std::vector<std::vector<float>>& weights_next, size_t batch_size, size_t num_current, size_t num_next) {
    for (size_t idx = 0; idx < batch_size * num_current; ++idx) {
        size_t i = idx / num_current;
        size_t j = idx % num_current;
        float sum = 0.0f;
        for (size_t k = 0; k < num_next; ++k) {
            sum += delta_next[i * num_next + k] * weights_next[k][j];
        }
        errors[idx] = sum;
    }
}

void network::compute_bn_grad(std::vector<float>& gamma_grad, std::vector<float>& beta_grad, const std::vector<float>& error, const std::vector<float>& hat, size_t batch_size, size_t num) {
    for (size_t j = 0; j < num; ++j) {
        float g_grad = 0.0f;
        float b_grad = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = i * num + j;
            g_grad += error[idx] * hat[idx];
            b_grad += error[idx];
        }
        gamma_grad[j] = g_grad;
        beta_grad[j] = b_grad;
    }
}

void network::update_error_for_bn(std::vector<float>& error, const std::vector<float>& gamma, const std::vector<float>& variance, size_t batch_size, size_t num) {
    for (size_t idx = 0; idx < batch_size * num; ++idx) {
        size_t j = idx % num;
        float stddev = std::sqrt(variance[j] + m_bn_epsilon);
        error[idx] *= gamma[j] / stddev;
    }
}

void network::compute_delta_from_error(std::vector<float>& delta, const std::vector<float>& error, const std::vector<float>& pre_act, size_t batch_size, size_t num, int act_type) {
    for (size_t idx = 0; idx < batch_size * num; ++idx) {
        float act_deriv;
        if (act_type == 0) { // Sigmoid
            float sigmoid = 1.0f / (1.0f + std::exp(-pre_act[idx]));
            act_deriv = sigmoid * (1.0f - sigmoid);
        } else if (act_type == 1) { // ReLU
            act_deriv = pre_act[idx] > 0 ? 1.0f : 0.0f;
        } else { // Tanh
            float tanh_val = std::tanh(pre_act[idx]);
            act_deriv = 1.0f - tanh_val * tanh_val;
        }
        delta[idx] = error[idx] * act_deriv;
    }
}

void network::apply_bn_backprop_correction(std::vector<float>& deltas, const std::vector<float>& hats, const std::vector<float>& gammas, const std::vector<float>& gamma_grads, const std::vector<float>& beta_grads, const std::vector<float>& variances, size_t batch_size, size_t num) {
    for (size_t idx = 0; idx < batch_size * num; ++idx) {
        size_t j = idx % num;
        float stddev = std::sqrt(variances[j] + m_bn_epsilon);
        float correction1 = gammas[j] * beta_grads[j] / stddev;
        float correction2 = gammas[j] * gamma_grads[j] / stddev;
        deltas[idx] -= correction1 + hats[idx] * correction2;
    }
}

void network::adam_update(std::vector<std::vector<float>>& weights, std::vector<float>& biases, std::vector<float>& gammas, std::vector<float>& betas, const std::vector<std::vector<float>>& weight_gradients, const std::vector<float>& bias_gradients, const std::vector<float>& gamma_gradients, const std::vector<float>& beta_gradients, size_t layer, size_t t) {
    float beta1_t = std::pow(m_beta1, t);
    float beta2_t = std::pow(m_beta2, t);

    for (size_t j = 0; j < m_number_of_neurons_per_layer[layer]; ++j) {
        auto& neuron = m_network[layer][j];
        for (size_t k = 0; k < neuron.m_weights.size(); ++k) {
            neuron.m_weight_m[k] = m_beta1 * neuron.m_weight_m[k] + (1.0f - m_beta1) * weight_gradients[j][k];
            neuron.m_weight_v[k] = m_beta2 * neuron.m_weight_v[k] + (1.0f - m_beta2) * weight_gradients[j][k] * weight_gradients[j][k];
            float m_hat = neuron.m_weight_m[k] / (1.0f - beta1_t);
            float v_hat = neuron.m_weight_v[k] / (1.0f - beta2_t);
            neuron.m_weights[k] -= m_learning_rate * m_hat / (std::sqrt(v_hat) + m_epsilon);
        }
        neuron.m_bias_m = m_beta1 * neuron.m_bias_m + (1.0f - m_beta1) * bias_gradients[j];
        neuron.m_bias_v = m_beta2 * neuron.m_bias_v + (1.0f - m_beta2) * bias_gradients[j] * bias_gradients[j];
        float m_hat = neuron.m_bias_m / (1.0f - beta1_t);
        float v_hat = neuron.m_bias_v / (1.0f - beta2_t);
        neuron.m_bias -= m_learning_rate * m_hat / (std::sqrt(v_hat) + m_epsilon);

        if (m_use_batch_norm && layer < m_layers - 1) {
            neuron.m_gamma_m = m_beta1 * neuron.m_gamma_m + (1.0f - m_beta1) * gamma_gradients[j];
            neuron.m_gamma_v = m_beta2 * neuron.m_gamma_v + (1.0f - m_beta2) * gamma_gradients[j] * gamma_gradients[j];
            float m_hat_gamma = neuron.m_gamma_m / (1.0f - beta1_t);
            float v_hat_gamma = neuron.m_gamma_v / (1.0f - beta2_t);
            neuron.m_bn_gamma -= m_learning_rate * m_hat_gamma / (std::sqrt(v_hat_gamma) + m_epsilon);

            neuron.m_beta_m = m_beta1 * neuron.m_beta_m + (1.0f - m_beta1) * beta_gradients[j];
            neuron.m_beta_v = m_beta2 * neuron.m_beta_v + (1.0f - m_beta2) * beta_gradients[j] * beta_gradients[j];
            float m_hat_beta = neuron.m_beta_m / (1.0f - beta1_t);
            float v_hat_beta = neuron.m_beta_v / (1.0f - beta2_t);
            neuron.m_bn_beta -= m_learning_rate * m_hat_beta / (std::sqrt(v_hat_beta) + m_epsilon);
        }
    }
}

void network::backpropagate(const std::vector<std::vector<float>>& batch_targets) {
    size_t B = batch_targets.size();
    size_t output_dim = m_number_of_neurons_per_layer[m_layers - 1];
    std::vector<float> targets(B * output_dim);
    for (size_t i = 0; i < B; ++i) {
        std::copy(batch_targets[i].begin(), batch_targets[i].end(), targets.begin() + i * output_dim);
    }

    // Output layer delta
    compute_output_delta(m_layer_deltas[m_layers - 1], m_layer_outputs[m_layers - 1], m_pre_acts[m_layers - 1], targets, B, output_dim, static_cast<int>(m_activations[m_layers - 2]));

    // Hidden layers
    for (size_t layer = m_layers - 1; layer > 0; --layer) {
        size_t num = m_number_of_neurons_per_layer[layer];
        size_t prev = m_number_of_neurons_per_layer[layer - 1];
        std::vector<std::vector<float>> weight_gradients(num, std::vector<float>(prev, 0.0f));
        std::vector<float> bias_gradients(num, 0.0f);
        std::vector<float> gamma_gradients(num, 0.0f);
        std::vector<float> beta_gradients(num, 0.0f);

        // Compute gradients
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < num; ++j) {
                for (size_t k = 0; k < prev; ++k) {
                    weight_gradients[j][k] += m_layer_deltas[layer][i * num + j] * m_layer_outputs[layer - 1][i * prev + k] / B;
                }
                bias_gradients[j] += m_layer_deltas[layer][i * num + j] / B;
            }
        }

        if (m_use_batch_norm && layer < m_layers - 1) {
            compute_bn_grad(gamma_gradients, beta_gradients, m_layer_deltas[layer], m_hats[layer], B, num);
            update_error_for_bn(m_layer_deltas[layer], std::vector<float>(num, m_network[layer][0].m_bn_gamma), m_variance[layer], B, num);
            apply_bn_backprop_correction(m_layer_deltas[layer], m_hats[layer], std::vector<float>(num, m_network[layer][0].m_bn_gamma), gamma_gradients, beta_gradients, m_variance[layer], B, num);
            compute_delta_from_error(m_layer_deltas[layer], m_layer_deltas[layer], m_pre_acts[layer], B, num, static_cast<int>(m_activations[layer - 1]));
        }

        if (layer > 1) {
            std::vector<float> errors(B * prev, 0.0f);
            std::vector<std::vector<float>> weights_next(num);
            for (size_t j = 0; j < num; ++j) {
                weights_next[j] = m_network[layer][j].m_weights;
            }
            compute_hidden_delta(errors, m_layer_deltas[layer], weights_next, B, prev, num);
            compute_delta_from_error(m_layer_deltas[layer - 1], errors, m_pre_acts[layer - 1], B, prev, static_cast<int>(m_activations[layer - 2]));
        }

        // Update parameters
        std::vector<float> gammas(num), betas(num);
        if (m_use_batch_norm && layer < m_layers - 1) {
            for (size_t j = 0; j < num; ++j) {
                gammas[j] = m_network[layer][j].m_bn_gamma;
                betas[j] = m_network[layer][j].m_bn_beta;
            }
        }
        adam_update(weight_gradients, bias_gradients, gammas, betas, weight_gradients, bias_gradients, gamma_gradients, beta_gradients, layer, layer);
    }
}

void network::train(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    const std::vector<std::vector<float>>& val_inputs,
                    const std::vector<std::vector<float>>& val_targets) {
    std::cout << "Starting training with " << inputs.size() << " samples..." << std::endl;
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (!val_inputs.empty() && val_inputs.size() != val_targets.size()) {
        throw std::invalid_argument("Number of validation inputs must match number of validation targets");
    }
    if (inputs.empty()) {
        std::cout << "No training data provided." << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indices(inputs.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    int t = 0;
    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        float total_error = 0.0f;

        for (size_t batch_start = 0; batch_start < inputs.size(); batch_start += m_batch_size) {
            size_t current_batch_size = std::min(m_batch_size, inputs.size() - batch_start);
            std::vector<std::vector<float>> batch_inputs(current_batch_size);
            std::vector<std::vector<float>> batch_targets(current_batch_size);

            for (size_t b = 0; b < current_batch_size; ++b) {
                size_t idx = indices[batch_start + b];
                batch_inputs[b] = inputs[idx];
                batch_targets[b] = targets[idx];
            }

            forward_propagate(batch_inputs, true);
            backpropagate(batch_targets);
            t++;

            // Compute batch error
            size_t output_dim = m_number_of_neurons_per_layer[m_layers - 1];
            for (size_t b = 0; b < current_batch_size; ++b) {
                for (size_t j = 0; j < output_dim; ++j) {
                    float diff = m_layer_outputs[m_layers - 1][b * output_dim + j] - batch_targets[b][j];
                    total_error += diff * diff;
                }
            }
        }

        total_error /= inputs.size();
        float val_error = val_inputs.empty() ? 0.0f : evaluate(val_inputs, val_targets);
        if (epoch % 100 == 0 || epoch == m_epochs - 1) {
            std::cout << "Epoch " << epoch << ", Train MSE: " << total_error;
            if (!val_inputs.empty()) {
                std::cout << ", Validation MSE: " << val_error;
            }
            std::cout << std::endl;
        }
    }
}

float network::evaluate(const std::vector<std::vector<float>>& inputs,
                       const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (inputs.empty()) {
        std::cout << "No evaluation data provided." << std::endl;
        return 0.0f;
    }
    float total_error = 0.0f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = predict(inputs[i]);
        for (size_t j = 0; j < targets[i].size(); ++j) {
            float diff = targets[i][j] - output[j];
            total_error += diff * diff;
        }
    }
    float mse = total_error / inputs.size();
    std::cout << "Evaluation MSE: " << mse << std::endl;
    return mse;
}

std::vector<float> network::predict(const std::vector<float>& input) {
    forward_propagate({input}, false);
    size_t output_dim = m_number_of_neurons_per_layer[m_layers - 1];
    std::vector<float> output(output_dim);
    std::copy(m_layer_outputs[m_layers - 1].begin(), m_layer_outputs[m_layers - 1].begin() + output_dim, output.begin());
    return output;
}

void network::display_outputs() const {
    for (size_t layer = 0; layer < m_layers; ++layer) {
        std::cout << "Layer " << layer + 1 << " outputs: ";
        for (float val : m_layer_outputs[layer]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void network::save_model(const std::string& filename) const {
    std::cout << "Saving model to " << filename << "..." << std::endl;
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open file for saving model: " << filename << std::endl;
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
    out << m_use_batch_norm << "\n";
    for (const auto& layer : m_network) {
        for (const auto& neuron : layer) {
            out << neuron.m_bias << " ";
            for (float w : neuron.m_weights) {
                out << w << " ";
            }
            out << neuron.m_bn_gamma << " " << neuron.m_bn_beta << " ";
            out << neuron.m_bn_mean << " " << neuron.m_bn_variance << "\n";
        }
    }
    out.close();
    std::cout << "Model saved successfully." << std::endl;
}

void network::load_model(const std::string& filename) {
    std::cout << "Loading model from " << filename << "..." << std::endl;
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Failed to open file for loading model: " << filename << std::endl;
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
        if (act < 0 || act > static_cast<int>(ActivationType::Tanh)) {
            throw std::runtime_error("Invalid activation type in model file");
        }
        m_activations[i] = static_cast<ActivationType>(act);
    }
    int use_bn;
    in >> use_bn;
    m_use_batch_norm = use_bn;
    m_network.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t layer = 0; layer < m_layers; ++layer) {
        size_t prev_layer_size = (layer == 0) ? 0 : m_number_of_neurons_per_layer[layer - 1];
        std::vector<neuron> layer_neurons;
        for (size_t j = 0; j < m_number_of_neurons_per_layer[layer]; ++j) {
            neuron n(prev_layer_size, gen);
            in >> n.m_bias;
            for (size_t k = 0; k < prev_layer_size; ++k) {
                in >> n.m_weights[k];
            }
            in >> n.m_bn_gamma >> n.m_bn_beta >> n.m_bn_mean >> n.m_bn_variance;
            layer_neurons.push_back(n);
        }
        m_network.push_back(layer_neurons);
    }
    in.close();
    std::cout << "Model loaded successfully." << std::endl;

    // Reinitialize storage
    m_layer_outputs.resize(m_layers);
    m_pre_acts.resize(m_layers);
    m_layer_deltas.resize(m_layers);
    m_hats.resize(m_layers);
    m_mean.resize(m_layers);
    m_variance.resize(m_layers);
    m_gamma_grad.resize(m_layers);
    m_beta_grad.resize(m_layers);
    for (size_t layer = 0; layer < m_layers; ++layer) {
        m_layer_outputs[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
        m_pre_acts[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
        m_layer_deltas[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
        if (m_use_batch_norm && layer > 0 && layer < m_layers - 1) {
            m_hats[layer].resize(m_batch_size * m_number_of_neurons_per_layer[layer]);
            m_mean[layer].resize(m_number_of_neurons_per_layer[layer]);
            m_variance[layer].resize(m_number_of_neurons_per_layer[layer]);
            m_gamma_grad[layer].resize(m_number_of_neurons_per_layer[layer]);
            m_beta_grad[layer].resize(m_number_of_neurons_per_layer[layer]);
        }
    }
}