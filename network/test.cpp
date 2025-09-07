#include "network.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// === Utility: Generate noisy XOR data ===
void generate_xor_data(std::vector<std::vector<float>>& inputs,
                      std::vector<std::vector<float>>& targets,
                      size_t num_samples, float noise_stddev) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> noise(0.0f, noise_stddev);
    std::uniform_int_distribution<int> binary(0, 1);

    inputs.clear();
    targets.clear();

    for (size_t i = 0; i < num_samples; ++i) {
        float x1 = binary(gen);
        float x2 = binary(gen);
        float y = (x1 != x2) ? 1.0f : 0.0f;

        inputs.push_back({x1 + noise(gen), x2 + noise(gen)});
        targets.push_back({y});
    }
    std::cout << "Generated " << inputs.size() << " XOR samples." << std::endl << std::flush;
}

// === Utility: Split into train, val, test ===
void split_data(const std::vector<std::vector<float>>& inputs,
                const std::vector<std::vector<float>>& targets,
                std::vector<std::vector<float>>& train_inputs,
                std::vector<std::vector<float>>& train_targets,
                std::vector<std::vector<float>>& val_inputs,
                std::vector<std::vector<float>>& val_targets,
                std::vector<std::vector<float>>& test_inputs,
                std::vector<std::vector<float>>& test_targets,
                float train_ratio = 0.6f, float val_ratio = 0.2f) {
    size_t n = inputs.size();
    size_t train_size = static_cast<size_t>(n * train_ratio);
    size_t val_size = static_cast<size_t>(n * val_ratio);

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    train_inputs.clear();
    train_targets.clear();
    val_inputs.clear();
    val_targets.clear();
    test_inputs.clear();
    test_targets.clear();

    for (size_t i = 0; i < train_size; ++i) {
        train_inputs.push_back(inputs[indices[i]]);
        train_targets.push_back(targets[indices[i]]);
    }
    for (size_t i = train_size; i < train_size + val_size; ++i) {
        val_inputs.push_back(inputs[indices[i]]);
        val_targets.push_back(targets[indices[i]]);
    }
    for (size_t i = train_size + val_size; i < n; ++i) {
        test_inputs.push_back(inputs[indices[i]]);
        test_targets.push_back(targets[indices[i]]);
    }
    std::cout << "Split data: Train=" << train_inputs.size() << ", Val=" << val_inputs.size()
              << ", Test=" << test_inputs.size() << std::endl << std::flush;
}

// === Train a network on XOR ===
void train_xor_example(bool use_batch_norm, const std::string& model_name) {
    std::vector<std::vector<float>> inputs, targets;
    generate_xor_data(inputs, targets, 1000, 0.05f); // Reduced noise for stability

    std::vector<std::vector<float>> train_inputs, train_targets;
    std::vector<std::vector<float>> val_inputs, val_targets;
    std::vector<std::vector<float>> test_inputs, test_targets;

    split_data(inputs, targets,
               train_inputs, train_targets,
               val_inputs, val_targets,
               test_inputs, test_targets);

    // Simplified network architecture
    std::vector<size_t> layers = {2, 8, 1};
    std::vector<network::ActivationType> activations = {
        network::ActivationType::ReLU,
        network::ActivationType::Sigmoid
    };

    // Lower learning rate and more epochs
    network net(layers, activations, 0.001f, 5000, 64, 0.9f, use_batch_norm);

    std::cout << "Training XOR network " << (use_batch_norm ? "with" : "without")
              << " Batch Normalization..." << std::endl << std::flush;

    net.train(train_inputs, train_targets, val_inputs, val_targets);

    std::cout << "\nTest MSE: " << net.evaluate(test_inputs, test_targets) << std::endl << std::flush;
    std::cout << "Predictions on test set (first 10):\n" << std::flush;
    for (size_t i = 0; i < 10 && i < test_inputs.size(); ++i) {
        auto output = net.predict(test_inputs[i]);
        std::cout << "Input: [" << test_inputs[i][0] << ", " << test_inputs[i][1]
                  << "] -> Output: " << output[0]
                  << ", Target: " << test_targets[i][0] << "\n" << std::flush;
    }

    net.save_model(model_name);
}

// === Load saved model and test on classic XOR ===
void test_load_and_predict(const std::string& model_name) {
    std::vector<size_t> layers = {2, 8, 1};
    std::vector<network::ActivationType> activations = {
        network::ActivationType::ReLU,
        network::ActivationType::Sigmoid
    };

    network net(layers, activations, 0.001f, 5000, 64, 0.9f);

    net.load_model(model_name);

    std::vector<std::vector<float>> inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    std::cout << "\nPredictions from loaded model:\n" << std::flush;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1]
                  << "] -> Output: " << output[0]
                  << ", Target: " << targets[i][0] << "\n" << std::flush;
    }
}

int main() {
    std::cout << "Starting program execution..." << std::endl << std::flush;
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to detect CUDA devices: " << cudaGetErrorString(err) << std::endl << std::flush;
        return 1;
    }
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl << std::flush;
        return 1;
    }
    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl << std::flush;
    try {
        std::cout << "=== Without Batch Normalization ===\n" << std::flush;
        train_xor_example(false, "xor_model_no_bn.txt");
        test_load_and_predict("xor_model_no_bn.txt");

        std::cout << "\n=== With Batch Normalization ===\n" << std::flush;
        train_xor_example(true, "xor_model_bn.txt");
        test_load_and_predict("xor_model_bn.txt");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl << std::flush;
        return 1;
    }
    std::cout << "Program completed successfully." << std::endl << std::flush;
    return 0;
}