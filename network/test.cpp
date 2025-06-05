#include "network.hpp"
#include <iostream>
#include <random>

void generate_xor_data(std::vector<std::vector<double>>& inputs,
                      std::vector<std::vector<double>>& targets,
                      size_t num_samples, double noise_stddev) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> noise(0.0, noise_stddev);
    std::uniform_int_distribution<int> binary(0, 1);
    inputs.clear();
    targets.clear();
    for (size_t i = 0; i < num_samples; ++i) {
        double x1 = binary(gen);
        double x2 = binary(gen);
        double y = (x1 != x2) ? 1.0 : 0.0;
        inputs.push_back({x1 + noise(gen), x2 + noise(gen)});
        targets.push_back({y});
    }
}

void split_data(const std::vector<std::vector<double>>& inputs,
                const std::vector<std::vector<double>>& targets,
                std::vector<std::vector<double>>& train_inputs,
                std::vector<std::vector<double>>& train_targets,
                std::vector<std::vector<double>>& val_inputs,
                std::vector<std::vector<double>>& val_targets,
                std::vector<std::vector<double>>& test_inputs,
                std::vector<std::vector<double>>& test_targets,
                double train_ratio = 0.6, double val_ratio = 0.2) {
    size_t n = inputs.size();
    size_t train_size = static_cast<size_t>(n * train_ratio);
    size_t val_size = static_cast<size_t>(n * val_ratio);
    size_t test_size = n - train_size - val_size;
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
}

void train_xor_example() {
    std::vector<std::vector<double>> inputs, targets;
    generate_xor_data(inputs, targets, 1000, 0.1);
    std::vector<std::vector<double>> train_inputs, train_targets;
    std::vector<std::vector<double>> val_inputs, val_targets;
    std::vector<std::vector<double>> test_inputs, test_targets;
    split_data(inputs, targets, train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets);
    std::vector<size_t> layers = {2, 3, 1};
    std::vector<network::ActivationType> activations = {
        network::ActivationType::ReLU,
        network::ActivationType::Sigmoid
    };
    network net(layers, activations, 0.1, 10000, 32, 0.9);
    std::cout << "Training XOR network...\n";
    net.train(train_inputs, train_targets, val_inputs, val_targets);
    std::cout << "\nTest MSE: " << net.evaluate(test_inputs, test_targets) << "\nPredictions on test set:\n";
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        auto output = net.predict(test_inputs[i]);
        std::cout << "Input: [" << test_inputs[i][0] << ", " << test_inputs[i][1]
                  << "] -> Output: " << output[0] << ", Target: " << test_targets[i][0] << "\n";
    }
    net.save_model("xor_model.txt");
}

void test_load_and_predict() {
    std::vector<size_t> layers = {2, 3, 1};
    std::vector<network::ActivationType> activations = {
        network::ActivationType::ReLU,
        network::ActivationType::Sigmoid
    };
    network net(layers, activations, 0.1, 10000, 32, 0.9);
    net.load_model("xor_model.txt");
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    std::cout << "\nPredictions from loaded model on original XOR inputs:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1]
                  << "] -> Output: " << output[0] << ", Target: " << targets[i][0] << "\n";
    }
}

int main() {
    try {
        train_xor_example();
        test_load_and_predict();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}