#include "network.hpp"
#include <iostream>
#include <vector>
#include <random>

int main() {
    // Define network architecture: input(2) -> hidden(2) -> output(1)
    std::vector<size_t> layers = {2, 2, 1};
    std::vector<network::ActivationType> activations = {
        network::ActivationType::Sigmoid,
        network::ActivationType::Sigmoid
    };

    // Create network
    network net(layers, activations, 0.01f, 5000, 4, 0.9f, false);

    // XOR training data
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> targets = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    // Validation data (same as training for simple XOR)
    std::vector<std::vector<float>> val_inputs = inputs;
    std::vector<std::vector<float>> val_targets = targets;

    // Train the network
    net.train(inputs, targets, val_inputs, val_targets);

    // Evaluate on validation
    std::cout << "Validation MSE: " << net.evaluate(val_inputs, val_targets) << std::endl;

    // Test predictions
    std::cout << "Predictions:" << std::endl;
    for (const auto& input : inputs) {
        auto output = net.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: " << output[0] << std::endl;
    }

    // Save model
    net.save_model("xor_model.net");

    return 0;
}