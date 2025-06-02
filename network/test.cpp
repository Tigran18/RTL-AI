#include "network.hpp"
#include <iostream>

void train_xor_example() {
    std::vector<size_t> layers = {2, 3, 1};
    // std::vector<network::ActivationType> activations = {
    //     network::ActivationType::ReLU,
    //     network::ActivationType::Sigmoid
    // };
    network net(layers, {1, 0}, 0.1, 10000, 2, 0.9);
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    std::cout << "Training XOR network...\n";
    net.train(inputs, targets);
    std::cout << "\nPredictions:\n";
    for (const auto& input : inputs) {
        auto output = net.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: " << output[0] << "\n";
    }
    net.save_model("xor_model.txt");
}

void test_load_and_predict() {
    std::vector<size_t> layers = {2, 3, 1};
    // std::vector<network::ActivationType> activations = {
    //     network::ActivationType::ReLU,
    //     network::ActivationType::Sigmoid
    // };
    network net(layers, {1, 0}, 0.1, 10000, 2, 0.9);
    net.load_model("xor_model.txt");
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::cout << "\nPredictions from loaded model:\n";
    for (const auto& input : inputs) {
        auto output = net.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: " << output[0] << "\n";
    }
}

int main() {
    try {
        test_load_and_predict();
        train_xor_example();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}