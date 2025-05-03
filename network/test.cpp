#include "network.hpp"
#include <iostream>
#include <filesystem>

int main() {
    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    network net({2, 15, 20, 1}, 0.1, 20000);

    const std::string model_file = "xor_model.txt";

    if (std::filesystem::exists(model_file)) {
        std::cout << "Loading saved model from file...\n";
        net.load_model(model_file);
    } else {
        std::cout << "Training model...\n";
        net.train(inputs, targets);
        net.save_model(model_file);
        std::cout << "Model saved to file.\n";
    }

    std::cout << "\nTesting network:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.predict(inputs[i]);
        std::cout << "Input (" << inputs[i][0] << ", " << inputs[i][1] << ") -> " << output[0] << std::endl;
    }

    return 0;
}
