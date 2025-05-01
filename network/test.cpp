#include "network.hpp"

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
        {1}
    };

    network net({2, 15, 20, 1}, 0.1, 20000); 

    net.train(inputs, targets);

    std::cout << "\nTesting network after training:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = net.predict(inputs[i]);
        std::cout << "Input (" << inputs[i][0] << ", " << inputs[i][1] << ") -> " << output[0] << std::endl;
        net.display_outputs();
    }

    return 0;
}
