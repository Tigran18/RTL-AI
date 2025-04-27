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

    network net(3, {2, 3, 1}, 0.9, 4000); 

    net.train(inputs, targets);

    std::cout << "\nTesting network after training:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        net.forward_propagate(inputs[i]);
        std::cout << "Input (" << inputs[i][0] << ", " << inputs[i][1] << ") -> ";
        net.display_outputs();
    }

    return 0;
}
