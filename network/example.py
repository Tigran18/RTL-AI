# example.py
import network_lib
import numpy as np

# Create a simple dataset (XOR problem)
inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
targets = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

# Define network architecture: 2 input neurons, 4 hidden neurons, 1 output neuron
layer_sizes = [2, 4, 1]
activations = [network_lib.ActivationType.ReLU, network_lib.ActivationType.Sigmoid]

# Initialize the network
net = network_lib.Network(
    layer_sizes=layer_sizes,
    activations=activations,
    learning_rate=0.01,
    epochs=1000,
    batch_size=4,
    momentum=0.9,
    use_batch_norm=False
)

# Train the network
net.train(inputs, targets)

# Evaluate the network
mse = net.evaluate(inputs, targets)
print(f"Final MSE: {mse}")

# Make predictions
for i, input in enumerate(inputs):
    prediction = net.predict(input)
    print(f"Input: {input}, Predicted: {prediction}, Target: {targets[i]}")

# Save the model
net.save_model("xor_model.txt")

# Load the model and verify
net2 = network_lib.Network(
    layer_sizes=layer_sizes,
    activations=activations
)
net2.load_model("xor_model.txt")
mse2 = net2.evaluate(inputs, targets)
print(f"Loaded model MSE: {mse2}")

# Display layer sizes
print(f"Layer sizes: {net.get_layer_sizes()}")