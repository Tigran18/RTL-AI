Network
A Python library for a neural network with CUDA support, built using pybind11.
Installation
pip install .

Requirements

Python 3.6+
NumPy (>=1.18.0)
pybind11 (>=2.6.0)
CUDA Toolkit (for GPU support)
NVIDIA GPU with cuBLAS

Usage
import numpy as np
from network import Network, ActivationType

# Define network architecture
layer_sizes = [2, 4, 1]  # Input: 2, Hidden: 4, Output: 1
activations = [ActivationType.ReLU, ActivationType.Sigmoid]
net = Network(layer_sizes, activations, learning_rate=0.01, batch_size=32)

# Generate dummy data
inputs = np.random.rand(100, 2).astype(np.float32)
targets = np.random.rand(100, 1).astype(np.float32)

# Train the network
net.train(inputs, targets)

# Make a prediction
input_sample = np.array([0.5, 0.5], dtype=np.float32)
output = net.predict(input_sample)
print("Prediction:", output)

# Evaluate the network
mse = net.evaluate(inputs, targets)
print("MSE:", mse)

# Save and load model
net.save_model("model.txt")
net.load_model("model.txt")

Building from Source

Ensure CUDA Toolkit and cuBLAS are installed.
Install dependencies: pip install numpy pybind11
Run: pip install .
