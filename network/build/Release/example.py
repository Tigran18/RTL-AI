import network_lib
import numpy as np

layer_sizes = [2, 4, 1]
activations = [network_lib.ActivationType.ReLU, network_lib.ActivationType.Sigmoid]
net = network_lib.Network(layer_sizes, activations)
print("Network initialized successfully")