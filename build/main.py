from ai_module import Network, JSON

# Load training data
json_data = JSON("training_data.json", True)  # assuming raw string
json_data.print(0)

# Create network
net = Network([3, 5, 1], 0.1, 20000)

# Example: train and predict
training_inputs = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
targets = [[0.4], [0.6]]
net.train(training_inputs, targets)

print(net.predict([0.2, 0.1, 0.3]))
print(net.predict([0.9, 0.8, 0.7]))
