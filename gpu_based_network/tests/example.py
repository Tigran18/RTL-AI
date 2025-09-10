import py_network
import numpy as np

# -----------------------------
# Helper functions
# -----------------------------
def generate_xor_data(n_samples=1000):
    """Generate XOR dataset with noise."""
    X = np.random.randint(0, 2, (n_samples, 2)).astype(np.float32)
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.float32).reshape(-1, 1)
    return X.tolist(), y.tolist()


def print_predictions(net, inputs):
    """Pretty print predictions for given inputs."""
    print("\nPredictions:")
    for x in inputs:
        pred = net.predict(x)
        print(f"Input: {x} -> Predicted: {pred}")


# -----------------------------
# Main Training Script
# -----------------------------
def main():
    print("CUDA Available:", py_network.cuda_available())

    # Generate XOR dataset
    X, y = generate_xor_data(1000)
    print(f"Generated {len(X)} XOR samples.")

    # Split into train, validation, and test sets
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Define network
    net = py_network.Network(
        [2, 8, 1],  # Layers: 2 → 8 → 1
        [py_network.ActivationType.ReLU, py_network.ActivationType.Sigmoid],
        learning_rate=0.01,
        epochs=200,
        batch_size=32
    )

    # Train
    print("\nTraining network...")
    net.train(X_train, y_train, X_val, y_val)

    # Evaluate
    train_loss = net.evaluate(X_train, y_train)
    val_loss = net.evaluate(X_val, y_val)
    test_loss = net.evaluate(X_test, y_test)

    print(f"\nFinal Losses:")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")
    print(f"Test Loss:  {test_loss:.6f}")

    # Predictions
    print_predictions(net, [[0, 0], [0, 1], [1, 0], [1, 1]])


if __name__ == "__main__":
    main()
