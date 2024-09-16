# Custom_machine_learning_model_from_scratch
# Implement a simple neural network or decision tree from scratch using only NumPy, without relying on machine learning libraries like TensorFlow or PyTorch.
# implementing a simple neural network

import numpy as np

# Activation functions and their derivatives


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Loss function: Binary Cross-Entropy Loss


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Neural Network Class


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(
            input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(
            hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(
            X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = relu(self.hidden_layer_input)
        self.output_layer_input = np.dot(
            self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        return self.output

    def backward(self, X, y):
        # Backpropagation
        output_error = self.output - y
        d_weights_hidden_output = np.dot(
            self.hidden_layer_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * \
            relu_derivative(self.hidden_layer_input)
        d_weights_input_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward and backward pass
            output = self.forward(X)
            self.backward(X, y)

            # Calculate loss
            loss = binary_cross_entropy(y, output)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.where(output > 0.5, 1, 0)


# Example usage
if __name__ == "__main__":
    # XOR problem (example dataset)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create a neural network instance
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4,
                             output_size=1, learning_rate=0.1)

    # Train the neural network
    nn.train(X, y, epochs=1000)

    # Test predictions
    predictions = nn.predict(X)
    print("\nPredictions:")
    print(predictions)
