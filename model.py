import numpy as np

class BPNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.output = self.Z2  # regression output
        return self.output

    def backward(self, X, y):
        m = X.shape[0]
        error = self.output - y
        dW2 = (1/m) * np.dot(self.A1.T, error)
        db2 = (1/m) * np.sum(error, axis=0, keepdims=True)

        dA1 = np.dot(error, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Gradient descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return np.mean(error**2)

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y)
            losses.append(loss)
        return losses

    def predict(self, X):
        return self.forward(X)
