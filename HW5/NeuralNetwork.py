import numpy as np

# Input data: each row is a sample with 3 features
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0]
], dtype=float)

# Output data: each row is a 2-element target
y = np.array([
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
], dtype=float)

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network class
class NeuralNetwork:
    def __init__(self, X, y):
        self.input = X
        self.y = y
        self.output = np.zeros(y.shape)
        self.weights1 = np.random.rand(self.input.shape[1], 6)  # 3x6
        self.bias1 = np.zeros((1, 6))
        self.weights2 = np.random.rand(6, 2)  # 6x2
        self.bias2 = np.zeros((1, 2))

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output

    def backprop(self):
        error = self.y - self.output
        d_output = error * sigmoid_derivative(self.output)
        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.layer1)

        self.weights2 += self.layer1.T.dot(d_output) * 0.1
        self.bias2 += np.sum(d_output, axis=0, keepdims=True) * 0.1
        self.weights1 += self.input.T.dot(d_hidden) * 0.1
        self.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

    def train(self, iterations):
        for i in range(iterations):
            self.output = self.feedforward()
            self.backprop()
            if i % 300 == 0:
                print("for iteration #", i, "\n")
                print("Input: \n", self.input)
                print("Actual Output: \n", self.y)
                print("Predicted Output: \n", self.output)
                print("Loss: \n", np.mean(np.square(self.y - self.output)))
                print("\n")

NN = NeuralNetwork(X, y)
NN.train(1500)

# Q4: Dimensions of weight matrices
print("Q4: Dimensions")
print("Weight1:", NN.weights1.shape)  # (3, 6)
print("Weight2:", NN.weights2.shape)  # (6, 2)
print("\n")

# Q5: Predict on new input [1, 1, 1]
X1 = np.array([[1, 1, 1]])
layer1 = sigmoid(np.dot(X1, NN.weights1) + NN.bias1)
y_pred = sigmoid(np.dot(layer1, NN.weights2) + NN.bias2)

print("Q5: Predicted y for input [1, 1, 1]:")
print(y_pred)
