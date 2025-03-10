import numpy as np

## activation functions
class Sigmoid:     
    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        # Derivative of sigmoid function
        return self.activation(x) * (1 - self.activation(x))
class Relu:
    def activation(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def derivative(self, x):
        # Derivative of ReLU function
        return np.where(x > 0, 1, 0)
class Tanh:
    def activation(self, x):
        # Tanh activation function
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def derivative(self, x):
        return 1 - np.square((self.activation(x)))