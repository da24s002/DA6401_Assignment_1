import numpy as np

## activation functions
class Sigmoid:
    def activation(self, x):
        # numerically stable sigmoid implementation
        x_ = np.asarray(x, dtype=np.float64)
        mask = x_ < 0
        
        result = np.empty_like(x_)
        
        # For negative values: exp(x) / (1 + exp(x))
        exp_x = np.exp(x_[mask])
        result[mask] = exp_x / (1.0 + exp_x)
        
        # For non-negative values: 1 / (1 + exp(-x))
        minus_exp_x = np.exp(-x_[~mask])
        result[~mask] = 1.0 / (1.0 + minus_exp_x)
        
        return result
       
    # def activation(self, x):
    #     # Sigmoid activation function
    #     return 1 / (1 + np.exp(-x))

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