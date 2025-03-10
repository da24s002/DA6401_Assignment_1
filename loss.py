import numpy as np    
class CrossEntropyLoss:
    def loss(self, output, target):
        # Cross-entropy loss function
        m = target.shape[0]
        return -np.sum(target * np.log(output + 1e-8)) / m
    
    def derivative(self, output, target):
        return output - target
    
class SquaredErrorLoss:
    def loss(self, output, target):
        # Squared error loss function
        m = target.shape[0]
        return np.sum(np.square(output - target)) / (2 * m)
    
    def derivative(self, output, target):
        m = target.shape[0]
        return (output - target)/m