import numpy as np

## optimizers
class SGD:
    def name(self):
        return "sgd"

    def optimize_params(self, weights, biases, d_w_array, d_b_array, beta, beta1, beta2, learning_rate, weight_decay, epsilon_for_optimizer, nn, X, y):
        for index in range(len(weights)):
            weights[index] -= (learning_rate * d_w_array[index]  - (weight_decay * learning_rate * weights[index]))
            biases[index] -= (learning_rate * d_b_array[index]  - (weight_decay * learning_rate * biases[index]))
        return (weights, biases)

class Momentum:
    def __init__(self):
        self.param_update_weights = None
        self.param_update_biases = None

    def name(self):
        return "momentum"

    def optimize_params(self, weights, biases, d_w_array, d_b_array, beta, beta1, beta2, learning_rate, weight_decay, epsilon_for_optimizer, nn, X, y):
        if (self.param_update_weights == None):
            self.param_update_weights = d_w_array
            self.param_update_biases = d_b_array
        else:
            self.param_update_weights = [beta * self.param_update_weights[index] + d_w_array[index] for index in range(len(self.param_update_weights))]
            self.param_update_biases = [beta * self.param_update_biases[index] + d_b_array[index] for index in range(len(self.param_update_biases))]
        
        for index in range(len(weights)):
            weights[index] -= (learning_rate * self.param_update_weights[index] - (weight_decay * learning_rate * weights[index]))
            biases[index] -= (learning_rate * self.param_update_biases[index] - (weight_decay * learning_rate * biases[index]))

        return (weights, biases)
    
class Rmsprop:

    def __init__(self):
        self.param_update_weights = None
        self.param_update_biases = None

        self.learning_rate_update_weights = None
        self.learning_rate_update_biases = None

    def name(self):
        return "rmsprop"


    def optimize_params(self, weights, biases, d_w_array, d_b_array, beta, beta1, beta2, learning_rate, weight_decay, epsilon_for_optimizer, nn, X, y):
        self.param_update_weights = d_w_array
        self.param_update_biases = d_b_array
        if (self.learning_rate_update_weights == None):
            self.learning_rate_update_weights = [(1 - beta) * np.square(d_w_array[index]) for index in range(len(d_w_array))]
            self.learning_rate_update_biases = [(1 - beta) * np.square(d_b_array[index]) for index in range(len(d_b_array))]
        else:
            self.learning_rate_update_weights = [
                (beta * self.learning_rate_update_weights[index]) + 
                ((1 - beta) * np.square(d_w_array[index])) for index in range(len(d_w_array))
            ]
            self.learning_rate_update_biases = [
                (beta * self.learning_rate_update_biases[index]) + 
                ((1 - beta) * np.square(d_b_array[index])) for index in range(len(d_b_array))
            ]

        effective_learning_rate_weights = [learning_rate / np.sqrt(self.learning_rate_update_weights[index] + epsilon_for_optimizer) for index in range(len(self.learning_rate_update_weights))]
        effective_learning_rate_biases = [learning_rate / np.sqrt(self.learning_rate_update_biases[index] + epsilon_for_optimizer) for index in range(len(self.learning_rate_update_biases))]

        for index in range(len(weights)):
            
            weights[index] -= (effective_learning_rate_weights[index] * self.param_update_weights[index] - (weight_decay * effective_learning_rate_weights[index] * weights[index]))
            biases[index] -= (effective_learning_rate_biases[index] * self.param_update_biases[index] - (weight_decay * effective_learning_rate_biases[index] * biases[index]))

        return (weights, biases)

class Nag:
    def __init__(self):
        self.param_update_weights = None
        self.param_update_biases = None


    def name(self):
        return "nag"
    def optimize_params(self, weights, biases, d_w_array, d_b_array, beta, beta1, beta2, learning_rate, weight_decay, epsilon_for_optimizer, nn, X, y):
        if (self.param_update_weights == None):

            self.param_update_weights = d_w_array
            self.param_update_biases = d_b_array

        else:
            weight_grad_offsets = [beta * self.param_update_weights[index] for index in range(len(weights))]
            bias_grad_offsets = [beta * self.param_update_biases[index] for index in range(len(biases))]
            (d_w_array, d_b_array) = nn.backward(X, y, weight_offsets=weight_grad_offsets, bias_offsets=bias_grad_offsets, learning_rate=learning_rate)

            self.param_update_weights = [(beta * self.param_update_weights[index]) + d_w_array[index] for index in range(len(d_w_array))]
            self.param_update_biases = [(beta * self.param_update_biases[index]) + d_b_array[index] for index in range(len(d_b_array))]

        for index in range(len(weights)):
            
            weights[index] -= (learning_rate * self.param_update_weights[index] - (weight_decay * learning_rate * weights[index]))
            biases[index] -= (learning_rate * self.param_update_biases[index] - (weight_decay * learning_rate * biases[index]))
        return (weights, biases)
    
class Adam:
    def __init__(self):
        
        self.param_update_weights = None
        self.param_update_biases = None

        self.learning_rate_update_weights = None
        self.learning_rate_update_biases = None
        self.t = 1
    def name(self):
        return "adam"
    def optimize_params(self, weights, biases, d_w_array, d_b_array, beta, beta1, beta2, learning_rate, weight_decay, epsilon_for_optimizer, nn, X, y):
        if (self.param_update_weights == None):
            self.param_update_weights = [d_w_array[index] * (1 - beta1) for index in range(len(d_w_array))]
            self.param_update_biases = [d_b_array[index] * (1 - beta1) for index in range(len(d_b_array))]
        else:
            self.param_update_weights = [beta1 * self.param_update_weights[index] + (beta1) * d_w_array[index] for index in range(len(self.param_update_weights))]
            self.param_update_biases = [beta1 * self.param_update_biases[index] + (1 - beta1) * d_b_array[index] for index in range(len(self.param_update_biases))]

        if (self.learning_rate_update_weights == None):
            self.learning_rate_update_weights = [(1 - beta2) * np.square(d_w_array[index]) for index in range(len(d_w_array))]
            self.learning_rate_update_biases = [(1 - beta2) * np.square(d_b_array[index]) for index in range(len(d_b_array))]
        else:
            self.learning_rate_update_weights = [
                (beta2 * self.learning_rate_update_weights[index]) + 
                ((1 - beta2) * np.square(d_w_array[index])) for index in range(len(d_w_array))
            ]
            self.learning_rate_update_biases = [
                (beta2 * self.learning_rate_update_biases[index]) + 
                ((1 - beta2) * np.square(d_b_array[index])) for index in range(len(d_b_array))
            ]
        
        self.learning_rate_update_weights = [self.learning_rate_update_weights[index] / (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_weights))]
        self.learning_rate_update_biases = [self.learning_rate_update_biases[index] / (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_biases))]

        self.param_update_weights = [self.param_update_weights[index] / (1 - (beta1 ** self.t)) for index in range(len(self.param_update_weights))]
        self.param_update_biases = [self.param_update_biases[index] / (1 - (beta1 ** self.t)) for index in range(len(self.param_update_biases))]

        effective_learning_rate_weights = [learning_rate / np.sqrt(self.learning_rate_update_weights[index] + epsilon_for_optimizer) for index in range(len(self.learning_rate_update_weights))]
        effective_learning_rate_biases = [learning_rate / np.sqrt(self.learning_rate_update_biases[index] + epsilon_for_optimizer) for index in range(len(self.learning_rate_update_biases))]

        for index in range(len(weights)):
            weights[index] -= (effective_learning_rate_weights[index] * self.param_update_weights[index] - (weight_decay * effective_learning_rate_weights[index] * weights[index]))
            biases[index] -= (effective_learning_rate_biases[index] * self.param_update_biases[index] - (weight_decay * effective_learning_rate_biases[index] * biases[index]))

        # adjusting back the bias correction term so that we can have the actual term in the later iterations
        self.learning_rate_update_weights = [self.learning_rate_update_weights[index] * (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_weights))]
        self.learning_rate_update_biases = [self.learning_rate_update_biases[index] * (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_biases))]

        self.param_update_weights = [self.param_update_weights[index] * (1 - (beta1 ** self.t)) for index in range(len(self.param_update_weights))]
        self.param_update_biases = [self.param_update_biases[index] * (1 - (beta1 ** self.t)) for index in range(len(self.param_update_biases))]

        self.t += 1

        return (weights, biases)
    

class Nadam:
    def __init__(self):
        
        self.param_update_weights = None
        self.param_update_biases = None

        self.learning_rate_update_weights = None
        self.learning_rate_update_biases = None
        self.t = 1

    def name(self):
        return "nadam"

    def optimize_params(self, weights, biases, d_w_array, d_b_array, beta, beta1, beta2, learning_rate, weight_decay, epsilon_for_optimizer, nn, X, y):
        if (self.param_update_weights == None):
            self.param_update_weights = [d_w_array[index] * (1 - beta1) for index in range(len(d_w_array))]
            self.param_update_biases = [d_b_array[index] * (1 - beta1) for index in range(len(d_b_array))]
        else:
            self.param_update_weights = [beta1 * self.param_update_weights[index] + (beta1) * d_w_array[index] for index in range(len(self.param_update_weights))]
            self.param_update_biases = [beta1 * self.param_update_biases[index] + (1 - beta1) * d_b_array[index] for index in range(len(self.param_update_biases))]

        if (self.learning_rate_update_weights == None):
            self.learning_rate_update_weights = [(1 - beta2) * np.square(d_w_array[index]) for index in range(len(d_w_array))]
            self.learning_rate_update_biases = [(1 - beta2) * np.square(d_b_array[index]) for index in range(len(d_b_array))]
        else:
            self.learning_rate_update_weights = [
                (beta2 * self.learning_rate_update_weights[index]) + 
                ((1 - beta2) * np.square(d_w_array[index])) for index in range(len(d_w_array))
            ]
            self.learning_rate_update_biases = [
                (beta2 * self.learning_rate_update_biases[index]) + 
                ((1 - beta2) * np.square(d_b_array[index])) for index in range(len(d_b_array))
            ]
        
        self.learning_rate_update_weights = [self.learning_rate_update_weights[index] / (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_weights))]
        self.learning_rate_update_biases = [self.learning_rate_update_biases[index] / (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_biases))]

        self.param_update_weights = [self.param_update_weights[index] / (1 - (beta1 ** self.t)) for index in range(len(self.param_update_weights))]
        self.param_update_biases = [self.param_update_biases[index] / (1 - (beta1 ** self.t)) for index in range(len(self.param_update_biases))]

        effective_learning_rate_weights = [learning_rate / np.sqrt(self.learning_rate_update_weights[index] + epsilon_for_optimizer) for index in range(len(self.learning_rate_update_weights))]
        effective_learning_rate_biases = [learning_rate / np.sqrt(self.learning_rate_update_biases[index] + epsilon_for_optimizer) for index in range(len(self.learning_rate_update_biases))]

        for index in range(len(weights)):
            weights[index] -= (effective_learning_rate_weights[index] * ((beta1 * self.param_update_weights[index]) + ((1 - beta1) * d_w_array[index]) / (1 - (beta1 ** self.t))) - (weight_decay * effective_learning_rate_weights[index] * weights[index]))
            biases[index] -= (effective_learning_rate_biases[index] * ((beta1 * self.param_update_biases[index]) + ((1 - beta1) * d_b_array[index]) / (1 - (beta1 ** self.t))) - (weight_decay * effective_learning_rate_biases[index] * biases[index]))

        # adjusting back the bias correction term so that we can have the actual term in the later iterations
        self.learning_rate_update_weights = [self.learning_rate_update_weights[index] * (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_weights))]
        self.learning_rate_update_biases = [self.learning_rate_update_biases[index] * (1 - (beta2 ** self.t)) for index in range(len(self.learning_rate_update_biases))]

        self.param_update_weights = [self.param_update_weights[index] * (1 - (beta1 ** self.t)) for index in range(len(self.param_update_weights))]
        self.param_update_biases = [self.param_update_biases[index] * (1 - (beta1 ** self.t)) for index in range(len(self.param_update_biases))]

        self.t += 1

        return (weights, biases)
        