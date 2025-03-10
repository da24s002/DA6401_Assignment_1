import numpy as np
import yaml
import wandb
import sys

wandb.login()

from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

import argparse

import os
# os.environ["WANDB_MODE"] = "offline"
os.environ['WANDB_TIMEOUT'] = '60'




np.random.seed(0)

class VanillaNeuralNetwork:
    def __init__(self, 
                 input_size, 
                 hidden_layer_sizes, 
                 output_size,
                 activation_function,
                 output_function,
                 loss_function,
                 initializer,
                 optimizer,
                 beta_optimizer=0.5,
                 beta1_optimizer=0.9,
                 beta2_optimizer=0.999,
                 epsilon_for_optimizer=0.000001,
                 weight_decay=0.01):
        
        # Initialize the architecture and parameters
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        self.activation_function = activation_function
        self.output_function = output_function
        self.loss_function = loss_function

        self.initializer = initializer
        self.optimizer = optimizer

        self.beta_optimizer = beta_optimizer
        self.beta1_optimizer = beta1_optimizer
        self.beta2_optimizer = beta2_optimizer

        self.epsilon_for_optimizer = epsilon_for_optimizer

        self.weight_decay = weight_decay

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input -> Hidden Layer 1
        self.weights.append(self.initializer(input_size, hidden_layer_sizes[0]))
        # self.biases.append(np.zeros((1, hidden_layer_sizes[0])))
        self.biases.append(np.random.randn(1, hidden_layer_sizes[0])* 0.1)

        for layer_index in range(1, len(hidden_layer_sizes)):
            self.weights.append(self.initializer(hidden_layer_sizes[layer_index - 1], hidden_layer_sizes[layer_index]))
            self.biases.append(np.random.randn(1, hidden_layer_sizes[layer_index]) * 0.1)

        # Output -> last layer
        self.weights.append(self.initializer(hidden_layer_sizes[-1], output_size))
        self.biases.append(np.random.randn(1, output_size) * 0.1)
    
    def forward(self, X):
        # Perform a forward pass through the network
        self.z_array = [] #preactivation array
        self.a_array = [] #activation array

        self.z_array.append(np.dot(X, self.weights[0]) + self.biases[0])
        # self.a_array.append(self.relu(self.z_array[-1]))
        # self.a_array.append(self.tanh(self.z_array[-1]))
        self.a_array.append(self.activation_function.activation(self.z_array[-1]))

        for layer_index in range(1,len(self.hidden_layer_sizes)):
            self.z_array.append(np.dot(self.a_array[-1], self.weights[layer_index]) + self.biases[layer_index])
            # self.a_array.append(self.relu(self.z_array[-1]))
            # self.a_array.append(self.tanh(self.z_array[-1]))
            self.a_array.append(self.activation_function.activation(self.z_array[-1]))

        self.z_array.append(np.dot(self.a_array[-1], self.weights[-1]) + self.biases[-1])
        # self.a_array.append(self.softmax(self.z_array[-1]))
        self.a_array.append(self.output_function(self.z_array[-1]))

        return self.a_array[-1]

        
    
    def backward(self, X, y, weight_offsets, bias_offsets, learning_rate=0.01):
        # Perform backward pass (backpropagation)
        m = X.shape[0]

        weights_for_grad = [self.weights[index] - weight_offsets[index] for index in range(len(weight_offsets))]
        biases_for_grad = [self.biases[index] - bias_offsets[index] for index in range(len(bias_offsets))]


        
        
        d_w_array = [None for index in range(len(weights_for_grad))]
        d_b_array = [None for index in range(len(biases_for_grad))]
        d_z = self.loss_function.derivative(self.a_array[-1], y)

        for layer_index in range(len(weights_for_grad) - 1, 0, -1):
            d_w = np.dot(self.a_array[layer_index - 1].T, d_z) / m
            d_b = np.sum(d_z, axis=0, keepdims=True) / m

            d_w_array[layer_index] = d_w
            d_b_array[layer_index] = d_b

            
            d_a = np.dot(d_z, weights_for_grad[layer_index].T)
            # d_z2 = d_a2 * self.sigmoid_derivative(self.a2)
            # d_z = d_a * self.relu_derivative(self.a_array[layer_index - 1])
            # d_z = d_a * self.tanh_derivative(self.a_array[layer_index - 1])
            d_z = d_a * self.activation_function.derivative(self.a_array[layer_index - 1])

        d_w = np.dot(X.T, d_z) / m
        d_b = np.sum(d_z, axis=0, keepdims=True) / m

        d_w_array[0] = d_w
        d_b_array[0] = d_b

        return (d_w_array, d_b_array)
        

        # for index in range(len(self.weights)):
        #     self.weights[index] -= learning_rate * d_w_array[index]
        #     self.biases[index] -= learning_rate * d_b_array[index]

        
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, learning_rate=0.01):
        # Training loop
        N = len(X_train)
        
        no_of_batches = int(N/batch_size)
        if (N % batch_size != 0):
            no_of_batches += 1
        
        # weight_gradient_history_list = []
        # bias_gradient_history_list = []

        param_update_weights = None
        param_update_biases = None

        learning_rate_update_weights = None
        learning_rate_update_biases = None
        t = 1


        for epoch in range(epochs):
            # Forward pass
            for batch in range(no_of_batches):
                X_train_batch = X_train[batch * batch_size : (batch + 1) * batch_size]
                y_train_batch = y_train[batch * batch_size : (batch + 1) * batch_size]

                output = self.forward(X_train_batch)
                weight_grad_offsets = [0 for index in range(len(self.weights))]
                bias_grad_offsets = [0 for index in range(len(self.biases))]
                # Backward pass
                if (self.optimizer != 'nag'):
                    (d_w_array, d_b_array) = self.backward(X_train_batch, y_train_batch, weight_offsets=weight_grad_offsets, bias_offsets=bias_grad_offsets, learning_rate=learning_rate)

                # weight_gradient_history_list.append(d_w_array)
                # bias_gradient_history_list.append(d_b_array)


                ## optimizers

                if (self.optimizer == 'sgd'):
                    param_update_weights = d_w_array
                    param_update_biases = d_b_array
                    effective_learning_rate_weights = [learning_rate for index in range(len(param_update_weights))]
                    effective_learning_rate_biases = [learning_rate for index in range(len(param_update_biases))]

                elif (self.optimizer == 'momentum'):
                    if (param_update_weights == None):
                        param_update_weights = d_w_array
                        param_update_biases = d_b_array
                    else:
                        param_update_weights = [self.beta_optimizer * param_update_weights[index] + d_w_array[index] for index in range(len(param_update_weights))]
                        param_update_biases = [self.beta_optimizer * param_update_biases[index] + d_b_array[index] for index in range(len(param_update_biases))]

                    effective_learning_rate_weights = [learning_rate for index in range(len(param_update_weights))]
                    effective_learning_rate_biases = [learning_rate for index in range(len(param_update_biases))]

                elif (self.optimizer == 'rmsprop'):
                    param_update_weights = d_w_array
                    param_update_biases = d_b_array

                    if (learning_rate_update_weights == None):
                        learning_rate_update_weights = [(1 - self.beta_optimizer) * np.square(d_w_array[index]) for index in range(len(d_w_array))]
                        learning_rate_update_biases = [(1 - self.beta_optimizer) * np.square(d_b_array[index]) for index in range(len(d_b_array))]
                    else:
                        learning_rate_update_weights = [
                            (self.beta_optimizer * learning_rate_update_weights[index]) + 
                            ((1 - self.beta_optimizer) * np.square(d_w_array[index])) for index in range(len(d_w_array))
                        ]
                        learning_rate_update_biases = [
                            (self.beta_optimizer * learning_rate_update_biases[index]) + 
                            ((1 - self.beta_optimizer) * np.square(d_b_array[index])) for index in range(len(d_b_array))
                        ]

                    effective_learning_rate_weights = [learning_rate / np.sqrt(learning_rate_update_weights[index] + self.epsilon_for_optimizer) for index in range(len(learning_rate_update_weights))]
                    effective_learning_rate_biases = [learning_rate / np.sqrt(learning_rate_update_biases[index] + self.epsilon_for_optimizer) for index in range(len(learning_rate_update_biases))]

                elif (self.optimizer == 'nag'):

                    ## TODO: debug (Tanh only)
                    if (param_update_weights == None):
                        weight_grad_offsets = [0 for index in range(len(self.weights))]
                        bias_grad_offsets = [0 for index in range(len(self.biases))]
                        (d_w_array, d_b_array) = self.backward(X_train_batch, y_train_batch, weight_offsets=weight_grad_offsets, bias_offsets=bias_grad_offsets, learning_rate=learning_rate)

                        param_update_weights = d_w_array
                        param_update_biases = d_b_array
                        effective_learning_rate_weights = [learning_rate for index in range(len(param_update_weights))]
                        effective_learning_rate_biases = [learning_rate for index in range(len(param_update_biases))]

                    else:
                        weight_grad_offsets = [self.beta_optimizer*param_update_weights[index] for index in range(len(self.weights))]
                        bias_grad_offsets = [self.beta_optimizer*param_update_biases[index] for index in range(len(self.biases))]
                        (d_w_array, d_b_array) = self.backward(X_train_batch, y_train_batch, weight_offsets=weight_grad_offsets, bias_offsets=bias_grad_offsets, learning_rate=learning_rate)

                        param_update_weights = [(self.beta_optimizer * param_update_weights[index]) + d_w_array[index] for index in range(len(d_w_array))]
                        param_update_biases = [(self.beta_optimizer * param_update_biases[index]) + d_b_array[index] for index in range(len(d_b_array))]
                        effective_learning_rate_weights = [learning_rate for index in range(len(param_update_weights))]
                        effective_learning_rate_biases = [learning_rate for index in range(len(param_update_biases))]


                elif (self.optimizer == 'adam' or self.optimizer == 'nadam'):
                    if (param_update_weights == None):
                        param_update_weights = [d_w_array[index] * (1 - self.beta1_optimizer) for index in range(len(d_w_array))]
                        param_update_biases = [d_b_array[index] * (1 - self.beta1_optimizer) for index in range(len(d_b_array))]
                    else:
                        param_update_weights = [self.beta1_optimizer * param_update_weights[index] + (1 - self.beta1_optimizer) * d_w_array[index] for index in range(len(param_update_weights))]
                        param_update_biases = [self.beta1_optimizer * param_update_biases[index] + (1 - self.beta1_optimizer) * d_b_array[index] for index in range(len(param_update_biases))]

                    if (learning_rate_update_weights == None):
                        learning_rate_update_weights = [(1 - self.beta2_optimizer) * np.square(d_w_array[index]) for index in range(len(d_w_array))]
                        learning_rate_update_biases = [(1 - self.beta2_optimizer) * np.square(d_b_array[index]) for index in range(len(d_b_array))]
                    else:
                        learning_rate_update_weights = [
                            (self.beta2_optimizer * learning_rate_update_weights[index]) + 
                            ((1 - self.beta2_optimizer) * np.square(d_w_array[index])) for index in range(len(d_w_array))
                        ]
                        learning_rate_update_biases = [
                            (self.beta2_optimizer * learning_rate_update_biases[index]) + 
                            ((1 - self.beta2_optimizer) * np.square(d_b_array[index])) for index in range(len(d_b_array))
                        ]
                    
                    learning_rate_update_weights = [learning_rate_update_weights[index] / (1 - (self.beta2_optimizer ** t)) for index in range(len(learning_rate_update_weights))]
                    learning_rate_update_biases = [learning_rate_update_biases[index] / (1 - (self.beta2_optimizer ** t)) for index in range(len(learning_rate_update_biases))]

                    param_update_weights = [param_update_weights[index] / (1 - (self.beta1_optimizer ** t)) for index in range(len(param_update_weights))]
                    param_update_biases = [param_update_biases[index] / (1 - (self.beta1_optimizer ** t)) for index in range(len(param_update_biases))]

                    effective_learning_rate_weights = [learning_rate / np.sqrt(learning_rate_update_weights[index] + self.epsilon_for_optimizer) for index in range(len(learning_rate_update_weights))]
                    effective_learning_rate_biases = [learning_rate / np.sqrt(learning_rate_update_biases[index] + self.epsilon_for_optimizer) for index in range(len(learning_rate_update_biases))]

                # effective_learning_rate, param_update_weights, param_update_biases = self.optimizer(weight_gradient_history_list, bias_gradient_history_list, self.beta_optimizer, learning_rate)

                # update params
                for index in range(len(self.weights)):
                    if (self.optimizer == 'nadam'):
                        self.weights[index] -= (effective_learning_rate_weights[index] * ((self.beta1_optimizer * param_update_weights[index]) + ((1 - self.beta1_optimizer) * d_w_array[index]) / (1 - (self.beta1_optimizer ** t))) - (self.weight_decay * effective_learning_rate_weights[index] * self.weights[index]))
                        self.biases[index] -= (effective_learning_rate_biases[index] * ((self.beta1_optimizer * param_update_biases[index]) + ((1 - self.beta1_optimizer) * d_b_array[index]) / (1 - (self.beta1_optimizer ** t))) - (self.weight_decay * effective_learning_rate_biases[index] * self.biases[index]))
                    else:
                        self.weights[index] -= (effective_learning_rate_weights[index] * param_update_weights[index] - (self.weight_decay * effective_learning_rate_weights[index] * self.weights[index]))
                        self.biases[index] -= (effective_learning_rate_biases[index] * param_update_biases[index] - (self.weight_decay * effective_learning_rate_biases[index] * self.biases[index]))

                        

                if (self.optimizer == 'adam' or self.optimizer == 'nadam'): # adjusting back the bias correction term so that we can have the actual term in the later iterations
                    learning_rate_update_weights = [learning_rate_update_weights[index] * (1 - (self.beta2_optimizer ** t)) for index in range(len(learning_rate_update_weights))]
                    learning_rate_update_biases = [learning_rate_update_biases[index] * (1 - (self.beta2_optimizer ** t)) for index in range(len(learning_rate_update_biases))]

                    param_update_weights = [param_update_weights[index] * (1 - (self.beta1_optimizer ** t)) for index in range(len(param_update_weights))]
                    param_update_biases = [param_update_biases[index] * (1 - (self.beta1_optimizer ** t)) for index in range(len(param_update_biases))]

                    t += 1


            # Compute loss
            output = self.forward(X_train)
            train_loss = self.loss_function.loss(output, y_train)

            validation_output = self.forward(X_val)
            validation_loss = self.loss_function.loss(validation_output, y_val)

            train_accuracy = self.accuracy(X_train, y_train)
            validation_accuracy = self.accuracy(X_val, y_val)


            # wandb.log(
            #     {
            #         "epoch": epoch,
            #         "train_accuracy": train_accuracy,
            #         "train_loss": train_loss,
            #         "validation_accuracy": validation_accuracy,
            #         "validation_loss": validation_loss,
            #     }
            # )

            print({
                    "epoch": epoch,
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                })
        
    
    def predict(self, X):
        # Predict class labels for input data X
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def accuracy(self, X, y):
        # Calculate accuracy of the model
        predictions = self.predict(X)
        return np.mean(predictions == np.argmax(y, axis=1))
    
    def compress_output(self, output):
        return np.array([list(output[index]).index(max(output[index])) for index in range(len(output))])


def preprocess_data(X):
    # Normalize the input data (scale to range [0, 1])
    X = X / 255.0

    return X

def xavier_init(input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))

def random_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.1


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
        


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(output, target):
    # Cross-entropy loss function
    m = target.shape[0]
    return -np.sum(target * np.log(output + 1e-8)) / m

def squared_error_loss(output, target):
    # Squared error loss function
    m = target.shape[0]
    return np.sum(np.square(output - target)) / (2 * m)

def sgd(weight_history_list, bias_history_list, beta_optimizer, learning_rate):
    return (learning_rate, weight_history_list[-1], bias_history_list[-1])

activation_dict = {
    "sigmoid": Sigmoid(),
    "relu": Relu(),
    "tanh": Tanh(),
}

weight_initialization_dict = {
    "random": random_init,
    "xavier": xavier_init,
}

loss_function_dict = {
    "cross_entropy": CrossEntropyLoss(),
    "mean_squared_error": SquaredErrorLoss()
}


## data loading and segmentation into train, validation and test dataset
(x_train_fashion_mnist, y_train_fashion_mnist), (x_test_fashion_mnist, y_test_fashion_mnist) = fashion_mnist.load_data()
x_train_fashion_mnist, x_val_fashion_mnist, y_train_fashion_mnist, y_val_fashion_mnist = train_test_split(x_train_fashion_mnist, y_train_fashion_mnist, train_size=0.9, shuffle=True, random_state=42)

x_train_new_fashion_mnist = x_train_fashion_mnist.reshape(x_train_fashion_mnist.shape[0], x_train_fashion_mnist.shape[1]*x_train_fashion_mnist.shape[2])
y_train_new_fashion_mnist = np.array([[0 if index != value else 1 for index in range(10)] for value in y_train_fashion_mnist])
x_train_new_fashion_mnist = preprocess_data(x_train_new_fashion_mnist)

x_val_new_fashion_mnist = x_val_fashion_mnist.reshape(x_val_fashion_mnist.shape[0], x_val_fashion_mnist.shape[1]*x_val_fashion_mnist.shape[2])
y_val_new_fashion_mnist = np.array([[0 if index != value else 1 for index in range(10)] for value in y_val_fashion_mnist])
x_val_new_fashion_mnist = preprocess_data(x_val_new_fashion_mnist)

x_test_new_fashion_mnist = x_test_fashion_mnist.reshape(x_test_fashion_mnist.shape[0], x_test_fashion_mnist.shape[1]*x_test_fashion_mnist.shape[2])
y_test_new_fashion_mnist = np.array([[0 if index != value else 1 for index in range(10)] for value in y_test_fashion_mnist])
x_test_new_fashion_mnist = preprocess_data(x_test_new_fashion_mnist)

x_train_dict = {
    "mnist": None,
    "fashion_mnist": x_train_new_fashion_mnist
}
y_train_dict = {
    "mnist": None,
    "fashion_mnist": y_train_new_fashion_mnist
}

x_val_dict = {
    "mnist": None,
    "fashion_mnist": x_val_new_fashion_mnist
}
y_val_dict = {
    "mnist": None,
    "fashion_mnist": y_val_new_fashion_mnist
}

x_test_dict = {
    "mnist": None,
    "fashion_mnist": x_test_new_fashion_mnist
}
y_test_dict = {
    "mnist": None,
    "fashion_mnist": y_test_new_fashion_mnist
}


wandb.init(entity="da24s002-indian-institute-of-technology-madras", project="DA6401_Assignment_1")

def main(args):


    hidden_layer_size = args.hidden_size
    hidden_layers = args.num_layers
    hidden_layer_sizes = [hidden_layer_size] * hidden_layers
    activation_function = activation_dict[args.activation]
    weight_decay = args.weight_decay
    optimizer = args.optimizer
    initializer = weight_initialization_dict[args.weight_init]
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    if (optimizer == "momentum"):
        beta_optimizer = args.momentum
    else:
        beta_optimizer = args.beta
    beta1_optimizer = args.beta1
    beta2_optimizer = args.beta2
    epsilon_for_optimizer = args.epsilon

    loss_function = loss_function_dict[args.loss]
    x_train_new = x_train_dict[args.dataset]
    y_train_new = y_train_dict[args.dataset]

    x_val_new = x_val_dict[args.dataset]
    y_val_new = y_val_dict[args.dataset]

    x_test_new = x_test_dict[args.dataset]
    y_test_new = y_test_dict[args.dataset]

    wandb.run.name = f"hls_{hidden_layer_size}_hl_{hidden_layers}_bs_{batch_size}_opt_{optimizer}_act_{args.activation}_id_test_data_{wandb.run.id}"

    


    # Initialize the neural network with 784 input, 2 hidden layers (128, 64), and 10 output
    nn = VanillaNeuralNetwork(input_size=784, 
                              hidden_layer_sizes=hidden_layer_sizes, 
                              output_size=10, 
                              activation_function=activation_function, 
                              output_function=softmax,
                              loss_function=loss_function,
                              initializer=initializer,
                              optimizer=optimizer,
                              beta_optimizer=beta_optimizer,
                              beta1_optimizer=beta1_optimizer,
                              beta2_optimizer=beta2_optimizer,
                              epsilon_for_optimizer=epsilon_for_optimizer,
                              weight_decay=weight_decay)
    
    # # Train the model
    
    nn.train(x_train_new, y_train_new, x_val_new, y_val_new, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    test_output = nn.forward(x_test_new)
    # test_loss = loss_function.loss(test_output, y_test_new)
    test_output_compressed = nn.compress_output(test_output)
    y_test_compressed = nn.compress_output(y_test_new)
    test_accuracy = nn.accuracy(x_test_new, y_test_new)
    print(test_accuracy)
    labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=y_test_compressed, preds=test_output_compressed, class_names=labels)})
    


# main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("-nhl", "--hidden_layers", type=int, default=4, help="Number of hidden layers in the model.")
    # parser.add_argument("-sz", "--hidden_layer_size", type=int, default=256, help="Number of neurons in each of the hidden layers.")
    # parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["sigmoid", "tanh", "relu"],help="Activation function to use.")
    # parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay.")
    # parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help="Optimizer for update rule.")
    # parser.add_argument("-w_i", "--initializer", type=str, default="random", choices=["random", "xavier"],help="Weight initialization strategy.")
    # parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs to train for.")
    # parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size.")
    # parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate.")


    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment_1", help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="puspak", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Which dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Which loss function to use.")
    parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Which optimizer to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.00031403208880736246, help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=0.6, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.6, help="Beta used by rmsprop optimizer.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.00000664420310002881, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices=["random", "xavier"], help="Which initialization strategy to use.")

    parser.add_argument("-nhl", "--num_layers", type=int, default=5, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer.") 
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["identity", "sigmoid", "tanh", "relu"], help="Which activation functions to use.")
    

    args = parser.parse_args()

    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    main(args)