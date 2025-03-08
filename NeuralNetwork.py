import numpy as np
from keras.datasets import fashion_mnist

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
                 epsilon_for_optimizer=0.000001):
        
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
        d_z = self.a_array[-1] - y

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

        
    
    def train(self, X_train, y_train, epochs=10, batch_size=16, learning_rate=0.01):
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
                        self.weights[index] -= effective_learning_rate_weights[index] * ((self.beta1_optimizer * param_update_weights[index]) + ((1 - self.beta1_optimizer) * d_w_array[index]) / (1 - (self.beta1_optimizer ** t)))
                        self.biases[index] -= effective_learning_rate_biases[index] * ((self.beta1_optimizer * param_update_biases[index]) + ((1 - self.beta1_optimizer) * d_b_array[index]) / (1 - (self.beta1_optimizer ** t)))
                    else:
                        self.weights[index] -= effective_learning_rate_weights[index] * param_update_weights[index]
                        self.biases[index] -= effective_learning_rate_biases[index] * param_update_biases[index]

                if (self.optimizer == 'adam' or self.optimizer == 'nadam'): # adjusting back the bias correction term so that we can have the actual term in the later iterations
                    learning_rate_update_weights = [learning_rate_update_weights[index] * (1 - (self.beta2_optimizer ** t)) for index in range(len(learning_rate_update_weights))]
                    learning_rate_update_biases = [learning_rate_update_biases[index] * (1 - (self.beta2_optimizer ** t)) for index in range(len(learning_rate_update_biases))]

                    param_update_weights = [param_update_weights[index] * (1 - (self.beta1_optimizer ** t)) for index in range(len(param_update_weights))]
                    param_update_biases = [param_update_biases[index] * (1 - (self.beta1_optimizer ** t)) for index in range(len(param_update_biases))]

                    t += 1


            # Compute loss
            output = self.forward(X_train)
            loss = self.loss_function(output, y_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
    
    def predict(self, X):
        # Predict class labels for input data X
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def accuracy(self, X, y):
        # Calculate accuracy of the model
        predictions = self.predict(X)
        return np.mean(predictions == np.argmax(y, axis=1))

# Example usage with Fashion MNIST:
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

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(output, target):
    # Cross-entropy loss function
    m = target.shape[0]
    return -np.sum(target * np.log(output + 1e-8)) / m

def sgd(weight_history_list, bias_history_list, beta_optimizer, learning_rate):
    return (learning_rate, weight_history_list[-1], bias_history_list[-1])


# Example usage
if __name__ == "__main__":
    # Assuming you have the Fashion MNIST dataset loaded as X_train, y_train
    # X_train shape: (num_samples, 784)
    # y_train shape: (num_samples, 10)
    
    # Preprocess the data (normalize and one-hot encode labels)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train_new = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    y_train_new = np.array([[0 if index != value else 1 for index in range(10)] for value in y_train])
    
    
    X_train = preprocess_data(x_train_new)
    
    
    # Initialize the neural network with 784 input, 2 hidden layers (128, 64), and 10 output
    nn = VanillaNeuralNetwork(input_size=784, 
                              hidden_layer_sizes=[128, 128, 128], 
                              output_size=10, 
                              activation_function=Relu(), 
                              output_function=softmax,
                              loss_function=cross_entropy_loss,
                              initializer=xavier_init,
                              optimizer='adam',
                              beta_optimizer=0.6,
                              beta1_optimizer=0.9,
                              beta2_optimizer=0.999,
                              epsilon_for_optimizer=0.000001)
    
    # Train the model
    nn.train(X_train, y_train_new, epochs=10, batch_size=32, learning_rate=0.001)
    
    # Evaluate the model's accuracy
    accuracy = nn.accuracy(X_train, y_train_new)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    
    x_train_new = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    y_train_new = np.array([[0 if index != value else 1 for index in range(10)] for value in y_train])
