import numpy as np






np.random.seed(0)

class NeuralNetwork:
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
                 weight_decay=0.01,
                 wandb=None):
        
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

        self.wandb = wandb

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
        


        for epoch in range(epochs):
            # Forward pass
            for batch in range(no_of_batches):
                X_train_batch = X_train[batch * batch_size : (batch + 1) * batch_size]
                y_train_batch = y_train[batch * batch_size : (batch + 1) * batch_size]

                output = self.forward(X_train_batch)
                weight_grad_offsets = [0 for index in range(len(self.weights))]
                bias_grad_offsets = [0 for index in range(len(self.biases))]
                # Backward pass
                # if (self.optimizer != 'nag'):
                #     (d_w_array, d_b_array) = self.backward(X_train_batch, y_train_batch, weight_offsets=weight_grad_offsets, bias_offsets=bias_grad_offsets, learning_rate=learning_rate)
                (d_w_array, d_b_array) = self.backward(X_train_batch, y_train_batch, weight_offsets=weight_grad_offsets, bias_offsets=bias_grad_offsets, learning_rate=learning_rate) 
                # weight_gradient_history_list.append(d_w_array)
                # bias_gradient_history_list.append(d_b_array)

                
                (self.weights, self.biases) = self.optimizer.optimize_params(
                    self.weights, self.biases, d_w_array, d_b_array, self.beta_optimizer, self.beta1_optimizer, self.beta2_optimizer, learning_rate, self.weight_decay, self.epsilon_for_optimizer, self, X_train_batch, y_train_batch
                )
              

            # Compute loss
            output = self.forward(X_train)
            train_loss = self.loss_function.loss(output, y_train)

            validation_output = self.forward(X_val)
            validation_loss = self.loss_function.loss(validation_output, y_val)

            train_accuracy = self.accuracy(X_train, y_train)
            validation_accuracy = self.accuracy(X_val, y_val)

            if (self.wandb != None):
                self.wandb.log(
                    {
                        "epoch": epoch,
                        "train_accuracy": train_accuracy,
                        "train_loss": train_loss,
                        "validation_accuracy": validation_accuracy,
                        "validation_loss": validation_loss,
                    }
                )

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

