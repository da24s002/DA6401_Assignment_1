import numpy as np
import argparse

import wandb


import os
# os.environ["WANDB_MODE"] = "offline"
os.environ['WANDB_TIMEOUT'] = '60'

from NeuralNetwork import NeuralNetwork

wandb.login()



from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

import argparse

from optimizers import SGD, Momentum, Rmsprop, Nag, Adam, Nadam
from activations import Sigmoid, Relu, Tanh
from loss import CrossEntropyLoss, SquaredErrorLoss
from initilizers import random_init, xavier_init
from output import softmax

def preprocess_data(X):
    # Normalize the input data (scale to range [0, 1])
    X = X / 255.0

    return X



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

optimizer_dict = {
    "sgd": SGD(),
    "momentum": Momentum(),
    "nag": Nag(),
    "rmsprop": Rmsprop(),
    "adam": Adam(),
    "nadam": Nadam()
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


# wandb.init(entity="da24s002-indian-institute-of-technology-madras", project="DA6401_Assignment_1")

def main(args):


    hidden_layer_size = args.hidden_size
    hidden_layers = args.num_layers
    hidden_layer_sizes = [hidden_layer_size] * hidden_layers
    activation_function = activation_dict[args.activation]
    weight_decay = args.weight_decay
    optimizer_name = args.optimizer
    initializer = weight_initialization_dict[args.weight_init]
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    if (optimizer_name == "momentum"):
        beta_optimizer = args.momentum
    else:
        beta_optimizer = args.beta
    beta1_optimizer = args.beta1
    beta2_optimizer = args.beta2
    epsilon_for_optimizer = args.epsilon

    optimizer = optimizer_dict[optimizer_name]

    loss_function = loss_function_dict[args.loss]
    x_train_new = x_train_dict[args.dataset]
    y_train_new = y_train_dict[args.dataset]

    x_val_new = x_val_dict[args.dataset]
    y_val_new = y_val_dict[args.dataset]

    x_test_new = x_test_dict[args.dataset]
    y_test_new = y_test_dict[args.dataset]

    wandb.run.name = f"hls_{hidden_layer_size}_hl_{hidden_layers}_bs_{batch_size}_opt_{optimizer_name}_act_{args.activation}_id_test_data_{wandb.run.id}"

    


    # Initialize the neural network with 784 input, 2 hidden layers (128, 64), and 10 output
    nn = NeuralNetwork(input_size=784, 
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
                    weight_decay=weight_decay,
                    wandb=wandb)
    
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

    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment_1", help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="puspak", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Which dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Which loss function to use.")
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Which optimizer to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.00011708255968970672, help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=0.6, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.6, help="Beta used by rmsprop optimizer.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0000026013264510596, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "xavier"], help="Which initialization strategy to use.")

    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer.") 
    parser.add_argument("-a", "--activation", type=str, default="tanh", choices=["identity", "sigmoid", "tanh", "relu"], help="Which activation functions to use.")
    

    args = parser.parse_args()
    print(args.optimizer)

    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    main(args)