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
from dataset_loader import dataset_loader





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


def prepare_and_load_dataset(dataset_name):
    (x_train_new_dataset, 
    y_train_new_dataset, 
    x_val_new_dataset, 
    y_val_new_dataset, 
    x_test_new_dataset, 
    y_test_new_dataset) = dataset_loader(dataset_name)

    return (x_train_new_dataset, y_train_new_dataset, x_val_new_dataset, y_val_new_dataset, x_test_new_dataset, y_test_new_dataset)




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
    
    (x_train_new, y_train_new, x_val_new, y_val_new, x_test_new, y_test_new) = prepare_and_load_dataset(args.dataset)


    wandb.run.name = f"hls_{hidden_layer_size}_hl_{hidden_layers}_bs_{batch_size}_opt_{optimizer_name}_act_{args.activation}_id_{wandb.run.id}"

    


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
    print("Test accuracy:",test_accuracy)
    labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=y_test_compressed, preds=test_output_compressed, class_names=labels)})
    


# main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    default_dict = {
        "activation":"relu",
        "batch_size":64,
        "beta":0.19006430071073768,
        "beta1":0.1050081262400274,
        "beta2":0.8731230311649747,
        "epochs":10,
        "hidden_size":128,
        "learning_rate":0.00021970549167311983,
        "momentum":0.2053344640014278,
        "num_layers":5,
        "optimizer":"nadam",
        "weight_decay":0.00000103176519285397,
        "weight_init":"xavier",
        "wandb_project": "DA6401_Assignment_1",
        "wandb_entity": "puspak",
        "loss": "cross_entropy",
        "dataset": "fashion_mnist",
        "epsilon": 0.000001,
    }

    
    parser.add_argument("-wp", "--wandb_project", type=str, default=default_dict["wandb_project"], help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default=default_dict["wandb_entity"], help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, default=default_dict["dataset"], choices=["mnist", "fashion_mnist"], help="Which dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=default_dict["epochs"], help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=default_dict["batch_size"], help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type=str, default=default_dict["loss"], choices=["mean_squared_error", "cross_entropy"], help="Which loss function to use.")
    parser.add_argument("-o", "--optimizer", type=str, default=default_dict["optimizer"], choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Which optimizer to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=default_dict["learning_rate"], help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=default_dict["momentum"], help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=default_dict["beta"], help="Beta used by rmsprop optimizer.")
    parser.add_argument("-beta1", "--beta1", type=float, default=default_dict["beta1"], help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=default_dict["beta2"], help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=default_dict["epsilon"], help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=default_dict["weight_decay"], help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default=default_dict["weight_init"], choices=["random", "xavier"], help="Which initialization strategy to use.")

    parser.add_argument("-nhl", "--num_layers", type=int, default=default_dict["num_layers"], help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=default_dict["hidden_size"], help="Number of hidden neurons in a feedforward layer.") 
    parser.add_argument("-a", "--activation", type=str, default=default_dict["activation"], choices=["identity", "sigmoid", "tanh", "relu"], help="Which activation functions to use.")
    

    args = parser.parse_args()
    print(args.optimizer)

    # wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    wandb.init(project=args.wandb_project)
    main(args)