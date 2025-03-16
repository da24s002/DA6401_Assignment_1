from keras.datasets import fashion_mnist
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

import numpy as np

def preprocess_data(X):
    # Normalize the input data (scale to range [0, 1])
    X = X / 255.0

    return X


def dataset_loader(dataset_name):
    if (dataset_name == "fashion_mnist"):
        (x_train_dataset, y_train_dataset), (x_test_dataset, y_test_dataset) = fashion_mnist.load_data()
        x_train_dataset, x_val_dataset, y_train_dataset, y_val_dataset = train_test_split(x_train_dataset, y_train_dataset, train_size=0.9, shuffle=True, random_state=42)
    elif (dataset_name == "mnist"):
        (x_train_dataset, y_train_dataset), (x_test_dataset, y_test_dataset) = mnist.load_data()
        x_train_dataset, x_val_dataset, y_train_dataset, y_val_dataset = train_test_split(x_train_dataset, y_train_dataset, train_size=0.9, shuffle=True, random_state=42)
    

    x_train_new_dataset = x_train_dataset.reshape(x_train_dataset.shape[0], x_train_dataset.shape[1]*x_train_dataset.shape[2])
    y_train_new_dataset = np.array([[0 if index != value else 1 for index in range(10)] for value in y_train_dataset])
    x_train_new_dataset = preprocess_data(x_train_new_dataset)

    x_val_new_dataset = x_val_dataset.reshape(x_val_dataset.shape[0], x_val_dataset.shape[1]*x_val_dataset.shape[2])
    y_val_new_dataset = np.array([[0 if index != value else 1 for index in range(10)] for value in y_val_dataset])
    x_val_new_dataset = preprocess_data(x_val_new_dataset)

    x_test_new_dataset = x_test_dataset.reshape(x_test_dataset.shape[0], x_test_dataset.shape[1]*x_test_dataset.shape[2])
    y_test_new_dataset = np.array([[0 if index != value else 1 for index in range(10)] for value in y_test_dataset])
    x_test_new_dataset = preprocess_data(x_test_new_dataset)


    return (x_train_new_dataset, y_train_new_dataset, x_val_new_dataset, y_val_new_dataset, x_test_new_dataset, y_test_new_dataset)
    
    