import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.neural_network import MLPClassifier
from ucimlrepo import fetch_ucirepo
from keras.datasets import mnist
from lib.utils import scale_dataset, labels_to_one_hot, flatten, shuffle, loss, accuracy, chars_to_indices


def train_and_test_repeat(
                repeat: int,
                *args, **kwargs) -> list:
    
    loss_train_vals = []
    acc_train_vals = []

    loss_test_vals = []
    acc_test_vals = []

    for _ in range(repeat):

        [loss_train, acc_train], [loss_test, acc_test] = train_and_test(*args, **kwargs)

        loss_train_vals.append(loss_train)
        acc_train_vals.append(acc_train)
        loss_test_vals.append(loss_test)
        acc_test_vals.append(acc_test)

    # mean, std_dev, min, max
    return {
        'loss_train': [np.mean(loss_train_vals), np.std(loss_train_vals), np.min(loss_train_vals), np.max(loss_train_vals)],
        'acc_train': [np.mean(acc_train_vals), np.std(acc_train_vals), np.min(acc_train_vals), np.max(acc_train_vals)],
        'loss_test': [np.mean(loss_test_vals), np.std(loss_test_vals), np.min(loss_test_vals), np.max(loss_test_vals)],
        'acc_test': [np.mean(acc_test_vals), np.std(acc_test_vals), np.min(acc_test_vals), np.max(acc_test_vals)]
    }

def train_and_test(x: np.ndarray,
                   y: np.ndarray,
                   inner_layers: list,
                   num_train: int,
                   learning_rate: float,
                   batch_size: int,
                   epochs: int) -> list:

    x, y = shuffle(x, y)
    
    x = np.array(x, dtype=np.float32)
    y = labels_to_one_hot(y)

    x_train, x_test = x[:num_train], x[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]

    vars_to_keep = ((x_train.max(axis=0) - x_train.min(axis=0)) > 0)
    x_train = x_train[:, vars_to_keep]
    x_test = x_test[:, vars_to_keep]

    x_train = scale_dataset(x_train, 0, 1)
    x_test = scale_dataset(x_test, 0, 1)

    input_size = x_train.shape[1]
    num_classes = y_train.shape[1]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=inner_layers,
        solver='sgd',
        alpha=0.0,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=epochs,
        nesterovs_momentum=False,
        early_stopping=True,
        n_iter_no_change=50
    )

    mlp.fit(x_train,
            y_train)
    
    y_train_pred = mlp.predict(x_train)
    y_test_pred = mlp.predict(x_test)

    # [[loss_train, acc_train], [loss_test, acc_test]]
    return [[loss(y_train, y_train_pred), accuracy(y_train, y_train_pred)],
            [loss(y_test, y_test_pred), accuracy(y_test, y_test_pred)]]


if __name__ == '__main__':

    ### Breast Cancer Dataset ###
    print('Breast Cancer Dataset')
    print(train_and_test_repeat(
                10,
                *load_breast_cancer(return_X_y=True),
                inner_layers=[64, 8, 64],
                num_train=350,
                learning_rate=0.05,
                batch_size=350,
                epochs=800))
    
    ### Wine Dataset ###
    print('Wine Dataset')
    print(train_and_test_repeat(
                10,
                *load_wine(return_X_y=True),
                inner_layers=[64, 8, 64],
                num_train=60,
                learning_rate=0.05,
                batch_size=60,
                epochs=800))
    
    ### Iris Dataset ###
    print('Iris Dataset')
    print(train_and_test_repeat(
                10,
                *load_iris(return_X_y=True),
                inner_layers=[64, 8, 64],
                num_train=75,
                learning_rate=0.05,
                batch_size=75,
                epochs=800))

    ### Ionosphere Dataset ###
    print('Ionosphere Dataset')
    ionosphere = fetch_ucirepo(id=52)
    print(train_and_test_repeat(
                10,
                np.array(ionosphere.data.features.values),
                chars_to_indices(ionosphere.data.targets.values),
                inner_layers=[64, 8, 64],
                num_train=175,
                learning_rate=0.01,
                batch_size=64,
                epochs=800))
    
    ### PIMA Dataset ###
    print('PIMA Dataset')
    diabetes = pd.read_csv('data/diabetes.csv')
    print(train_and_test_repeat(
                10,
                diabetes.values[:, :-1],
                diabetes.values[:, -1],
                inner_layers=[64, 8, 64],
                num_train=384,
                learning_rate=0.01,
                batch_size=32,
                epochs=800))
