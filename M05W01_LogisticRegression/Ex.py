import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(dataset_path):
    df = pd.read_csv(
        dataset_path,
        index_col='PassengerId'
    )
    dataset_arr = df.to_numpy().astype(np.float64)
    X, y = dataset_arr[:, :-1], dataset_arr[:, -1]
    return X, y


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=2, is_shuffle=True):
    X_train, X_val, y_train, y_val = train_test_split(
        X_b, y,
        test_size=val_size,
        random_state=random_state,
        shuffle=is_shuffle
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    dot_product = np.dot(X, theta)
    y_hat = sigmoid(dot_product)
    return y_hat


def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()


def compute_gradient(X, y, y_hat):
    return np.dot(X.T, (y_hat - y)) / y.size


def update_theta(theta, gradient, lr):
    return theta - lr * gradient


def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta).round()
    acc = (y_hat == y).mean()
    return acc


def compute_accuracy_new(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    accuracy = np.mean(y_true == y_pred_rounded)
    return accuracy


if __name__ == '__main__':
    X, y = load_data('titanic_modified_dataset.csv')

    intercept = np.ones((X.shape[0], 1))
    X_b = np.concatenate((intercept, X), axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_b, y)

    normalizer = StandardScaler()
    X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
    X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
    X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

    X = [[22.3, -1.5, 1.1, 1]]
    theta = [0.1, -0.15, 0.3, -0.2]
    y_hat = predict(X, theta)
    print(y_hat)

    y = np.array([1, 0, 0, 1])
    y_hat = np.array([0.8, 0.75, 0.3, 0.95])
    loss = compute_loss(y_hat, y)
    print(loss)

    X = np.array([[1, 2], [2, 1], [1, 1], [2, 2]])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.25, 0.75, 0.4, 0.8])
    gradient = compute_gradient(X, y_true, y_pred)
    print(gradient)

    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0.85, 0.35, 0.9, 0.75])
    acc = compute_accuracy_new(y_true, y_pred)
    print(acc)

    X = np.array([[1, 3], [2, 1], [3, 2], [1, 2]])
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0.7, 0.4, 0.6, 0.85])
    gradient = compute_gradient(X, y_true, y_pred)
    print(gradient)