import numpy as np

from Ex1 import compute_loss, softmax

def compute_accuracy(y_hat, y):
    return np.mean([y_hat[i] == y[i] for i in range(len(y))])

if __name__ == "__main__":
    print(compute_loss(softmax(np.array([0.4, 0.15, 0.05, 0.4])), np.array([1, 0, 0, 0])))
    print(softmax(np.array([-1, -2, 3, 2])))
    print(compute_accuracy(y_hat=[0, 1, 3, 2, 0, 2, 1, 2], y=[0, 0, 3, 2, 1, 2, 2, 1]))