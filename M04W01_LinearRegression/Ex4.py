import numpy as np
from matplotlib import pyplot as plt


def get_column(data, index):
    result = np.array(data)[:, index]
    return result.tolist()


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    X = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y


def initialize_params():
    # bias = 0
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    return [0, -0.01268850433497871, 0.004752496982185252, 0.0073796171538643845]


def predict(features, weights):
    y_hat = sum([x * w for x, w in zip(features, weights)])
    return y_hat


def compute_loss(y, y_hat):
    return (y - y_hat) ** 2


def compute_gradient_w(features_i, y, y_hat):
    return [2 * (y_hat - y) * x for x in features_i]


def update_weight(weights, dl_dweights, lr):
    return [w - lr * dl_dw for w, dl_dw in zip(weights, dl_dweights)]


def implement_linear_regression(X_feature, y_output, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()
    N = len(y_output)

    for epoch in range(epoch_max):
        print("epoch", epoch)
        for i in range(N):
            features_i = X_feature[i]
            y = y_output[i]

            y_hat = predict(features_i, weights)
            loss = compute_loss(y, y_hat)
            dl_dweights = compute_gradient_w(features_i, y, y_hat)

            weights = update_weight(weights, dl_dweights, lr)

            losses.append(loss)

    return weights, losses


if __name__ == '__main__':
    X, y = prepare_data('advertising.csv')
    W, L = implement_linear_regression(X, y)
    print(L[9999])
    # plt.plot(L[0:100])
    # plt.xlabel("# iteration")
    # plt.ylabel("Loss")
    # plt.show()