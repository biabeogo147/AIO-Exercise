from matplotlib import pyplot as plt

from M04W01_LinearRegression.Ex1 import prepare_data


def initialize_params():
    w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)
    # w1 = np.random.randn()
    # w2 = np.random.randn()
    # w3 = np.random.randn()
    # b = np.random.randn()
    return w1, w2, w3, b


def predict(x1, x2, x3, w1, w2, w3, b):
    return w1 * x1 + w2 * x2 + w3 * x3 + b


def compute_loss_mse(y, y_hat):
    return (y - y_hat) ** 2


def compute_loss_mae(y, y_hat):
    return abs(y - y_hat)


def compute_gradient_wi(xi, y, y_hat):
    return -2 * xi * (y - y_hat)


def compute_gradient_b(y, y_hat):
    return -2 * (y - y_hat)


def update_weight_wi(wi, dl_dwi, lr):
    return wi - lr * dl_dwi


def update_weight_b(b, dl_db, lr):
    return b - lr * dl_db


def implement_linear_regression(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []
    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for epoch in range(epoch_max):
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            loss = compute_loss_mse(y, y_hat)

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            losses.append(loss)

    return w1, w2, w3, b, losses


if __name__ == '__main__':
    y = predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
    print(y)

    l = compute_loss_mse(y_hat=1, y=0.5)
    print(l)

    g_wi = compute_gradient_wi(xi=1.0, y=1.0, y_hat=0.5)
    print(g_wi)

    g_b = compute_gradient_b(y=2.0, y_hat=0.5)
    print(g_b)

    after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
    print(after_wi)

    after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
    print(after_b)

    X, y = prepare_data('advertising.csv')
    (w1, w2, w3, b, losses) = implement_linear_regression(X, y)
    plt.plot(losses[:100])
    plt.xlabel("# iteration ")
    plt.ylabel(" Loss ")
    plt.show()
    print(w1, w2, w3)

    tv = 19.2
    radio = 35.9
    newspaper = 51.3
    X, y = prepare_data('advertising.csv')
    (w1, w2, w3, b, losses) = implement_linear_regression(X, y, epoch_max=50, lr=1e-5)
    sales = predict(tv, radio, newspaper, w1, w2, w3, b)
    print(f'predicted sales is {sales}')

    l = compute_loss_mae(y_hat=1, y=0.5)
    print(l)
