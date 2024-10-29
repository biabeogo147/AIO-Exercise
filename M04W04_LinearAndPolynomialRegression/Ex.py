import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def r2score(y_pred, y_target):
    rss = np.sum((y_pred - y_target) ** 2)
    tss = np.sum((y_target - y_target.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2


def compute_loss(y_pred, y_target):
    loss = np.mean((y_pred - y_target) ** 2) / 2
    return loss


def create_polynomial_features(X, degree=2):
    X_mem = []
    for X_sub in X.T:
        X_new = X_sub
        for d in range(2, degree + 1):
            X_new = np.c_[X_new, np.power(X_sub, d)]
        X_mem.extend(X_new.T)
    return np.c_[X_mem].T


class CustomLinearRegression:
    def __init__(self, X_data, y_target, learning_rate=0.01, num_epochs=10000):
        self.num_samples = X_data.shape[0]
        self.X_data = np.c_[np.ones((self.num_samples, 1)), X_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.theta = np.random.randn(self.X_data.shape[1], 1)
        self.losses = []

    def compute_gradient(self, y_pred, y_target):
        gradients = self.X_data.T.dot(y_pred - y_target) / self.num_samples
        return gradients

    def predict(self, X_data):
        y_pred = X_data.dot(self.theta)
        return y_pred

    def fit(self):
        for epoch in range(self.num_epochs):
            y_pred = self.predict(self.X_data)

            loss = compute_loss(y_pred, self.y_target)
            self.losses.append(loss)

            gradients = self.X_data.T.dot(y_pred - self.y_target) / self.num_samples
            self.theta -= self.learning_rate * gradients

            if (epoch % 50) == 0:
                print(f'Epoch: {epoch} - Loss: {loss}')

        return {
            'loss': sum(self.losses) / len(self.losses),
            'weight': self.theta
        }


if __name__ == "__main__":
    y_pred_1 = np.array([1, 2, 3, 4, 5])
    y_1 = np.array([1, 2, 3, 4, 5])
    print("R2 Score Case 1:", r2score(y_pred_1, y_1))

    y_pred_2 = np.array([1, 2, 3, 4, 5])
    y_2 = np.array([3, 5, 5, 2, 4])
    print("R2 Score Case 2:", r2score(y_pred_2, y_2))

    X = np.array([[1, 2], [2, 3], [3, 4]])
    degree = 2
    X_poly = create_polynomial_features(X, degree)
    print(X_poly)

    df = pd.read_csv('SalesPrediction.csv')
    df = pd.get_dummies(df)
    df = df.fillna(df.mean())
    X = df[['TV', 'Radio', 'Social Media',
            'Influencer_Macro', 'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]
    y = df[['Sales']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed = scaler.transform(X_test)
    print(scaler.mean_[0])

    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_processed)
    X_test_poly = poly_features.transform(X_test_processed)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    preds = poly_model.predict(X_test_poly)
    print(r2score(y_test, preds))
