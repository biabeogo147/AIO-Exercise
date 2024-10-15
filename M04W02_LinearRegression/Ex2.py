import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def visualize_data():
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    unique_years = df['year'].unique()
    for year in unique_years:
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        year_month_day = pd.DataFrame({'date': dates})
        year_month_day['year'] = year_month_day['date'].dt.year
        year_month_day['month'] = year_month_day['date'].dt.month
        year_month_day['day'] = year_month_day['date'].dt.day

        merged_data = pd.merge(year_month_day, df, on=['year', 'month', 'day'], how='left')

        plt.figure(figsize=(10, 6))
        plt.plot(merged_data['date_x'], merged_data['close'])
        plt.title(f'Bitcoin Closing Prices - {year}')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def predict(X, w, b):
    return X.dot(w) + b


def gradient(y_hat, y, x):
    loss = y_hat - y
    dw = x.T.dot(loss) / len(y)
    db = np.sum(loss) / len(y)
    cost = np.sum(loss ** 2) / (2 * len(y))
    return dw, db, cost


def update_weight(w, b, lr, dw, db):
    w_new = w - lr * dw
    b_new = b - lr * db
    return w_new, b_new


def linear_regression_vectorized(X, y, learning_rate=0.01, num_iterations=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    losses = []

    for _ in range(num_iterations):
        y_hat = predict(X, w, b)
        dw, db, cost = gradient(y_hat, y, X)
        w, b = update_weight(w, b, learning_rate, dw, db)
        losses.append(cost)

    return w, b, losses


if __name__ == "__main__":
    df = pd.read_csv('./BTC-Daily.csv')
    df = df.drop_duplicates()

    df['date'] = pd.to_datetime(df['date'])
    # date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())
    # print(date_range)
    # visualize_data()

    scalar = StandardScaler()
    df["Standardized_Close_Prices"] = scalar.fit_transform(df["close"].values.reshape(-1, 1))
    df["Standardized_Open_Prices"] = scalar.fit_transform(df["open"].values.reshape(-1, 1))
    df["Standardized_High_Prices"] = scalar.fit_transform(df["high"].values.reshape(-1, 1))
    df["Standardized_Low_Prices"] = scalar.fit_transform(df["low"].values.reshape(-1, 1))
    df['date_str'] = df['date'].dt.strftime('%Y%m%d%H%M%S')
    df['NumericalDate'] = pd.to_numeric(df['date_str'])
    df.drop(columns=['date_str'], inplace=True)

    X = df[["Standardized_Open_Prices", "Standardized_High_Prices", "Standardized_Low_Prices"]]
    y = df["Standardized_Close_Prices"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    b = 0
    w = np.zeros(X_train.shape[1])
    lr = 0.01
    epochs = 200

    _, _, losses = linear_regression_vectorized(X_train.values, y_train.values, lr, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Function during Gradient Descent')
    plt.show()

    y_pred = predict(X_test, w, b)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae = np.mean(np.abs(y_pred - y_test))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    y_train_pred = predict(X_train, w, b)
    train_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_pred)

    print("Root Mean Square Error (RMSE):", round(rmse, 4))
    print("Mean Absolute Error (MAE):", round(mae, 4))
    print("Training Accuracy (R-squared):", round(train_accuracy, 4))
    print("Testing Accuracy (R-squared):", round(test_accuracy, 4))
