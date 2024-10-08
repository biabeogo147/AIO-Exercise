import numpy as np


def get_column(data, index):
    result = np.array(data)[:, index]
    return result


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    X = np.array([tv_data, radio_data, newspaper_data])
    y = np.array(sales_data)
    return X, y


if __name__ == '__main__':
    X, y = prepare_data('advertising.csv')
    list = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
    print(list)
