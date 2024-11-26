import numpy as np
from util import df_w


def sgd(W, dW, lr):
    """
    Thực hiện thuật toán Gradient Descent để update w1 và w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    lr -- float: learning rate
    Returns:
    W -- np.array: [w1, w2] w1 và w2 sau khi đã update
    """
    W = W - lr * dW
    return W


def train_pl(optimizer, lr, epochs):
    """
    Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán được truyền vào từ optimizer
    Arguments:
    optimizer -- function thực hiện thuật toán optimization cụ thể
    lr -- float: learning rate
    epochs -- int: số lượng lần (epoch) lặp để tìm điểm minimum
    Returns:
    results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
    """
    W = np.array([-5, -2], dtype=np.float32)
    results = [W]

    for _ in range(epochs):
        dW = df_w(W)
        W = optimizer(W, dW, lr)
        results.append(W)

    return results


if __name__ == "__main__":
    lr = 0.4
    epochs = 30
    results = train_pl(sgd, lr, epochs)
    for i, W in enumerate(results):
        print(f"Epoch {i}: {W}")