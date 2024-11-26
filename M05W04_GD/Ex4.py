import numpy as np
from util import df_w


def rms_prop(W, V, S, dW, lr, epoch, drV=0.9, drS=0.999):
    """
    Thực hiện thuật toán Gradient Descent để update w1 và w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    lr -- float: learning rate
    dr -- float: decay rate
    Returns:
    W -- np.array: [w1, w2] w1 và w2 sau khi đã update
    """
    V = drV * V + (1 - drV) * dW
    S = drS * S + (1 - drS) * (dW ** 2)
    Vcorr = V / (1 - (drV ** epoch))
    Scorr = S / (1 - (drS ** epoch))
    W = W - lr * Vcorr / np.sqrt(Scorr + 1e-6)
    return W, V, S


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
    S, V = np.zeros_like(W), np.zeros_like(W)
    results = [W]

    for epoch in range(epochs):
        dW = df_w(W)
        W, V, S = optimizer(W, V, S, dW, lr, epoch + 1)
        results.append(W)

    return results


if __name__ == "__main__":
    lr = 0.2
    epochs = 30
    results = train_pl(rms_prop, lr, epochs)
    for i, W in enumerate(results):
        print(f"Epoch {i}: {W}")
