import numpy as np


def df_w(W):
    """
    Thực hiện tính gradient của dw1 và dw2
    Arguments:
    W -- np.array [w1, w2]
    Returns:
    dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    """
    dw = np.array([0.2 * W[0], 4 * W[1]])
    return dw