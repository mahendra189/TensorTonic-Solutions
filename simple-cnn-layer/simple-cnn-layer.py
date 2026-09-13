import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    N, C_in, H, W_in = x.shape
    C_out, _, K, _ = W.shape

    H_out = H - K + 1
    W_out = W_in - K + 1

    out = np.zeros((N, C_out, H_out, W_out), dtype=float)

    for n in range(N):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i:i+K, j:j+K]
                    out[n, oc, i, j] = np.sum(patch * W[oc]) + b[oc]

    return out