import numpy as np
def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    if x.ndim == 2:
        # x: (N, D)
        axes = (0,)
        shape = (1, x.shape[1])

    elif x.ndim == 4:
        # x: (N, C, H, W)
        axes = (0, 2, 3)
        shape = (1, x.shape[1], 1, 1)

    else:
        raise ValueError("Expected 2D or 4D input")

    mean = x.mean(axis=axes, keepdims=True)

    var = ((x - mean) ** 2).mean(
        axis=axes,
        keepdims=True
    )

    x_hat = (x - mean) / np.sqrt(var + eps)

    gamma = gamma.reshape(shape)
    beta = beta.reshape(shape)

    out = gamma * x_hat + beta

    return out