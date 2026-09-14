import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.array(x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return x*sigmoid(x)
    