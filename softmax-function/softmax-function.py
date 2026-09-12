import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)