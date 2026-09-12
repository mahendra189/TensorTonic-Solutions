import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    # Write code here
    y = np.asarray(y,dtype=int)
    if num_classes is None:
        num_classes = np.max(y) +1
    n = np.zeros((y.size,num_classes),dtype=float)
    n[np.arange(len(y)),y] = 1
    return n