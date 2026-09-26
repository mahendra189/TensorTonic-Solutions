import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    p = np.array(p)
    y = np.array(y)
    dpy = (2*np.sum(p*y) + eps) / (np.sum(p) + np.sum(y) + eps)
    L = 1.0 - dpy
    return float(L)