import numpy as np

def global_avg_pool(x: list) -> np.ndarray:
    """
    Returns a spatially averaged NumPy array with shape (C,) or (N, C).
    """
    # Write code here
    # print(x)
    x = np.array(x)
    # print(x.shape)
    e = x.shape[0]
    hw = x.shape[-1] * x.shape[-2]
    c = x.sum(axis=(-1,-2)) / hw
    return c
    
    
    