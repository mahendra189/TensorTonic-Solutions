import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.asarray(x)
    # if x >= 0:
    #     return x
    # else:
    #     return x*alpha
    return np.where(x >=0 , x,x*alpha)