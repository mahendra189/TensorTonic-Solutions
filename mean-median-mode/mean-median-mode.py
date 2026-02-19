import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    X = np.array(x)
    freq = Counter(x)
    return np.mean(X),np.median(X),min(freq)