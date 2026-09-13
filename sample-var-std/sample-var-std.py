import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.array(x)
    n = len(x)
    variance = (1/(n-1))*np.sum((x-np.mean(x))**2)
    std = np.sqrt(variance)
    return {
        "variance":float(variance),
        "standard_deviation":float(std)
    }