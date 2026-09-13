import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    # Write code here
    k = np.array(k)
    pmf = (1-p)**(k-1)*p
    print(pmf)
    mean = 1/(p)
    return {
        "pmf":pmf,
        "mean":mean
    }