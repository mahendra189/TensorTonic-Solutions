import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    # Write code here
    fpr = np.asarray(fpr,dtype=float)
    tpr = np.asarray(tpr,dtype=float)
    return float(np.trapezoid(tpr,fpr))