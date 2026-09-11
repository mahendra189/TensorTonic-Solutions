import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    # Write code here
    y_true = np.asarray(y_true,dtype=float)
    y_pred = np.asarray(y_pred,dtype=float)
    y_mean = np.mean(y_true)
    rss = np.sum((y_true-y_pred)**2)
    tss = np.sum((y_true-y_mean)**2)
    rs = (1 - rss/tss)
    if tss == 0:
        return 1.0 if rss == 0 else 0.0
    return float(rs)