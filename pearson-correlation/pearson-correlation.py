import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X,dtype=float)
    mean = np.mean(X,axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered)/ (X.shape[0] -1)
    std = np.std(X,axis=0,ddof=1)
    corr = cov / np.outer(std,std)
    return corr
    
    
    