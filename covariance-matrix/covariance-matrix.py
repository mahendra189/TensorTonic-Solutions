import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    # u = np.mean(X,axis=0)
    # X_centered = X - u
    # N = len(X)
    # return X_centered.T* X_centered/(N-1)
    if X is None or len(X) == 0:
        return None

    X = np.array(X)

    # Need at least 2 samples
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    return np.atleast_2d(np.cov(X,rowvar=False))
    