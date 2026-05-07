import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    if np.linalg.det(A) == 0:
        return None
    return np.linalg.inv(A)
