import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T = np.asarray(T,dtype=float)
    points = np.asarray(points,dtype=float)

    if points.ndim == 1:
        p = np.append(points,1)
        result = T @ p
        return result[:3]
    ones = np.ones((points.shape[0],1))
    homogenous_points = np.hstack([points,ones])
    results = homogenous_points @ T.T
    return results[:,:3]

    
    