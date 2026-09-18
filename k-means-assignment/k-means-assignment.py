import numpy as np

def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    points = np.asarray(points)
    centroids = np.asarray(centroids)

    distances = np.linalg.norm(
        points[:, None, :] - centroids[None, :, :],
        axis=2
    )

    return np.argmin(distances, axis=1).tolist()