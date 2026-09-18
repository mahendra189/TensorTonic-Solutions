import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = len(X)

    # Pairwise Euclidean distance matrix
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    scores = []

    for i in range(n):
        own_cluster = labels[i]

        # Points in the same cluster, excluding itself
        same = labels == own_cluster
        same[i] = False

        if np.any(same):
            a = distances[i, same].mean()
        else:
            # Singleton cluster
            scores.append(0.0)
            continue

        # Mean distance to every other cluster
        b = np.inf

        for cluster in np.unique(labels):
            if cluster == own_cluster:
                continue

            other = labels == cluster
            cluster_distance = distances[i, other].mean()
            b = min(b, cluster_distance)

        scores.append((b - a) / max(a, b))

    return float(np.mean(scores))