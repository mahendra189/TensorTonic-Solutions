import numpy as np

def normalize_3d(v: list) -> np.ndarray:
    """
    Normalize one 3D vector or every row in a batch to unit length.
    """
    v = np.asarray(v, dtype=float)

    if v.ndim == 1:
        norm = np.linalg.norm(v)

        if norm == 0:
            return np.zeros_like(v)

        return v / norm

    norm = np.linalg.norm(v, axis=1, keepdims=True)

    return np.divide(
        v,
        norm,
        out=np.zeros_like(v),
        where=norm != 0
    )