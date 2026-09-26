import numpy as np

def contrastive_loss(
    a: list,
    b: list,
    y: list,
    margin: float = 1.0,
    reduction: str = "mean"
) -> float:

    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    # Distance for each pair
    if a.ndim == 1:
        d = np.linalg.norm(a - b)
    else:
        d = np.linalg.norm(a-b,axis=1)

    # Contrastive loss
    l = (
        y * d**2
        + (1 - y) * np.maximum(0, margin - d)**2
    )

    if reduction == "mean":
        return float(np.mean(l))

    elif reduction == "sum":
        return float(np.sum(l))

    elif reduction == "none":
        return float(l)

    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")