import numpy as np
import torch

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """

    y_true = torch.tensor(y_true,dtype=torch.float)
    y_pred = torch.tensor(y_pred,dtype=torch.float)
    # Write code here
    error = y_pred - y_true
    abs_error = torch.abs(error)

    loss = torch.where(
        abs_error <= delta,0.5 * error ** 2,
        delta * ( abs_error - 0.5 * delta)
    )
    return loss.mean().item()