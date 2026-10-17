import torch

def tanh_forward_backward(x: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: tanh output and input gradient.
    """
    y = torch.tanh(x)
    dL_dx = upstream_gradient* ( 1 - y**2)
    return y,dL_dx
