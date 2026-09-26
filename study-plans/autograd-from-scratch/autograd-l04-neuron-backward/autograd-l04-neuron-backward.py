import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = torch.sum(weights @ inputs) + bias
    y = torch.tanh(a)

    loss = upstream_gradient * ( 1 - y**2)
    dL_dxi = loss * weights
    dL_dwi = loss * inputs

    dL_db = loss
    return (y,dL_dxi,dL_dwi,dL_db)